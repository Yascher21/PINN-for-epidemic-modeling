import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# N_POP = 200
# T = 962

class PINN_with_params(nn.Module):
    def __init__(self, layers_state, layers_param, N_POP, T):
        super().__init__()
        self.E0_raw = nn.Parameter(torch.tensor([0.0]))
        self.T = T
        
        net = []
        for i in range(len(layers_state) - 2):
            net.append(nn.Linear(layers_state[i], layers_state[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers_state[-2], layers_state[-1]))
        self.state_net = nn.Sequential(*net)

        net2 = []
        for i in range(len(layers_param) - 2):
            net2.append(nn.Linear(layers_param[i], layers_param[i + 1]))
            net2.append(nn.Tanh())
        net2.append(nn.Linear(layers_param[-2], layers_param[-1]))
        self.param_net = nn.Sequential(*net2)

        self.N_POP = N_POP

    def forward(self, t):
        y = self.state_net(t)
        E = torch.sigmoid(y[:, 0:1]) * self.N_POP
        I = torch.sigmoid(y[:, 1:2]) * self.N_POP
        R = torch.sigmoid(y[:, 2:3]) * self.N_POP

        bg = self.param_net(t)
        beta = F.softplus(bg[:, 0:1])
        sigma = F.softplus(bg[:, 1:2])
        gamma = 0.03 * self.T / 2.0 + F.softplus(bg[:, 2:3])
        E0 = 5.0 * torch.sigmoid(self.E0_raw)

        return E, I, R, beta, sigma, gamma, E0

def gradients(y, t):
    return torch.autograd.grad(
        y, t, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]

def loss_fn_params(model, t_colloc, t_data_t, I_data_t, S0, I0, R0, weights):
    (w_pde_S, w_pde_E, w_pde_I, w_pde_R,
     w_ic_E, w_ic_I, w_ic_R, w_data, w_mono_S, w_mono_R) = weights

    t_colloc.requires_grad_(True)
    E, I, R, beta, sigma, gamma, E0_train = model(t_colloc)
    S = model.N_POP - E - I - R

    dE_dt = gradients(E, t_colloc)
    dI_dt = gradients(I, t_colloc)
    dR_dt = gradients(R, t_colloc)
    dS_dt = -dE_dt - dI_dt - dR_dt

    dbeta_dt = gradients(beta, t_colloc)
    dsigma_dt = gradients(sigma, t_colloc)
    dgamma_dt = gradients(gamma, t_colloc)
    loss_smooth = 5e-8 * (dbeta_dt**2).mean() + 5e-7 * (dsigma_dt**2).mean() + 5e-7 * (dgamma_dt**2).mean()

    gamma_min_norm = 12.0
    loss_gamma_floor = torch.mean(torch.relu(gamma_min_norm - gamma) ** 2)
    sigma_min_norm = 12.0
    loss_sigma_floor = torch.mean(torch.relu(sigma_min_norm - sigma) ** 2)
    beta_min_norm = 12.0
    loss_beta_floor = torch.mean(torch.relu(beta_min_norm - beta) ** 2)

    beta_eff = beta / model.N_POP
    f1 = dS_dt + beta_eff * S * I
    f2 = dE_dt - beta_eff * S * I + sigma * E
    f3 = dI_dt - sigma * E + gamma * I
    f4 = dR_dt - gamma * I

    loss_pde_S = (f1**2).mean()
    loss_pde_E = (f2**2).mean()
    loss_pde_I = (f3**2).mean()
    loss_pde_R = (f4**2).mean()

    t0 = torch.tensor([[-1.0]], dtype=torch.float32, device=device)
    E_pred0, I_pred0, R_pred0, _, _, _, E0_pred = model(t0)

    loss_ic_E = ((E_pred0 - E0_pred) ** 2).mean()
    loss_ic_I = ((I_pred0 - I0) ** 2).mean()
    loss_ic_R = ((R_pred0 - R0) ** 2).mean()

    _, I_pred_data, _, _, _, _, _ = model(t_data_t)
    # Комбинированный лосс (логарифмический + MSE на пике для точного захвата Омикрона)
    loss_log = torch.mean((torch.log(I_pred_data + 1e-8) - torch.log(I_data_t + 1e-8)) ** 2)
    loss_mse = torch.mean((I_pred_data - I_data_t) ** 2)
    loss_data = loss_log

    mono_S_loss = torch.mean(torch.relu(dS_dt))
    mono_R_loss = torch.mean(torch.relu(-dR_dt))

    loss = (
        w_pde_S * loss_pde_S + w_pde_E * loss_pde_E + w_pde_I * loss_pde_I + w_pde_R * loss_pde_R
        + w_ic_E * loss_ic_E + w_ic_I * loss_ic_I + w_ic_R * loss_ic_R
        + w_data * loss_data
        + w_mono_S * mono_S_loss + w_mono_R * mono_R_loss
        + loss_smooth
        + 1e-2 * loss_gamma_floor  # <--- не дает gamma уйти в 0 в начале
        + 1e-2 * loss_sigma_floor  # <--- не дает sigma уйти в 0 в начале
        + 1e-2 * loss_beta_floor  # <--- не дает beta уйти в 0 в начале
    )
    return loss, loss_data.item()

def get_region_dates(df, date_col='Date', start_idx=44, end_idx=1007, n_points=200):
    """
    Извлекает даты для среза региона и строит календарную шкалу
    для n_points точек предсказания PINN (t in [-1, 1]).
    """
    # Приводим к формату datetime
    dates = pd.to_datetime(df[date_col].iloc[start_idx:end_idx].values)
    
    t_start = dates[0]
    t_end = dates[-1]
    
    # 1. Точные даты исходных измерений (для scatter/наблюдений)
    # 2. Равномерная сетка дат под размер t_test (200 точек)
    t_norm = np.linspace(-1, 1, n_points)
    tau = (t_norm + 1.0) / 2.0
    
    # Интерполяция дат: t_start + tau * (t_end - t_start)
    dates_pred = pd.to_datetime(t_start.value + tau * (t_end.value - t_start.value))
    
    return dates, dates_pred


