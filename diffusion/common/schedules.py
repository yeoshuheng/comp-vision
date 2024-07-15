import torch, math

def get_cosine_schedule(ts : int):
    linspace = torch.linspace(0, 1, ts + 1)
    f_t = torch.cos((linspace + 0.008) / (1 + 0.008) * math.pi/2) ** 2
    bar_alpha_t = f_t / f_t[0]
    beta_t = torch.zeros_like(bar_alpha_t)
    beta_t[1:] = (1 - bar_alpha_t[1:] / bar_alpha_t[:-1]).clamp(0, 0.999)
    alpha_t = torch.cumprod(1. - beta_t, dim=0) ** 0.5
    return alpha_t 

def get_linear_schedule(ts : int):
    beta_t = torch.linspace(1e-4, 2e-2, ts + 1)
    alpha_t = torch.cumprod(1. - beta_t, dim=0) ** 0.5
    return alpha_t