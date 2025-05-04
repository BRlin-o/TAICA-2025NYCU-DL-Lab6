# src/model/ddpm.py
import torch, torch.nn.functional as F
from .scheduler import cosine_beta_schedule

class DDPM:
    def __init__(self, model, T=1000, device="cuda"):
        self.model, self.T, self.device = model, T, device
        betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1 - self.alphas_cumprod)

    # -------- forward diffusion q(x_t | x_0) -------------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        acp = self.sqrt_alphas_cumprod[t][:, None, None, None]
        omacp = self.sqrt_one_minus_acp[t][:, None, None, None]
        return acp * x0 + omacp * noise, noise

    # -------- training loss -----------------------------------------
    def p_losses(self, x0, y, t):
        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.model(x_t, t, y)
        return F.mse_loss(noise_pred, noise)

    # -------- DDPM sampling -----------------------------------------
    @torch.no_grad()
    def p_sample(self, x, y, t):
        beta = 1 - self.alphas[t]
        sqrt_recip_alpha = torch.rsqrt(self.alphas[t])
        sqrt_beta = torch.sqrt(beta)

        noise_pred = self.model(x, t, y)
        model_mean = sqrt_recip_alpha[:,None,None,None] * (x - beta[:,None,None,None] * noise_pred / sqrt_beta[:,None,None,None])
        if t[0] == 0:
            return model_mean
        noise = torch.randn_like(x)
        posterior_var = beta[:,None,None,None]
        return model_mean + torch.sqrt(posterior_var) * noise