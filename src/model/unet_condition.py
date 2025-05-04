# src/model/unet_condition.py
import math, torch, torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def timestep_emb(t, dim):
    """Sinusoidal t → (B, dim)"""
    device, half = t.device, dim // 2
    exp = torch.arange(half, device=device).float() / half
    emb = t[:, None] * torch.exp(-math.log(10000) * exp)[None]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1),
            nn.GroupNorm(8, c_out),
            nn.SiLU()
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = DownBlock(c_in + c_out, c_out)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], 1)
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, base=128, num_labels=24, cond_dim=128, T=1000):
        super().__init__()
        self.T = T
        self.label_emb = nn.Linear(num_labels, cond_dim)
        self.time_emb  = nn.Linear(cond_dim, cond_dim)

        self.inc  = DownBlock(3, base)
        self.down1= DownBlock(base, base*2)
        self.down2= DownBlock(base*2, base*4)

        self.mid  = DownBlock(base*4, base*4)

        self.up2  = UpBlock(base*4, base*2)
        self.up1  = UpBlock(base*2, base)
        self.outc = nn.Conv2d(base, 3, 1)

    def add_cond(self, x, y, t):
        """FiLM：x + MLP(label+time)"""
        emb = self.label_emb(y) + self.time_emb(t)
        while emb.dim() < x.dim(): emb = emb[..., None]
        return x + emb

    def forward(self, x, t, y):
        t_emb = timestep_emb(t, self.label_emb.out_features)
        # --- Encoder
        x1 = self.inc(x)
        x2 = self.down1(F.avg_pool2d(x1, 2))
        x3 = self.down2(F.avg_pool2d(x2, 2))
        # --- Bottleneck
        xm = self.add_cond(self.mid(x3), y, t_emb)
        # --- Decoder
        xu = self.up2(xm, x2)
        xu = self.up1(xu, x1)
        return self.outc(xu)