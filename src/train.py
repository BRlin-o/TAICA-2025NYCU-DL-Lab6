# src/train.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, itertools, torch, tqdm
from pathlib import Path
from datamodule import get_loader
from model.unet_condition import ConditionalUNet
from model.ddpm import DDPM

from utils import get_device

def main(cfg):
    device = get_device()
    loader = get_loader("data", "train.json",
                        batch=cfg.batch, shuffle=True, train=True)
    model = ConditionalUNet(cond_dim=cfg.cond_dim).to(device)
    ddpm  = DDPM(model, T=cfg.T, device=device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    global_step = 0
    for epoch in range(cfg.epochs):
        pbar = tqdm.tqdm(loader, desc=f"ep{epoch}")
        for img, y in pbar:
            img, y = img.to(device), y.to(device)
            t = torch.randint(0, cfg.T, (img.size(0),), device=device)
            loss = ddpm.p_losses(img, y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=loss.item())
            global_step += 1
        if (epoch+1) % cfg.save_every == 0:
            ckpt = Path("checkpoints")
            ckpt.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt / f"unet_ep{epoch+1}.pt")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--batch",  type=int, default=128)
    pa.add_argument("--lr",     type=float, default=3e-4)
    pa.add_argument("--T",      type=int, default=1000)
    pa.add_argument("--cond_dim", type=int, default=128)
    pa.add_argument("--save_every", type=int, default=10)
    main(pa.parse_args())