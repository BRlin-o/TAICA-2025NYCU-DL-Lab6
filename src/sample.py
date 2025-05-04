# src/sample.py
import argparse, json, torch, torchvision.utils as vutils
from pathlib import Path
from datamodule import encode
from model.unet_condition import ConditionalUNet
from model.ddpm import DDPM

from sample import get_device

@torch.no_grad()
def sample(cfg):
    device = get_device()
    # 讀條件
    labels = json.load(open(cfg.json_path))
    if isinstance(labels, dict):  # train.json-like dict
        labels = list(labels.values())
    N = len(labels)

    model = ConditionalUNet().to(device)
    model.load_state_dict(torch.load(cfg.ckpt))
    model.eval()
    ddpm = DDPM(model)

    x = torch.randn(N, 3, 64, 64, device=device)
    y = torch.stack([encode(l) for l in labels]).to(device)

    for t in reversed(range(ddpm.T)):
        x = ddpm.p_sample(x, y, torch.full((N,), t, device=device, dtype=torch.long))

    x = (x.clamp(-1, 1) + 1) / 2          # 0‑1
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(x):
        vutils.save_image(img, out_dir / f"{i}.png")

if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--json", dest="json_path", required=True)
    pa.add_argument("--ckpt", default="checkpoints/unet_ep200.pt")
    pa.add_argument("--out",  required=True)
    sample(pa.parse_args())