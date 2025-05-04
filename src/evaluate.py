# src/evaluate.py
import argparse, torch, json
from pathlib import Path
from torchvision import transforms
from PIL import Image
from evaluator import evaluation_model
from datamodule import encode

def load_images(img_dir):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    imgs = []
    for i in sorted(Path(img_dir).glob("*.png"), key=lambda p:int(p.stem)):
        imgs.append(tf(Image.open(i).convert("RGB")))
    return torch.stack(imgs)

def main(cfg):
    labels = json.load(open(cfg.json))
    if isinstance(labels, dict):
        labels = list(labels.values())
    onehots = torch.stack([encode(l) for l in labels])

    imgs = load_images(cfg.img_dir)

    ev = evaluation_model()
    acc = ev.eval(imgs.cuda(), onehots.cuda())
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--json", required=True)
    pa.add_argument("--img_dir", required=True)
    main(pa.parse_args())