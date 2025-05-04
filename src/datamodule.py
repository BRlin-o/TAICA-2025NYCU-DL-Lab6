# src/datamodule.py
import json, random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils import get_device

N_CLASS = 24
IMG_SIZE = 64

# --- Encode / Decode -------------------------------------------------
OBJ2IDX = json.load(open(Path(__file__).parents[1] / "data/objects.json"))

def encode(labels: List[str]) -> torch.Tensor:
    """multi‑hot one‑hot (24,)"""
    v = torch.zeros(N_CLASS)
    for name in labels:
        v[OBJ2IDX[name]] = 1
    return v

# --- Dataset ---------------------------------------------------------
class ICLEVR(Dataset):
    def __init__(self, root: Path, json_file: str, train: bool = True):
        self.root = root
        self.train = train
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
        jpath = root / json_file
        self.js = json.load(open(jpath))
        if train:
            # dict : {fname: [lbls]}
            self.items = list(self.js.items())
        else:
            # list : [[lbls], ...]
            self.items = list(enumerate(self.js))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        if self.train:
            fname, lbls = self.items[idx]
            img = Image.open(self.root / "iclevr" / fname).convert("RGB")
            return self.tf(img), encode(lbls)
        else:
            _, lbls = self.items[idx]
            # 測試集沒有真實圖片，回傳全 0 image 省空間
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), encode(lbls)

# --- Dataloader helper -----------------------------------------------
def get_loader(root: str | Path,
               json_file: str,
               batch: int,
               shuffle: bool,
               train: bool):
    ds = ICLEVR(Path(root), json_file, train)
    device = get_device()
    if device == "mps":
        # MPS 不支援 pin_memory
        return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                          num_workers=4, pin_memory=False)
    else:
        # CUDA / CPU
        # pin_memory = True 會加速資料傳輸
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        # https://pytorch.org/docs/stable/cuda.html#torch.cuda.is_available
        # https://pytorch.org/docs/stable/mps.html#torch.backends.mps.is_available
        # https://pytorch.org/docs/stable/mps.html#torch.backends.mps.is_built
        return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=4, pin_memory=True)