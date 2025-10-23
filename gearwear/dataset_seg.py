
import os, csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SegDataset(Dataset):
    """
    Supervised segmentation dataset.
    Expects:
      - img_root/<relpath>
      - mask_root/<relpath>  (binary mask; >0 means wear)
    A CSV with column 'relpath' defines the list (or we scan folder).
    """
    def __init__(self, img_root, mask_root, meta_csv=None, size=256):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.size = size
        if meta_csv and os.path.exists(meta_csv):
            rows = []
            with open(meta_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r['relpath'])
            self.rels = rows
        else:
            # scan all files under img_root
            self.rels = []
            for p in self.img_root.rglob("*.*"):
                rel = str(p.relative_to(self.img_root))
                self.rels.append(rel)
        self.resize = transforms.Resize((size, size))

    def __len__(self): return len(self.rels)

    def __getitem__(self, idx):
        rel = self.rels[idx]
        img = Image.open(self.img_root / rel).convert('RGB')
        mask = Image.open(self.mask_root / rel).convert('L')
        img = self.resize(img); mask = self.resize(mask)
        img_t = transforms.functional.to_tensor(img)             # [3,H,W]
        mask_t= transforms.functional.to_tensor(mask)            # [1,H,W], 0..1
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t, rel
