
import csv, os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

def compute_area_ratio_from_mask_tensor(mask_t):
    return float((mask_t > 0.5).sum().item()) / float(mask_t.numel())

class GradeDataset(Dataset):
    """
    For grading training/inference.
    Inputs:
      - img_root (RGB)
      - optional mask_root: if provided, we load mask and compute area_ratio from GT mask
      - meta_csv with columns: relpath,label,(depth_mm|depth_bin), (optional area_ratio)
      - Optional seg_model to auto-predict mask; if provided, overrides area_ratio by predicted mask
    We also build a 4th channel as aux:
      - If we have mask (GT or predicted), use it as aux channel
      - Else use zeros
    """
    def __init__(self, img_root, meta_csv, mask_root=None, size=224, use_depth_bins=False, seg_model=None, device='cpu'):
        self.img_root = Path(img_root)
        self.mask_root= Path(mask_root) if mask_root else None
        self.size = size
        self.use_depth_bins = use_depth_bins
        self.seg_model = seg_model
        self.device = device

        rows = []
        with open(meta_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)
        self.rows = rows

        self.resize = transforms.Resize((size, size))

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        rel = r['relpath']
        label = int(r.get('label', 0))
        depth_mm = r.get('depth_mm', '')
        depth_bin = r.get('depth_bin', '')

        img = Image.open(self.img_root / rel).convert('RGB')
        img = self.resize(img)
        img_t = transforms.functional.to_tensor(img)  # [3,H,W]

        mask_t = None
        # load GT mask if exists
        if self.mask_root is not None and (self.mask_root / rel).exists():
            m = Image.open(self.mask_root / rel).convert('L')
            m = self.resize(m)
            mask_t = transforms.functional.to_tensor(m)  # [1,H,W] 0..1
            mask_t = (mask_t > 0.5).float()

        # else run seg model if provided
        if mask_t is None and self.seg_model is not None:
            with torch.no_grad():
                x = img_t.unsqueeze(0).to(self.device)
                logits = self.seg_model(x)
                pred = torch.sigmoid(logits)
                mask_t = (pred > 0.5).float()[0]  # [1,H,W]

        if mask_t is None:
            aux = torch.zeros(1, self.size, self.size)
            area_ratio = float(r.get('area_ratio', 0.0) or 0.0)
        else:
            aux = mask_t
            area_ratio = compute_area_ratio_from_mask_tensor(mask_t)

        x4 = torch.cat([img_t, aux], dim=0)  # [4,H,W]

        y = {
            'label': torch.tensor(label, dtype=torch.long),
            'area_ratio': torch.tensor(area_ratio, dtype=torch.float32),
        }
        if self.use_depth_bins:
            y['depth_bin'] = torch.tensor(int(depth_bin) if depth_bin!='' else 0, dtype=torch.long)
        else:
            y['depth_mm'] = torch.tensor(float(depth_mm) if depth_mm!='' else 0.0, dtype=torch.float32)
        return x4, y, rel
