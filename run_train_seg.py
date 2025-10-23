
import os, argparse, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from gearwear.seg_unet import UNet
from gearwear.dataset_seg import SegDataset

def dice_loss(pred, target, eps=1e-6):
    # pred: logits [B,1,H,W] -> sigmoid
    p = torch.sigmoid(pred)
    num = 2 * (p*target).sum(dim=(1,2,3))
    den = (p+target).sum(dim=(1,2,3)) + eps
    dice = 1 - (num / den)
    return dice.mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-root', type=str, required=True)
    ap.add_argument('--mask-root', type=str, required=True)
    ap.add_argument('--meta-csv', type=str, default=None)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', type=str, default='./runs/seg_unet')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SegDataset(args.img_root, args.mask_root, args.meta_csv, size=args.size)
    val_len = max(1, int(0.2*len(ds)))
    tr_len = len(ds) - val_len
    tr, va = random_split(ds, [tr_len, val_len], generator=torch.Generator().manual_seed(42))

    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(in_channels=3, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    best = {'loss': 1e9, 'epoch': -1}
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        tl = 0.0
        for img, mask, _ in tr_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits = model(img)
                loss = 0.5*bce(logits, mask) + 0.5*dice_loss(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += float(loss.item()) * img.size(0)
        tl /= max(1, len(tr_loader.dataset))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for img, mask, _ in va_loader:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                loss = 0.5*bce(logits, mask) + 0.5*dice_loss(logits, mask)
                vl += float(loss.item()) * img.size(0)
        vl /= max(1, len(va_loader.dataset))

        print(f"[Epoch {epoch:03d}] seg train {tl:.4f} | val {vl:.4f}")
        if vl < best['loss']:
            best = {'loss': vl, 'epoch': epoch}
            torch.save({'model': model.state_dict(), 'args': vars(args)}, os.path.join(args.out, 'best_seg.pt'))
            with open(os.path.join(args.out, 'best.json'), 'w') as f: f.write(json.dumps(best, indent=2))

    print("Done. Best seg:", best)

if __name__ == '__main__':
    main()
