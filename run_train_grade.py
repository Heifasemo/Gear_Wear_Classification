
import os, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from gearwear.models_grade import GradeModel, coral_loss
from gearwear.dataset_grade import GradeDataset
from gearwear.seg_unet import UNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-root', type=str, required=True)
    ap.add_argument('--meta-csv', type=str, required=True)
    ap.add_argument('--mask-root', type=str, default=None, help='If provided, use GT masks to compute area_ratio & aux channel')
    ap.add_argument('--use-depth-bins', action='store_true', help='Use depth_bin in CSV instead of depth_mm')
    ap.add_argument('--num-classes', type=int, default=5)
    ap.add_argument('--in-ch', type=int, default=4, help='RGB+aux(=mask or zeros)')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=str, default='./runs/grade_model')
    ap.add_argument('--seg-checkpoint', type=str, default=None, help='If provided, run U-Net to predict masks & compute area_ratio')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seg_model = None
    if args.seg_checkpoint:
        seg_model = UNet(in_channels=3, base=32).to(device)
        ckpt = torch.load(args.seg_checkpoint, map_location=device)
        seg_model.load_state_dict(ckpt['model']); seg_model.eval()

    ds = GradeDataset(args.img_root, args.meta_csv, mask_root=args.mask_root,
                      size=args.size, use_depth_bins=args.use_depth_bins,
                      seg_model=seg_model, device=device)
    val_len = int(len(ds)*args.val_split)
    tr_len = len(ds) - val_len
    tr, va = random_split(ds, [tr_len, val_len], generator=torch.Generator().manual_seed(42))

    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    va_loader = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = GradeModel(in_channels=args.in_ch, num_classes=args.num_classes, depth_bins=(1 if args.use_depth_bins else 0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    best = {'loss': 1e9, 'epoch': -1}

    huber = nn.SmoothL1Loss(beta=1.0)

    for epoch in range(1, args.epochs+1):
        model.train()
        tl = 0.0
        for x4, y, _ in tr_loader:
            x4 = x4.to(device)
            label = y['label'].to(device)
            area_t = y['area_ratio'].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(x4, area_ratio_input=area_t)  # use measured/predicted area as input for grade
                # depth supervision (optional if you have labels)
                if args.use_depth_bins:
                    # depth from CSV bins -> train via CORAL; here we just let model's internal depth be free (not supervised)
                    loss_depth = torch.tensor(0.0, device=device)
                else:
                    depth_t = y['depth_mm'].to(device).unsqueeze(1)
                    loss_depth = huber(out['depth_mm_pred'], depth_t)
                loss_grade = coral_loss(out['grade']['logits'], label, args.num_classes)
                # also stabilize area head to predict correct ratio (auxiliary)
                loss_area = huber(out['area_ratio_pred'], area_t)
                loss = loss_grade + 0.5*loss_area + 0.5*loss_depth
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += float(loss.item()) * x4.size(0)
        tl /= max(1, len(tr_loader.dataset))

        # eval
        model.eval()
        vl = 0.0; y_true=[]; y_pred=[]
        with torch.no_grad():
            for x4, y, _ in va_loader:
                x4 = x4.to(device); label = y['label'].to(device)
                area_t = y['area_ratio'].to(device).unsqueeze(1)
                out = model(x4, area_ratio_input=area_t)
                lg = coral_loss(out['grade']['logits'], label, args.num_classes)
                la = huber(out['area_ratio_pred'], area_t)
                if args.use_depth_bins:
                    ld = torch.tensor(0.0, device=device)
                else:
                    depth_t = y['depth_mm'].to(device).unsqueeze(1)
                    ld = huber(out['depth_mm_pred'], depth_t)
                l = lg + 0.5*la + 0.5*ld
                vl += float(l.item()) * x4.size(0)
                # decode grade
                pred = torch.sum(out['grade']['probs']>0.5, dim=1).cpu().numpy().tolist()
                y_true.extend(label.cpu().numpy().tolist()); y_pred.extend(pred)
        vl /= max(1, len(va_loader.dataset))

        print(f"[Epoch {epoch:03d}] grade train {tl:.4f} | val {vl:.4f}")
        if vl < best['loss']:
            best = {'loss': vl, 'epoch': epoch}
            torch.save({'model': model.state_dict(), 'args': vars(args)}, os.path.join(args.out, 'best_grade.pt'))
            with open(os.path.join(args.out, 'best.json'), 'w') as f: f.write(json.dumps(best, indent=2))

    print("Done. Best grade:", best)

if __name__ == '__main__':
    main()
