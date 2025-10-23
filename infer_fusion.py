
import os, argparse, csv
import numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image

from gearwear.seg_unet import UNet
from gearwear.models_grade import GradeModel
from gearwear.dataset_grade import GradeDataset
from gearwear.gradcam import GradCAM

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-root', type=str, required=True)
    ap.add_argument('--meta-csv', type=str, required=True)
    ap.add_argument('--seg-checkpoint', type=str, required=True)
    ap.add_argument('--grade-checkpoint', type=str, required=True)
    ap.add_argument('--num-classes', type=int, default=5)
    ap.add_argument('--in-ch', type=int, default=4)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out-dir', type=str, default='runs/infer_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seg = UNet(in_channels=3, base=32).to(device)
    seg_ckpt = torch.load(args.seg_checkpoint, map_location=device)
    seg.load_state_dict(seg_ckpt['model']); seg.eval()

    ds = GradeDataset(args.img_root, args.meta_csv, mask_root=None, size=args.size,
                      use_depth_bins=False, seg_model=seg, device=device)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    grade = GradeModel(in_channels=args.in_ch, num_classes=args.num_classes, depth_bins=0).to(device)
    gckpt = torch.load(args.grade_checkpoint, map_location=device)
    grade.load_state_dict(gckpt['model']); grade.eval()

    # Grad-CAM on single samples (per-batch first element for demo)
    cam = GradCAM(grade)

    rows = []
    for x4, y, rel in dl:
        x4 = x4.to(device)
        out = grade(x4, area_ratio_input=y['area_ratio'].to(device).unsqueeze(1))
        area = out['area_ratio_pred'].cpu().numpy().reshape(-1)
        depth = out['depth_mm_pred'].cpu().numpy().reshape(-1)
        grade_pred = torch.sum(out['grade']['probs']>0.5, dim=1).cpu().numpy().reshape(-1)
        score = out['grade']['score'].cpu().numpy().reshape(-1)
        theta = out['grade']['theta'].detach().cpu().numpy().tolist()
        wd, wa, wda, b = [float(v) for v in out['grade']['weights']]

        # save predicted mask (from seg model run inside dataset) by recomputing quickly
        # x4 = [RGB, aux]; aux is mask, extract and save
        aux = x4[:,3:4]  # [B,1,H,W]
        for i, r in enumerate(rel):
            # save mask
            mask_path = os.path.join(args.out_dir, r.replace('/', '_') + "_mask.png")
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            save_image(aux[i], mask_path)

            # grad-cam: use mean score of grade as target
            x_single = x4[i:i+1]
            s = out['grade']['score'][i:i+1].mean()
            hm = cam.generate(x_single, s)  # [H,W] in [0,1]
            # save heatmap as PNG
            hm_img = (hm.numpy()*255).astype(np.uint8)
            pil = Image.fromarray(hm_img)
            heat_path = os.path.join(args.out_dir, r.replace('/', '_') + "_heat.png")
            pil.save(heat_path)

            rows.append({
                'relpath': r,
                'area_ratio_pred': float(area[i]),
                'depth_mm_pred': float(depth[i]),
                'grade': int(grade_pred[i]),
                'score': float(score[i]),
                'theta_list': theta,
                'w_depth': wd, 'w_area': wa, 'w_inter': wda, 'bias': b,
                'mask_path': mask_path,
                'heatmap_path': heat_path
            })

    out_csv = os.path.join(args.out_dir, "preds.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print("Saved:", out_csv)

if __name__ == '__main__':
    main()
