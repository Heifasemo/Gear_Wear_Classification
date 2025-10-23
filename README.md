
# Plan A: Segmentation (U-Net) + Grading (ResNet + CORAL) + Grad-CAM

This package implements the recommended **Plan A**:
- **U-Net** for pixel-wise wear segmentation to compute **area_ratio** precisely.
- **ResNet18(SE)** grading model that takes **RGB + aux (mask)** and predicts:
  - area_ratio (auxiliary regression),
  - depth_mm (regression),
  - **grade** via a **monotonic CORAL** head based on (depth, area).
- **Grad-CAM** for interpretability, exported as heatmaps.

## Folder
```
gearwear_fusion_planA/
  gearwear/
    seg_unet.py       # U-Net
    models_grade.py   # ResNet backbone + area/depth heads + CORAL grade head
    gradcam.py        # Grad-CAM helper
    dataset_seg.py    # Supervised segmentation dataset
    dataset_grade.py  # Grading dataset (uses GT mask or a seg checkpoint to auto-predict masks)
  run_train_seg.py     # Train U-Net (BCE + Dice)
  run_train_grade.py   # Train grading model (uses GT/predicted area_ratio)
  infer_fusion.py      # End-to-end inference: save masks, heatmaps, and CSV
  README.md
```

## Data layout
```
data/
  images/            # RGB images, referenced by CSV relpath
  masks/             # binary masks (optional if you use seg-checkpoint)
  meta.csv           # columns: relpath,label,area_ratio,depth_mm,depth_bin
```

## 1) Train segmentation
```bash
python run_train_seg.py \
  --img-root data/images \
  --mask-root data/masks \
  --meta-csv data/meta.csv \
  --size 256 \
  --batch 8 \
  --epochs 60 \
  --out runs/seg_unet
```

## 2) Train grading
Use GT masks (or predicted masks via `--seg-checkpoint`). Area ratio from mask is fed to grade head.
```bash
# Use GT masks
python run_train_grade.py \
  --img-root data/images \
  --mask-root data/masks \
  --meta-csv data/meta.csv \
  --num-classes 5 \
  --in-ch 4 \
  --epochs 50 \
  --out runs/grade_gt

# Or use predicted masks from U-Net
python run_train_grade.py \
  --img-root data/images \
  --meta-csv data/meta.csv \
  --num-classes 5 \
  --in-ch 4 \
  --seg-checkpoint runs/seg_unet/best_seg.pt \
  --epochs 50 \
  --out runs/grade_predmask
```

## 3) Inference (end-to-end)
```bash
python infer_fusion.py \
  --img-root data/images \
  --meta-csv data/meta.csv \
  --seg-checkpoint runs/seg_unet/best_seg.pt \
  --grade-checkpoint runs/grade_predmask/best_grade.pt \
  --num-classes 5 \
  --in-ch 4 \
  --out-dir runs/infer_out
```
The script saves:
- `*_mask.png` (predicted wear masks)
- `*_heat.png` (Grad-CAM heatmaps of the grading model)
- `preds.csv` (relpath, area_ratio, depth_mm, grade, score, theta, weights, paths)

## Notes
- If you only have `depth_bin`, pass `--use-depth-bins` to `run_train_grade.py`. The code currently supervises depth_mm; modify where needed.
- For better grading stability in production, consider hysteresis thresholds and moving-window voting.
- You can extend the aux channel to use **heatmaps** or **morphological ops** on masks (e.g., dilation) to help focus on wear borders.
