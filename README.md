
# Model framework: Segmentation (U-Net) + Grading (ResNet + CORAL) + Grad-CAM

This package implements the recommended:
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
