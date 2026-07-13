"""
infer_severstal.py
===================
Inference script for the Severstal Steel Defect Detection FPN checkpoints.

Verified directly against the real uploaded checkpoint files:
    fpn_efficientnetb5_best (2).pth        -> key 'model_state', encoder 'efficientnet-b5'
    fpn_seresnext_fold0.pth                -> raw state_dict, encoder 'se_resnext50_32x4d'
    fpn_seresnext_fold1.pth                -> raw state_dict, encoder 'se_resnext50_32x4d'
    fpn_seresnext_CORRECTED_0.6325.pth     -> key 'model_state', encoder 'se_resnext50_32x4d'
                                               also embeds its own 'pixel_thresholds': [400,1500,1500,2000]

All four load cleanly with:
    smp.FPN(encoder_name=..., encoder_weights=None, classes=4, activation=None)

Follows the spec in PREPROCESSING.md:
  - RGB, native 256x1600, no resize
  - ImageNet normalization
  - sigmoid -> threshold 0.5 -> per-class small-region removal
  - RLE encoded with order='F' (column-major)
  - Ensemble = average of sigmoid probabilities (not logits, not binary masks)

USAGE
-----
Single model, single image, print per-class RLE:
    python infer_severstal.py \\
        --checkpoint fpn_efficientnetb5_best.pth \\
        --encoder efficientnet-b5 \\
        --image path/to/image.jpg

Ensemble of two checkpoints:
    python infer_severstal.py \\
        --checkpoint fpn_efficientnetb5_best.pth --encoder efficientnet-b5 \\
        --checkpoint fpn_seresnext_CORRECTED_0.6325.pth --encoder se_resnext50_32x4d \\
        --image path/to/image.jpg

Whole folder -> submission.csv (RLE per ImageId/ClassId row, Severstal format):
    python infer_severstal.py \\
        --checkpoint fpn_efficientnetb5_best.pth --encoder efficientnet-b5 \\
        --checkpoint fpn_seresnext_CORRECTED_0.6325.pth --encoder se_resnext50_32x4d \\
        --image_dir path/to/test_images \\
        --out submission.csv

As a library:
    from infer_severstal import load_model, predict_probs, postprocess, mask_to_rle
"""

import os
import argparse
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage

IMG_SHAPE = (256, 1600)          # (height, width)
NUM_CLASSES = 4
DEFAULT_THRESHOLD = 0.5
DEFAULT_AREA_THRESHOLDS = [400, 1500, 1500, 2000]   # per ClassId 1..4, from PREPROCESSING.md / CORRECTED checkpoint


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
def load_model(checkpoint_path, encoder_name, device="cpu"):
    """
    Loads a checkpoint into a fresh smp.FPN model.
    Handles both checkpoint formats seen in this project:
      - wrapped dict with a 'model_state' key (efficientnetb5_best, seresnext_CORRECTED)
      - raw state_dict with no wrapper (seresnext_fold0, seresnext_fold1)
    """
    model = smp.FPN(encoder_name=encoder_name, encoder_weights=None, classes=NUM_CLASSES, activation=None)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state"}
    else:
        # raw state_dict checkpoint (e.g. fold0 / fold1)
        state_dict = ckpt
        meta = {}

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    if meta:
        print(f"[{os.path.basename(checkpoint_path)}] loaded. metadata: {meta}")
    else:
        print(f"[{os.path.basename(checkpoint_path)}] loaded (raw state_dict, no metadata).")

    return model


# ------------------------------------------------------------
# Preprocessing (must match PREPROCESSING.md exactly)
# ------------------------------------------------------------
def get_inference_transform():
    return A.Compose([A.Normalize(), ToTensorV2()])


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != IMG_SHAPE:
        raise ValueError(
            f"{image_path} has shape {image.shape[:2]}, expected {IMG_SHAPE}. "
            "This pipeline does not resize -- see PREPROCESSING.md."
        )
    return image


# ------------------------------------------------------------
# Forward pass -> probabilities (sigmoid applied here, NOT inside the model)
# ------------------------------------------------------------
@torch.no_grad()
def predict_probs(models, image, device="cpu"):
    """
    models: single model or list of models to ensemble (probability-averaged).
    image: HxWx3 RGB uint8 array.
    Returns: numpy array (NUM_CLASSES, H, W) of averaged sigmoid probabilities.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    transform = get_inference_transform()
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)  # 1x3xHxW

    probs_sum = None
    for model in models:
        logits = model(tensor)               # 1x4xHxW, raw logits (activation=None)
        probs = torch.sigmoid(logits)
        probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_avg = (probs_sum / len(models)).squeeze(0).cpu().numpy()  # 4xHxW
    return probs_avg


# ------------------------------------------------------------
# Postprocessing: threshold + small-region cleanup
# ------------------------------------------------------------
def remove_small_regions(binary_mask, min_area):
    labeled, num = ndimage.label(binary_mask)
    if num == 0:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, num + 1))
    for region_id, size in enumerate(sizes, start=1):
        if size < min_area:
            binary_mask[labeled == region_id] = 0
    return binary_mask


def postprocess(probs, threshold=DEFAULT_THRESHOLD, area_thresholds=None):
    """
    probs: (NUM_CLASSES, H, W) sigmoid probabilities.
    Returns: (NUM_CLASSES, H, W) uint8 binary masks, cleaned.
    """
    if area_thresholds is None:
        area_thresholds = DEFAULT_AREA_THRESHOLDS

    masks = np.zeros_like(probs, dtype=np.uint8)
    for c in range(probs.shape[0]):
        binary = (probs[c] > threshold).astype(np.uint8)
        binary = remove_small_regions(binary, area_thresholds[c])
        masks[c] = binary
    return masks


# ------------------------------------------------------------
# RLE encode (column-major, matches training-time rle_to_mask(..., order='F'))
# ------------------------------------------------------------
def mask_to_rle(mask):
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)


# ------------------------------------------------------------
# End-to-end single image
# ------------------------------------------------------------
def predict_image(models, image_path, device="cpu", threshold=DEFAULT_THRESHOLD, area_thresholds=None):
    image = load_image(image_path)
    probs = predict_probs(models, image, device=device)
    masks = postprocess(probs, threshold=threshold, area_thresholds=area_thresholds)
    rles = {cls_id: mask_to_rle(masks[cls_id - 1]) for cls_id in range(1, NUM_CLASSES + 1)}
    return rles, masks, probs


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                         help="Path to a checkpoint. Repeat for ensembling.")
    parser.add_argument("--encoder", action="append", required=True,
                         choices=["efficientnet-b5", "se_resnext50_32x4d"],
                         help="Encoder for the matching --checkpoint (same order/count).")
    parser.add_argument("--image", help="Single image to run inference on.")
    parser.add_argument("--image_dir", help="Directory of images to run inference on (batch mode).")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path for batch mode.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if len(args.checkpoint) != len(args.encoder):
        raise ValueError("--checkpoint and --encoder must be passed the same number of times, in matching order.")

    if not args.image and not args.image_dir:
        raise ValueError("Provide either --image (single) or --image_dir (batch).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = [
        load_model(ckpt_path, enc, device=device)
        for ckpt_path, enc in zip(args.checkpoint, args.encoder)
    ]
    print(f"Loaded {len(models)} model(s) for {'ensemble' if len(models) > 1 else 'single-model'} inference.")

    if args.image:
        rles, masks, probs = predict_image(models, args.image, device=device, threshold=args.threshold)
        print(f"\nPredictions for {args.image}:")
        for cls_id, rle in rles.items():
            pixel_count = int(masks[cls_id - 1].sum())
            print(f"  ClassId {cls_id}: {pixel_count} px" + (f" | RLE: {rle[:80]}{'...' if len(rle) > 80 else ''}" if rle else " | (no defect predicted)"))
        return

    # Batch mode -> submission.csv
    rows = []
    image_files = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"Running inference on {len(image_files)} images in {args.image_dir} ...")
    for fname in image_files:
        image_path = os.path.join(args.image_dir, fname)
        rles, _, _ = predict_image(models, image_path, device=device, threshold=args.threshold)
        for cls_id in range(1, NUM_CLASSES + 1):
            rows.append({"ImageId_ClassId": f"{fname}_{cls_id}", "EncodedPixels": rles[cls_id]})

    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
