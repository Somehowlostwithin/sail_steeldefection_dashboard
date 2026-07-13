"""
SAIL Steel Surface Defect Detection — Dashboard
=================================================
Gradio dashboard for running the trained FPN checkpoints (EfficientNet-B5 /
SE-ResNeXt50) against pre-uploaded Severstal-format steel images.

Design intent:
  - No per-run upload widgets. Drop your dataset images into DATASET_DIR and
    your .pth checkpoints into MODEL_DIR (see README_DASHBOARD.md), then just
    pick an image + model(s) from dropdowns and hit Run.
  - Reuses infer_severstal.py exactly as verified in the handoff package
    (load_model / predict_probs / postprocess) rather than reimplementing
    preprocessing, so results match what was validated (0.9238 per-image Dice
    for the EfficientNet-B5 + SE-ResNeXt CORRECTED ensemble).

Run:
  VS Code / local:  python app.py            (opens http://127.0.0.1:7860)
  Google Colab:      !python app.py --share  (prints a public gradio.live link)
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import gradio as gr
import pandas as pd

from infer_severstal import (
    load_model,
    predict_probs,
    postprocess,
    IMG_SHAPE,
    DEFAULT_THRESHOLD,
    DEFAULT_AREA_THRESHOLDS,
)

# ------------------------------------------------------------------
# Config — edit these paths, or override with env vars, to point at
# your pre-uploaded dataset / checkpoints.
# ------------------------------------------------------------------
DATASET_DIR = os.environ.get("SAIL_DATASET_DIR", "./dataset_images")
MODEL_DIR = os.environ.get("SAIL_MODEL_DIR", "./checkpoints")

# Encoder for each known checkpoint filename (from the handoff README).
# Anything not listed here is guessed from the filename at load time.
KNOWN_ENCODERS = {
    "fpn_efficientnetb5_best (2).pth": "efficientnet-b5",
    "fpn_seresnext_fold0.pth": "se_resnext50_32x4d",
    "fpn_seresnext_fold1.pth": "se_resnext50_32x4d",
    "fpn_seresnext_CORRECTED_0.6325.pth": "se_resnext50_32x4d",
}

# Reported per-image Dice from validation (for display only — purely cosmetic).
REPORTED_DICE = {
    "fpn_efficientnetb5_best (2).pth": 0.9200,
    "fpn_seresnext_CORRECTED_0.6325.pth": 0.9169,
    "fpn_seresnext_fold0.pth": 0.8694,
    "fpn_seresnext_fold1.pth": 0.8653,
}

# Best known ensemble, per the handoff report — pre-selected by default.
DEFAULT_CHECKPOINTS = {
    "fpn_efficientnetb5_best (2).pth",
    "fpn_seresnext_CORRECTED_0.6325.pth",
}

CLASS_COLORS_BGR = {
    1: (60, 25, 230),    # red
    2: (75, 180, 60),    # green
    3: (25, 225, 255),   # yellow
    4: (200, 130, 0),    # blue
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE = {}  # checkpoint_path -> loaded model


# ------------------------------------------------------------------
# Discovery helpers
# ------------------------------------------------------------------
def guess_encoder(filename):
    lower = filename.lower()
    if "efficientnet" in lower:
        return "efficientnet-b5"
    if "seresnext" in lower or "se_resnext" in lower:
        return "se_resnext50_32x4d"
    return None


def list_images():
    if not os.path.isdir(DATASET_DIR):
        return []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(DATASET_DIR, ext)))
    return sorted(os.path.basename(f) for f in files)


def list_checkpoints():
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".pth"))


def checkpoint_label(fname):
    dice = REPORTED_DICE.get(fname)
    enc = KNOWN_ENCODERS.get(fname) or guess_encoder(fname) or "unknown encoder"
    return f"{fname}  [{enc}" + (f", ~{dice:.4f} per-image Dice]" if dice else "]")


def refresh_choices():
    images = list_images()
    ckpts = list_checkpoints()
    ckpt_labels = [checkpoint_label(f) for f in ckpts]
    default_selected = [checkpoint_label(f) for f in ckpts if f in DEFAULT_CHECKPOINTS]
    status = (
        f"Found {len(images)} image(s) in `{DATASET_DIR}`, "
        f"{len(ckpts)} checkpoint(s) in `{MODEL_DIR}`. Device: {DEVICE}."
    )
    return (
        gr.update(choices=images, value=(images[0] if images else None)),
        gr.update(choices=ckpt_labels, value=default_selected),
        status,
    )


def label_to_filename(label):
    return label.split("  [", 1)[0]


# ------------------------------------------------------------------
# Model loading (cached)
# ------------------------------------------------------------------
def get_models(checkpoint_filenames):
    models = []
    for fname in checkpoint_filenames:
        path = os.path.join(MODEL_DIR, fname)
        if path not in _MODEL_CACHE:
            encoder = KNOWN_ENCODERS.get(fname) or guess_encoder(fname)
            if encoder is None:
                raise gr.Error(
                    f"Can't determine encoder for '{fname}'. Add it to KNOWN_ENCODERS "
                    f"in app.py (efficientnet-b5 or se_resnext50_32x4d)."
                )
            _MODEL_CACHE[path] = load_model(path, encoder, device=DEVICE)
        models.append(_MODEL_CACHE[path])
    return models


# ------------------------------------------------------------------
# Inference + visualization
# ------------------------------------------------------------------
def build_overlay(image_rgb, masks):
    overlay = image_rgb.copy()
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    for cls_id in range(1, 5):
        mask = masks[cls_id - 1].astype(bool)
        if not mask.any():
            continue
        color = np.array(CLASS_COLORS_BGR[cls_id], dtype=np.uint8)
        colored = np.zeros_like(bgr)
        colored[mask] = color
        bgr = np.where(mask[..., None], cv2.addWeighted(bgr, 0.55, colored, 0.45, 0), bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_inference(image_name, checkpoint_labels, threshold):
    if not image_name:
        raise gr.Error(f"No image selected. Add images to {DATASET_DIR} and click Refresh.")
    if not checkpoint_labels:
        raise gr.Error(f"No checkpoint selected. Add .pth files to {MODEL_DIR} and click Refresh.")

    checkpoint_filenames = [label_to_filename(l) for l in checkpoint_labels]
    image_path = os.path.join(DATASET_DIR, image_name)

    image = cv2.imread(image_path)
    if image is None:
        raise gr.Error(f"Could not read {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != IMG_SHAPE:
        raise gr.Error(
            f"{image_name} has shape {image.shape[:2]}, expected {IMG_SHAPE} "
            f"(this pipeline does not resize — see PREPROCESSING.md)."
        )

    models = get_models(checkpoint_filenames)
    probs = predict_probs(models, image, device=DEVICE)
    masks = postprocess(probs, threshold=threshold, area_thresholds=DEFAULT_AREA_THRESHOLDS)

    overlay = build_overlay(image, masks)

    total_px = IMG_SHAPE[0] * IMG_SHAPE[1]
    rows = []
    for cls_id in range(1, 5):
        mask = masks[cls_id - 1]
        px_count = int(mask.sum())
        present = px_count > 0
        mean_conf = float(probs[cls_id - 1][mask.astype(bool)].mean()) if present else 0.0
        rows.append({
            "Class": cls_id,
            "Defect Present": "Yes" if present else "No",
            "Pixel Count": px_count,
            "Area %": round(100 * px_count / total_px, 3),
            "Mean Confidence (defect px)": round(mean_conf, 3),
            "Area Threshold (cleanup)": DEFAULT_AREA_THRESHOLDS[cls_id - 1],
        })
    stats_df = pd.DataFrame(rows)

    class_thumbs = []
    for cls_id in range(1, 5):
        mask_vis = (masks[cls_id - 1] * 255).astype(np.uint8)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
        class_thumbs.append((mask_vis, f"Class {cls_id}"))

    n_ckpts = len(models)
    mode = "ensemble" if n_ckpts > 1 else "single-model"
    summary = (
        f"**{image_name}** — {mode} inference with {n_ckpts} checkpoint(s) on **{DEVICE}**, "
        f"threshold={threshold:.2f}. "
        f"{int(stats_df['Defect Present'].eq('Yes').sum())}/4 classes detected."
    )

    return overlay, stats_df, class_thumbs, summary


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def build_demo():
    with gr.Blocks(title="SAIL Steel Defect Detection Dashboard") as demo:
        gr.Markdown(
            "# 🔩 SAIL Steel Surface Defect Detection\n"
            "Run the trained FPN checkpoints (EfficientNet-B5 / SE-ResNeXt50) on "
            "pre-uploaded Severstal-format images. Best known ensemble scores "
            "**0.9238 per-image Dice** (EfficientNet-B5 + SE-ResNeXt CORRECTED)."
        )

        status_box = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1):
                image_dd = gr.Dropdown(label="Dataset image", choices=[])
                ckpt_cbg = gr.CheckboxGroup(label="Checkpoint(s) — select multiple to ensemble", choices=[])
                threshold_slider = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05, label="Sigmoid threshold")
                refresh_btn = gr.Button("🔄 Refresh (rescan folders)")
                run_btn = gr.Button("▶ Run inference", variant="primary")
                gr.Markdown(
                    f"Drop images into `{DATASET_DIR}` and checkpoints into `{MODEL_DIR}`, "
                    "then Refresh. Legend: 🔴 Class1  🟢 Class2  🟡 Class3  🔵 Class4"
                )
            with gr.Column(scale=2):
                summary_md = gr.Markdown()
                overlay_img = gr.Image(label="Prediction overlay", type="numpy")
                stats_table = gr.Dataframe(label="Per-class stats")
                class_gallery = gr.Gallery(label="Per-class raw masks", columns=4, height=180)

        demo.load(fn=refresh_choices, outputs=[image_dd, ckpt_cbg, status_box])
        refresh_btn.click(fn=refresh_choices, outputs=[image_dd, ckpt_cbg, status_box])
        run_btn.click(
            fn=run_inference,
            inputs=[image_dd, ckpt_cbg, threshold_slider],
            outputs=[overlay_img, stats_table, class_gallery, summary_md],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link (use in Colab).")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    demo = build_demo()
    port = int(os.environ.get("PORT", args.port))
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=port)