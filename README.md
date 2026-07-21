# SAIL Steel Defect Detection — Dashboard

A small Gradio dashboard around your verified `infer_severstal.py` pipeline.
No upload widgets — you pre-place your dataset images and checkpoints in two
folders, the dashboard scans them, and you pick from dropdowns.

```
sail_defect_dashboard/
├── app.py                  # the dashboard
├── infer_severstal.py      # your verified inference script (used as-is)
├── requirements.txt
├── dataset_images/         # put steel images here (256x1600, jpg/png)
└── checkpoints/             # put your .pth files here
```

## 1. Set up

Copy your images into `dataset_images/` and your checkpoint files into
`checkpoints/` (the four from the handoff package: `fpn_efficientnetb5_best
(2).pth`, `fpn_seresnext_CORRECTED_0.6325.pth`, `fpn_seresnext_fold0.pth`,
`fpn_seresnext_fold1.pth`, or any subset).

```bash
pip install -r requirements.txt
```

## 2a. Run in VS Code / locally

```bash
python app.py
```

Opens at `http://127.0.0.1:7860` — VS Code will offer to open it in a
Simple Browser tab, or open it in your regular browser.

If you have a local GPU, make sure your `torch` install matches your CUDA
version (see the note in `requirements.txt`) — the dashboard auto-detects
and uses CUDA if `torch.cuda.is_available()`.

## 2b. Run in Google Colab

```python
!pip install -q segmentation-models-pytorch albumentations gradio
!python app.py --share
```

Upload `dataset_images/` and `checkpoints/` into the Colab file browser (or
mount Drive and set `SAIL_DATASET_DIR` / `SAIL_MODEL_DIR` env vars to point
at them) before running. `--share` prints a public `*.gradio.live` link since
Colab doesn't expose localhost directly.

## 3. Using it

1. Click **Refresh** to scan the two folders.
2. Pick an image from the **Dataset image** dropdown.
3. Pick one or more checkpoints to ensemble (defaults to the best-known pair:
   EfficientNet-B5 + SE-ResNeXt CORRECTED, 0.9238 per-image Dice).
4. Adjust the sigmoid threshold if needed (default 0.5, matches training).
5. Click **Run inference** to see:
   - the original image with predicted defect masks overlaid (colored per class)
   - a stats table: which classes fired, pixel count, % of image area, mean
     confidence in the defect region, and the area-cleanup threshold used
   - a small gallery of each class's raw binary mask

Models are loaded once and cached in memory, so switching images/thresholds
after the first run is fast; switching which checkpoints are selected loads
any new ones on demand.

## Notes / limits carried over from the handoff package

- Images must be the native Severstal shape (256×1600) — this pipeline does
  not resize (see `PREPROCESSING.md` in the original handoff package).
- Class 3 is the weakest class across every model tested (~0.73–0.78 Dice) —
  treat its predictions with extra caution.
- Area-cleanup thresholds `[400, 1500, 1500, 2000]` (classes 1–4) are fixed,
  matching what's embedded in the CORRECTED checkpoint's own metadata.
- This dashboard shows masks + stats only — it does not yet include the
  Green/Yellow/Red severity tiering system; happy to add that as a tab if
  you want the percentile thresholds wired in too.
