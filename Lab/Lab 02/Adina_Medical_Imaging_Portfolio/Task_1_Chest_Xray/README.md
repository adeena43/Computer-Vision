# Task 1 — Diagnostic Enhancement of Chest X-Rays

## Dataset
[COVID19+PNEUMONIA+NORMAL Chest X-Ray Image Dataset](https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images)
(only three representative real samples — one per class — are kept in `data/`, per the
100MB rule; the full dataset is ~5,200 images across the three classes).

## Files
- `data/pneumonia_sample.png` — the real, low-contrast grayscale X-ray used as the primary
  input for the pipeline below.
- `data/covid_sample.png`, `data/normal_sample.png` — two additional real samples (one per
  remaining class) included for comparison; cell 0 of the notebook prints their stats.
- `xray_enhancement.ipynb` — full processing pipeline notebook.
- `output/` — every intermediate result saved as PNG, plus side-by-side comparison figures.

## Processing Pipeline

1. **Load & Display** — read the X-ray with `cv2.IMREAD_GRAYSCALE`; inspect its
   min/max/mean/std to confirm it is underexposed (low dynamic range, low mean).
2. **Histogram Equalization** (`cv2.equalizeHist`) — forcefully redistributes intensities
   across the full 0–255 range, revealing lung structure that was invisible in the raw
   scan. Verified quantitatively by comparing raw vs. equalized standard deviation and
   histograms.
3. **Color Mapping** (`cv2.applyColorMap(..., cv2.COLORMAP_JET)`) — converts the
   single-channel equalized image into a false-color heatmap so subtle intensity
   differences become obvious color-boundary shifts to the human eye.
4. **Color Balance** — a custom `color_balance()` function scales the B/G/R channels
   independently to neutralize any artificial cast the JET colormap introduces.
5. **Color Filtering / Thresholding** (`cv2.threshold`) — a strict binary threshold
   (`> 200`) isolates only the densest tissue (bone / dense fluid), masking everything
   else to black.
6. **Logarithmic Transformation** — `s = c * log(1 + r)` applied to the *raw* image
   expands dark background values, revealing faint outer ribcage edges that are
   normally lost in the underexposed shadows.
7. **Power-Law (Gamma) Transformation** — `s = c * r^gamma` with `gamma = 0.6` (< 1.0)
   brightens midtones relatively more than highlights, softening the harsh bone contrast
   introduced by equalization so that soft lung tissue detail remains legible.

Each stage is documented in-notebook with a short written explanation of *why* the
transform helps (see the markdown cells directly under each code cell).

## How to Run
```bash
cd Task_1_Chest_XRay
jupyter notebook xray_enhancement.ipynb
```
Run all cells top-to-bottom. Outputs are written to `output/`.

## Notes
All three files in `data/` are real, unmodified images extracted directly from the linked
Kaggle dataset (`COVID_10.png`, `NORMAL_10.png`, `PNEUMONIA_10.png` from their respective
class folders). The image chosen for the main pipeline (`pneumonia_sample.png`) has the
lowest mean intensity of the three (mean ≈ 98.6 vs. 125.9 for NORMAL and 153.0 for COVID),
making it the clearest real example of underexposure for this pipeline. To run the
pipeline on a different image, just point `IMAGE_PATH` at any other file in `data/`, or
drop in additional samples copied from the full dataset.
