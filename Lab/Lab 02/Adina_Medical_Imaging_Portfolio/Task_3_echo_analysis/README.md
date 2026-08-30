# Task 3 — Real-Time Echocardiogram Video Analysis

## Dataset
[Stanford EchoNet-Dynamic Dataset](https://www.kaggle.com/datasets/manojkumarcs28/echonet-dynamic-by-stanford-university)
(only one short sample clip is kept in `data/`).

## Files
- `data/echo_clip.mp4` — a real ~3.5s / 173-frame EchoNet-Dynamic clip (`0X1A2C60147AF9FDAE`,
  112×112, 50 FPS), used as input.
- `data/FileList_sample.csv`, `data/VolumeTracings_sample.csv` — the rows from the
  dataset's `FileList.csv` (ejection fraction, split, etc.) and `VolumeTracings.csv`
  (left-ventricle trace coordinates) that correspond to this one clip, trimmed from the
  full ~31MB tracing file so `data/` stays small.
- `realtime_echo.ipynb` — real-time OpenCV video-processing pipeline notebook.
- `requirements.txt` — dependencies needed specifically to run this task locally
  (mirrors the root `requirements.txt`; see root for the authoritative version).
- `output/side_by_side_pipeline.mp4` — saved raw-vs-enhanced comparison video.
- `output/sample_pipeline_frames.png` — screenshots of several frames from the pipeline.

## Pipeline (applied to every frame)
1. **Histogram Equalization** — fights the murky, low-contrast nature of ultrasound.
2. **`COLORMAP_JET` heatmap** — highlights blood-flow / tissue-density intensity.
3. **Color balance adjustment** — neutralizes the artificial color cast from the
   colormap.
4. **Logarithmic transform** — reveals detail in the darkest heart-chamber regions.
5. **Power-law (gamma) transform**, `gamma = 0.7` — suppresses the blinding white
   backscatter noise typical of ultrasound probes without crushing shadow detail.
6. **Monitoring array** — `np.hstack([raw, enhanced])` concatenates the raw and enhanced
   frames side-by-side for live/monitoring display.

## How to Run

### In a notebook / headless environment (as submitted)
```bash
cd Task_3_Echo_Analysis
jupyter notebook realtime_echo.ipynb
```
Run all cells with `LIVE_DISPLAY = False` (the default). Since `cv2.imshow()` requires a
GUI/display and does not work inside a headless Jupyter kernel, this mode instead writes
the full side-by-side comparison to `output/side_by_side_pipeline.mp4` and renders sample
frame screenshots inline.

### Live, on your own machine with a display attached
1. `pip install -r requirements.txt`
2. Open `realtime_echo.ipynb` and change the setup cell to `LIVE_DISPLAY = True`.
3. Run all cells. A window titled **"Raw (left) vs Enhanced (right)"** opens showing the
   real-time side-by-side feed, driven by a `while True: cap.read()` loop exactly per the
   assignment spec. Press **`q`** at any time to stop early.
4. Alternatively, copy the "Video Capture Setup" → "Per-Frame Enhancement" →
   "Real-Time Video Processing Loop" cells into a standalone `realtime_echo.py` and run
   `python realtime_echo.py` from the terminal.
