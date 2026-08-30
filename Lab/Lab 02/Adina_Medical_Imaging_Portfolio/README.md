# Medical Imaging Portfolio

This repository contains three self-contained mini-projects applying classical
computer-vision / image-processing mathematics (histogram equalization, false-color
mapping, color balance, thresholding, logarithmic and power-law transforms, and
multi-modal weighted fusion) to real medical imaging modalities: chest X-ray, cardiac
CT/MRI, and echocardiogram ultrasound video.

## Repository Structure

```
StudentName_Medical_Imaging_Portfolio/
├── requirements.txt                 # global dependencies for all 3 tasks
├── README.md                        # this file
├── Task_1_Chest_XRay/
│   ├── data/                        # sample X-ray image(s) used
│   ├── output/                      # saved enhanced images / screenshots
│   ├── xray_enhancement.ipynb       # notebook: contrast/heatmap/threshold/log/gamma
│   └── README.md                    # explanation of the processing pipeline
├── Task_2_Cardiac_Fusion/
│   ├── data/                        # sample CT + MRI matched slice pair
│   ├── output/                      # fused heatmaps and comparison charts
│   ├── modal_fusion.ipynb           # notebook: CT/MRI weighted fusion pipeline
│   └── README.md                    # fusion weighting logic and analysis
└── Task_3_Echo_Analysis/
    ├── data/                        # short ultrasound .mp4 snippet used
    ├── output/                      # side-by-side video + sample frame screenshots
    ├── realtime_echo.ipynb          # notebook: real-time OpenCV video pipeline
    └── README.md                    # instructions on how to run the live video loop
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open any of the three task notebooks and run all cells top to bottom. All scripts
use **relative paths** (`data/...`, `output/...`), so notebooks must be run from *within*
their own task folder (this is the default behaviour when you open them from Jupyter).

## About the sample data

Per the 100MB / GitHub size-limit rule, the full source datasets are **not** included in
this repo:

- Task 1 source dataset: [COVID19+PNEUMONIA+NORMAL Chest X-Ray Image Dataset](https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images)
- Task 2 source dataset: [Heart CT & MRI Dataset](https://www.kaggle.com/datasets/ziya07/heart-ct-and-mri-dataset)
- Task 3 source dataset: [Stanford EchoNet-Dynamic Dataset](https://www.kaggle.com/datasets/manojkumarcs28/echonet-dynamic-by-stanford-university)

Each `data/` folder instead contains only the one or two small sample file(s) actually
read by that task's notebook, as required by the submission guidelines. Every file in
every `data/` folder is a **real, unmodified sample extracted directly from the linked
dataset** — no synthetic or generated placeholder data is used anywhere in this repo. All
notebooks have already been executed end-to-end against these real samples, and every
image in every `output/` folder is a genuine result of running the pipeline on that real
data (not a mock-up).

## Tasks Overview

| Task | Modality | Core Techniques |
|------|----------|------------------|
| 1 | Chest X-Ray | Histogram equalization, JET colormap, color balance, thresholding, log & gamma transforms |
| 2 | Cardiac CT + MRI | Independent equalization, colormap fusion, `cv2.addWeighted`, log & gamma correction |
| 3 | Echocardiogram video | Real-time `cv2.VideoCapture` loop, per-frame equalization/heatmap/color-balance/log/gamma, side-by-side monitoring |

See each task's own `README.md` for pipeline-specific details.
