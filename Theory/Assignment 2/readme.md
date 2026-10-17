# UPAR Challenge 2027, Track 1: Pedestrian Attribute Recognition

Computer Vision, Assignment 2
Adina Faraz (23K-0008) and Syed Muneeb Ur Rehman (23K-0038)

## Task

We worked on Track 1 of the UPAR Challenge 2027 (WACV Real-World Surveillance
workshop). Given a cropped image of a person, the model predicts 40 yes/no
attributes, such as gender, age group, hair length, clothes type and colour, and
accessories like bag, hat and glasses. The training data joins PA-100K, PETA and
Market1501. The hidden test images come from cameras that are not in the training
data, so the model has to handle a domain shift.

The score is Avg, the harmonic mean of mA (mean accuracy over attributes) and
instance F1, averaged over the datasets.

## Approach

We fine-tuned a ConvNeXt-Small model (torchvision, pretrained on ImageNet) with a
40-output head. Main parts:

- Weighted binary cross-entropy that ignores missing labels, with light label
  smoothing (0.02), so rare attributes get more weight.
- AdamW, one-cycle learning rate, bf16 mixed precision, EMA of the weights.
- Strong augmentation: low-resolution simulation, colour jitter, blur, small
  rotation and shift, horizontal flip, random erasing.
- At test time we average the prediction of the image and its mirror image.
- One threshold per attribute, tuned on the validation set and folded into the
  model output, so the platform rule (probability above 0.5 = present) still works.

We made two runs: Run 1 with input 256 x 128 and Run 2 with input 320 x 160.

## Results (Codabench, hidden test set)

| Submission | Avg    |
|------------|--------|
| Run 1      | 0.7565 |
| Run 2      | 0.7584 |
| Rank 1     | 0.8000 |

Run 2 is our best submission (submission ID 996154). Full results and analysis
are in the report.

## Files

- `upar_par.py`: training, threshold tuning, packaging and local test.
- `analyze.py`: per-dataset and per-attribute result tables.
- `download_rest.py`: downloads the PA-100K and PETA images.
- `demo.py`: runs the submitted model on one image and prints its attributes.
- `submission_320/`: the submitted code (`run.py`, `metadata.yaml`, weights).

## How to run

Install: Python 3.11, PyTorch with CUDA, torchvision, numpy, pandas, Pillow.

# submission 1
<img width="1280" height="658" alt="submission" src="https://github.com/user-attachments/assets/47beb83b-59ea-43b4-8a43-b85a24ec5420" />

# submission 2
<img width="1280" height="663" alt="submission_2" src="https://github.com/user-attachments/assets/cf7a41fb-a209-430e-8fa0-60b3fdf6fa4b" />

# leaderboard
<img width="1272" height="662" alt="leaderboard" src="https://github.com/user-attachments/assets/66584256-8a8e-4c8d-99de-66dbd0542ad9" />

# rank
<img width="1272" height="662" alt="leaderboard" src="https://github.com/user-attachments/assets/66584256-8a8e-4c8d-99de-66dbd0542ad9" />

# submission folders on drive
https://drive.google.com/drive/folders/19goBZ_G6ONS_MI8_pLQ7klzsiuCzqNoO?usp=sharing
