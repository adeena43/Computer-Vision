# Task 2 — Multi-Modal Cardiac Image Fusion (CT + MRI)

## Dataset
[Heart CT & MRI Dataset](https://www.kaggle.com/datasets/ziya07/heart-ct-and-mri-dataset)
(only one CT/MRI slice pair — Patient_001 — is kept in `data/`, per the 100MB rule).

**Note on this dataset's content:** the Kaggle download is organized as
`synthetic_slices/Patient_XXX/{CT,MRI}/slice_NNN.png` — i.e. the dataset's own author
labels and ships it as procedurally generated (synthetic) slices, not real clinical scans;
each slice is a simple binary/solid disc, not textured anatomy. This is a property of the
publicly available dataset itself, not something altered here — `data/ct_slice.png` and
`data/mri_slice.png` are the exact, unmodified files extracted from the linked Kaggle
archive. The fusion pipeline and math below are fully general and will produce much richer
results on any dataset with real textured CT/MRI anatomy; the current outputs simply
reflect the flat geometry of this particular source dataset.

CT and MRI slice counts differ per patient (e.g. Patient_001 has 66 CT slices vs. 84 MRI
slices), so there is no explicit 1:1 index correspondence file. The slice pair used here
was selected by matching **relative depth position** (roughly the middle slice of each
stack) as a reasonable proxy for "the same anatomical level," and this assumption is
called out explicitly in the notebook.

## Files
- `data/ct_slice.png`, `data/mri_slice.png` — the real CT and MRI slices used (Patient_001,
  proportionally matched middle slices — see note above).
- `modal_fusion.ipynb` — full fusion pipeline notebook.
- `output/` — equalized modalities, colormapped heatmaps, fused output, and comparison
  charts.

## Fusion Weighting Logic

`cv2.addWeighted(ct_heat, ALPHA_CT, mri_heat, BETA_MRI, GAMMA_OFFSET)` combines the two
colormapped modalities as `dst = alpha*CT + beta*MRI + gamma`.

- **`ALPHA_CT = 0.65`** — the CT channel is weighted more heavily because CT has far
  crisper anatomical/bone boundaries; a higher alpha keeps those edges dominant in the
  fused result.
- **`BETA_MRI = 0.35`** — the MRI channel contributes soft-tissue texture/contrast, but
  at a lower weight so it augments rather than overwhelms the sharp CT edges.
- **`GAMMA_OFFSET = 0`** — no constant brightness offset was needed after equalization.

These values were chosen empirically: lower CT weights (≈0.5) made chamber boundaries
noticeably blurrier, while higher CT weights (≈0.8) suppressed almost all visible MRI
soft-tissue texture. 0.65/0.35 was the best visual trade-off tested. Feel free to tweak
`ALPHA_CT` / `BETA_MRI` in the notebook and re-run to explore this trade-off.

## Processing Pipeline

1. **Load Modalities** — load CT and MRI slices, verify identical dimensions (resize if
   needed) since pixel-wise fusion requires aligned matrices.
2. **Histogram Equalization** — applied **independently** to CT and MRI (their
   intensity distributions carry different physical meaning, so a shared equalization
   curve would distort one relative to the other).
3. **Color Mapping** — CT → `COLORMAP_BONE` (structural), MRI → `COLORMAP_JET`
   (soft-tissue); a naive 50/50 overlay is shown first for baseline comparison.
4. **Weighted Fusion** — `cv2.addWeighted` with the alpha/beta weighting described above.
5. **Log & Gamma Correction of the fused result** — because two already-bright heatmaps
   were added together, some pixels risk clipping to pure black or pure white. A log
   transform recovers shadow detail; a gamma (`gamma=0.8`) transform lifts midtones
   without further blowing out highlights. The notebook prints the percentage of
   crushed/blown pixels before vs. after correction to quantify the improvement.
6. **Comparative Analysis** — CT-only, MRI-only, fused, and fused+gamma-corrected panels
   are displayed side-by-side for visual/mathematical validation.

## How to Run
```bash
cd Task_2_Cardiac_Fusion
jupyter notebook modal_fusion.ipynb
```
Run all cells top-to-bottom. Outputs are written to `output/`.