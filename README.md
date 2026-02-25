# FPN-MIL for Prostate Cancer (PI-CAI)

Pipeline for **clinically significant prostate cancer (csPCa)** detection using **FPN-MIL** (Feature Pyramid Network + Multiple Instance Learning) on the [PI-CAI](https://pi-cai.grand-challenge.org/) dataset. The workflow: **preprocess** MHA → nnU-Net raw, **extract** FPN features (one patch per slice, one bag per case), **train** an attention-based MIL model, and **visualize** attention heatmaps on MRI slices.

---

## Pipeline overview

```
PI-CAI (MHA)  →  [1] Preprocess  →  nnU-Net raw (T2W/ADC/HBV)
                        ↓
                [2] Feature extraction  →  C4/C5 patch features + labels CSV
                        ↓
                [3] FPN-MIL training    →  checkpoint + attention
                        ↓
                [4] Attention viz       →  heatmap overlay on T2W
```

| Step | Notebook | Input | Output |
|------|----------|--------|--------|
| 1 | `pi-cai-preprocess.ipynb` | PI-CAI dataset (Kaggle or local) | `nnUNet_raw_data_fold{N}/` (T2W, ADC, HBV) |
| 2 | `pi-cai-feature-extraction.ipynb` | Preprocessed roots (folds 0–2 and 3–4) + optional masks | `picai_extracted_features/`, `picai_labels.csv` |
| 3 | `fpn-mil.ipynb` | Feature dir + labels CSV | Trained model, `attention/<case_id>.pt` |
| 4 | (in `fpn-mil.ipynb`) | Saved attention + optional NIfTI roots | Bar chart + red heatmap overlay on T2W |

---

## Data

- **PI-CAI dataset**: [Prostate Cancer (PI-CAI) Dataset](https://www.kaggle.com/datasets/varshithpsingh/prostate-cancer-pi-cai-dataset) on Kaggle (or official grand-challenge). Contains `picai_public_images_fold0` … `fold4` and `Metadata(for ISUP).csv`.
- **Labels**: csPCa proxy from ISUP ≥ 2; built in the feature-extraction notebook from metadata.
- **Prostate masks** (optional): [picai_labels](https://github.com/DIAGNijmegen/picai_labels) (e.g. Bosma22b) for ROI cropping.

---

## 1. Preprocessing (`pi-cai-preprocess.ipynb`)

Runs the official [picai_prep](https://github.com/DIAGNijmegen/picai_prep) pipeline: **MHA → nnU-Net raw** (resampled T2W, ADC, HBV).

- **Set** `KAGGLE_INPUT` (or local path) and `OUTPUT_ROOT`.
- **Set** `FOLDS`: e.g. `[0, 1, 2]` for one run, or `[3, 4]` for a second run (useful on Kaggle where output size is limited).
- **Run**: Install → Paths → PREPROCESSING cell → (optional) view preprocessed images.
- **Output**: `nnUNet_raw_data_fold{N}/Task2201_picai_fold{N}/imagesTr/` with `*_0000.nii.gz` (T2W), `*_0001` (ADC), `*_0002` (HBV).

For **all 5 folds** on Kaggle, run the notebook twice and save two versions: one with folds 0,1,2 and one with folds 3,4.

---

## 2. Feature extraction (`pi-cai-feature-extraction.ipynb`)

Extracts **ResNet18 + FPN** patch features from the preprocessed NIfTIs. **One 2D slice** (from the 3-channel volume) = **one patch**; **one case** = **one bag** for MIL.

- **Add inputs**: Two preprocessed datasets (folds 0,1,2 and 3,4) and optionally the PI-CAI dataset for labels.
- **Set** `PREPROCESSED_ROOTS`: list of `(path, [folds])`, e.g.  
  `("/kaggle/input/.../pi-cai-preprocess", [0,1,2]), ("/kaggle/input/.../pi-cai-preprocess-2", [3,4])`.
- **Optional**: Clone picai_labels, set `MASKS_DIR` for prostate ROI; run “Batch crop all cases” or let extraction crop on the fly.
- **Run**: Paths → labels → (optional batch crop) → write extractor script → run extraction.
- **Output**:  
  - `picai_extracted_features/multi_scale/<case_id>/<case_id>/`: `C4_patch_features.pt`, `C5_patch_features.pt`, `info_patches.h5`.  
  - `picai_labels.csv`: `image_id`, `fold`, `cs_pca`, `preprocessed_root`.

---

## 3. FPN-MIL training (`fpn-mil.ipynb`)

Self-contained training: **ISAB** encoders + **gated attention** per scale, then scale aggregation and classifier.

- **Add input**: Output of the feature-extraction notebook (feature folder + `picai_labels.csv`).
- **Set** `INPUT_ROOT`, `TRAIN_FOLDS`, `VAL_FOLDS` (e.g. train on 0,1,2,3 and validate on 4).
- **Run**: Config → Load labels/split → Dataset/loaders → Model → Training loop.
- **Output**: `checkpoints/best.pth`, and (after the attention cell) `attention/<case_id>.pt` for validation cases.

---

## 4. Attention visualization (in `fpn-mil.ipynb`)

- **Extract and save attention**: Loads best checkpoint, runs validation set with `forward_with_attention`, saves per-slice attention + coords + pred/label to `attention/<case_id>.pt`.
- **Visualize**:  
  - **Bar chart**: attention weight vs slice index.  
  - **Overlay on T2W** (if image roots are set): red heatmap on slices (light red = low attention, dark red = high attention).

Set **`IMAGE_ROOT`** and/or **`FOLD_TO_IMAGE_ROOT`** (e.g. `{0,1,2: path1, 3,4: path2}`) to the same preprocessed NIfTI roots used in feature extraction; the notebook will load T2W and overlay the attention.

---

## Requirements

- Python 3.10+ (tested on Kaggle).
- **Preprocessing**: `picai_prep`.
- **Feature extraction**: `SimpleITK`, `torch`, `h5py`, `pandas`; optional `nibabel`. picai_labels for masks (cloned in notebook).
- **Training**: `torch`, `scikit-learn`, `matplotlib`, `h5py`, `pandas`, `numpy`.

---

## Notebooks in this repo

| File | Description |
|------|-------------|
| `pi-cai-preprocess.ipynb` | MHA → nnU-Net raw (set `FOLDS` to [0,1,2] or [3,4] for two runs). |
| `pi-cai-feature-extraction.ipynb` | FPN feature extraction for all 5 folds from two preprocessed roots; optional prostate crop. |
| `fpn-mil.ipynb` | FPN-MIL training, attention extraction, and heatmap visualization. |
| `fpn-mil-retina.ipynb` | Alternative/experimental (e.g. Retina-style) FPN-MIL setup. |

---

## Quick start (Kaggle)

1. **Preprocess**  
   - Add [Prostate Cancer (PI-CAI) Dataset](https://www.kaggle.com/datasets/varshithpsingh/prostate-cancer-pi-cai-dataset).  
   - Open `pi-cai-preprocess.ipynb`, set `FOLDS = [0, 1, 2]`, run all, **Save version** (e.g. “pi-cai-preprocess”).  
   - Duplicate or change `FOLDS = [3, 4]`, run, **Save version** (e.g. “pi-cai-preprocess-2”).

2. **Feature extraction**  
   - Add both preprocess outputs + PI-CAI dataset.  
   - Open `pi-cai-feature-extraction.ipynb`, set `PREPROCESSED_ROOTS` to the two notebook output paths.  
   - Run all; **Save version** to keep `picai_extracted_features` and `picai_labels.csv`.

3. **Training**  
   - Add the feature-extraction notebook output.  
   - Open `fpn-mil.ipynb`, set `INPUT_ROOT` to that output.  
   - Run all. Optionally set `FOLD_TO_IMAGE_ROOT` to the two preprocess paths, then run the attention cells to get heatmaps on T2W.

---

## License

See repository license. PI-CAI data and picai tools have their own terms of use.
