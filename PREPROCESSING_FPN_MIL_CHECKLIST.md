# Additional preprocessing for FPN-MIL (on top of picai_prep)

**picai_prep** gives you resampled, aligned 3D volumes (T2W, ADC, HBV) in nnU-Net NIfTI format. For **FPN-MIL** you need a few more steps so the data matches what the MIL pipeline expects.

---

## What picai_prep already does ✓

- Resamples T2W, ADC, HBV to **shared voxel spacing** per case (aligned volumes).
- Outputs **nnU-Net raw**: `imagesTr/<case_id>_0000.nii.gz` (T2W), `_0001.nii.gz` (ADC), `_0002.nii.gz` (HBV) per patient.
- No extra alignment or resampling needed for FPN-MIL.

---

## What you still need for FPN-MIL

### 1. **Slice or patch extraction** (required)

FPN-MIL works on **2D instances** (one bag = one patient = many 2D instances). You must define instances from the 3D NIfTIs:

| Option | Description | Output for MIL |
|--------|-------------|----------------|
| **2D slices** | Extract axial (or other) slices from each volume. Each slice = one instance. | Stack T2W/ADC/HBV as 3 channels per slice, or use T2W only. |
| **2D patches** | Extract a grid of 2D patches (e.g. 224×224 or 512×512) per slice, with optional overlap. | One patch = one instance (3-channel if you stack modalities). |

You need code that:

- Loads the nnU-Net NIfTIs for each case (`*_0000.nii.gz`, `*_0001.nii.gz`, `*_0002.nii.gz`).
- Extracts 2D slices or patches (with coordinates, e.g. slice index, row, col).
- Outputs either:
  - **Online**: (patches/slices tensor, coords, padding) for the dataloader, or  
  - **Offline**: runs backbone+FPN on these and saves `C4_patch_features.pt`, `C5_patch_features.pt`, `info_patches.h5` per patient under `feat_dir/multi_scale/<patient_id>/`.

### 2. **Intensity normalization** (recommended)

- Normalize each modality (e.g. z-score or 0–1 clip per volume or per slice).
- Can be done **in the dataloader** (on-the-fly) or in a preprocessing script before feature extraction.
- Prevents one modality or scan from dominating.

### 3. **Labels and CSV** (required)

- **Label table**: `patient_id`, `cs_pca` (and optionally `fold`, `isup`), aligned with the **case IDs** from picai_prep (e.g. `10000_1000000`).
- **CSV for MIL**: e.g. `patient_id`, `image_id` (= `patient_id` for PI-CAI), `cs_pca`, `fold` — and paths pointing to either:
  - preprocessed NIfTI folders (if you use **online** FPN-MIL), or  
  - `feat_dir/multi_scale/<patient_id>/` (if you use **offline** features).

You can build this from PI-CAI metadata (e.g. `main.ipynb` + `data/build_picai_labels_csv.py`), then map metadata `patient_id` to the nnU-Net case ID (e.g. `patient_id_study_id`).

### 4. **Optional: prostate ROI**

- If you have prostate masks (e.g. from PI-CAI or another tool), crop or mask to the gland so instances are only inside the ROI. Reduces irrelevant background.
- If you don’t have masks, you can skip this and use all slices/patches (optionally with a simple intensity threshold to drop empty slices).

### 5. **Offline feature extraction** (only if you use offline FPN-MIL)

- Input: nnU-Net NIfTIs (or your extracted slices/patches).
- For each patient: extract patches/slices → run **backbone + FPN** → save `C4_patch_features.pt`, `C5_patch_features.pt`, `info_patches.h5` in `feat_dir/multi_scale/<patient_id>/`.
- Then the MIL dataloader only loads these `.pt`/`.h5` files (no image loading at training time).

---

## Summary

| Step | picai_prep | You add for FPN-MIL |
|------|------------|---------------------|
| Align modalities | ✓ | — |
| Resample to same grid | ✓ | — |
| 2D slices or patches | — | ✓ (from NIfTIs) |
| Normalization | — | ✓ (in loader or pre-step) |
| Labels CSV | — | ✓ (from PI-CAI metadata) |
| Optional ROI | — | ✓ if you have masks |
| Offline C4/C5 features | — | ✓ if using offline MIL |

So: **no extra imaging preprocessing** beyond picai_prep; the extra work is **defining instances** (slices/patches), **normalization**, **labels/CSV**, and optionally **offline feature extraction** and **ROI**.
