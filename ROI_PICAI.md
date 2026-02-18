# Using prostate ROI masks with PI-CAI (for FPN-MIL)

PI-CAI **does** provide prostate masks you can use as an ROI.

---

## What PI-CAI provides

From the official [picai_labels](https://github.com/DIAGNijmegen/picai_labels) repo:

| Resource | Description |
|----------|-------------|
| **Prostate whole-gland (ROI)** | `anatomical_delineations/whole_gland/AI/Bosma22b/` — **AI-derived** prostate segmentations (one NIfTI per case, e.g. `10000_1000000.nii.gz`). Use these as ROI masks. |
| **csPCa lesion delineations** | Human expert and AI lesion masks (for detection/segmentation tasks, not required for ROI). |

The **whole-gland** masks are automated (Bosma et al.), so they can occasionally be wrong; for FPN-MIL ROI they are usually good enough to restrict patches/slices to the gland.

---

## Where to get the masks (you need this separately)

The **prostate whole-gland masks are not in the main PI-CAI image dataset**. They live in the official annotations repo **picai_labels**.

| Option | What to do |
|--------|------------|
| **Easiest (Kaggle)** | In your ROI notebook, run the cell that **clones picai_labels from GitHub**. It runs `git clone https://github.com/DIAGNijmegen/picai_labels.git` and uses the folder `anatomical_delineations/whole_gland/AI/Bosma22b/`. No extra Kaggle dataset needed. |
| **Kaggle dataset** | If you prefer to add an input: upload the [picai_labels](https://github.com/DIAGNijmegen/picai_labels) repo (or just the `anatomical_delineations` folder) as a Kaggle dataset, add it to your notebook, and set `MASKS_DIR` to the path to `Bosma22b` inside that dataset (e.g. `/kaggle/input/your-dataset/anatomical_delineations/whole_gland/AI/Bosma22b`). |
| **Local** | Clone or download: [https://github.com/DIAGNijmegen/picai_labels](https://github.com/DIAGNijmegen/picai_labels). You need the folder **`anatomical_delineations/whole_gland/AI/Bosma22b/`**. Files are named `patient_id_study_id.nii.gz` (e.g. `10000_1000000.nii.gz`), matching your preprocessed case IDs. |

**Path layout:** `.../Bosma22b/10000_1000000.nii.gz` (one NIfTI per case).

---

## Using the ROI in your pipeline

1. **Align mask to your preprocessed images**
   - Your nnU-Net NIfTIs (from picai_prep) are already resampled. The whole-gland masks in picai_labels are at **T2W resolution** (or original); you may need to **resample the mask** to the same grid as your preprocessed volume (e.g. with SimpleITK or nibabel) so shapes and spacing match.

2. **Crop or mask before slice/patch extraction**
   - **Option A – Crop to bounding box:** Compute the 3D bounding box of the prostate mask, crop the T2W/ADC/HBV volumes to that box, then extract 2D slices or patches from the crop. Reduces field of view to the gland.
   - **Option B – Mask and extract inside gland:** For each slice (or patch), keep only pixels where the mask > 0 (or use the mask to weight instances). You can drop slices that have almost no prostate (e.g. mask sum < threshold).

3. **In your FPN-MIL code**
   - When you extract instances (slices or patches), do it **after** cropping to the ROI (or only include patches that overlap the mask). That way each bag only contains instances from the prostate region.

---

## Minimal code idea (after you have the mask aligned)

```python
# Pseudocode: load volume and mask (same grid), crop to mask bbox, then extract slices
import nibabel as nib
import numpy as np

vol = nib.load("case_0000.nii.gz").get_fdata()   # T2W
mask = nib.load("10000_1000000.nii.gz").get_fdata()  # prostate, resampled to same grid as vol

# Bounding box of prostate
inds = np.where(mask > 0)
zmin, zmax = inds[0].min(), inds[0].max()
ymin, ymax = inds[1].min(), inds[1].max()
xmin, xmax = inds[2].min(), inds[2].max()

vol_roi = vol[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
# Then extract 2D slices from vol_roi for MIL (and same crop for ADC/HBV)
```

---

## References

- **picai_labels:** [https://github.com/DIAGNijmegen/picai_labels](https://github.com/DIAGNijmegen/picai_labels)
- **Whole-gland masks:** `anatomical_delineations/whole_gland/AI/Bosma22b/` (AI-derived; cite the challenge and Bosma et al. if you use them).
