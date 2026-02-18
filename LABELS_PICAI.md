# What to do about labels (PI-CAI → FPN-MIL)

You need one **labels CSV** that the FPN-MIL pipeline can read: each row = one **case** (one preprocessed volume) with its **cs_pca** label and **fold**.

---

## 1. Where labels come from

- **PI-CAI metadata** (with the images or from [picai_labels](https://github.com/DIAGNijmegen/picai_labels)):
  - **ISUP grade** (and optionally study_id) so you can define **cs_pca = 1 if ISUP ≥ 2, else 0**.
  - Often in: `Metadata(for ISUP).csv` (Kaggle dataset) or `clinical_information/marksheet.csv` (picai_labels). Column names can vary (e.g. `patient_id`, `study_id`, `isup` / `isup_grade`).
- Your **preprocessed data** (picai_prep output) gives you the **case IDs** that actually exist: they are the filenames in `imagesTr`, e.g. `10000_1000000_0000.nii.gz` → case_id = `10000_1000000`.

So: **labels** come from the metadata; **which cases to keep** and **which fold they’re in** come from your preprocessed folds (e.g. fold 0, 1, 2).

---

## 2. What the MIL CSV should look like

The FPN-MIL dataset (and your `Datasets/dataset_picai.py`) expects a table with at least:

| Column     | Meaning |
|-----------|--------|
| `patient_id` | Patient identifier (can be the first part of case_id). |
| `image_id`   | Bag/instance set ID. For PI-CAI one bag = one case → set **image_id = case_id** (e.g. `10000_1000000`). |
| `cs_pca`     | Binary label: 1 = clinically significant PCa (ISUP ≥ 2), 0 = not. |
| `fold`       | 0, 1, or 2 (must match which preprocessed folder the case came from). |

So each row = one case (one preprocessed volume), with its label and fold.

---

## 3. Steps to build the CSV

1. **Load PI-CAI metadata**  
   - From the Kaggle dataset: e.g. `Metadata(for ISUP).csv` (or “Metadata(for ISUP without lesion info).csv”).  
   - Or from picai_labels: `clinical_information/marksheet.csv` (and use lesion/ISUP info if available elsewhere).

2. **Get ISUP (and study_id if needed)**  
   - Ensure you have a column for ISUP grade (e.g. `isup`, `isup_grade`, `grade_group`).  
   - If the table has both `patient_id` and `study_id`, form **case_id = `patient_id` + `_` + `study_id`** (e.g. `10000_1000000`). If the CSV only has one ID column that already looks like `10000_1000000`, use that as **case_id**.

3. **Compute cs_pca**  
   - **cs_pca = 1** when ISUP ≥ 2, else **cs_pca = 0**.

4. **Restrict to cases that exist in your preprocessed data**  
   - Scan your preprocessed output (e.g. `nnUNet_raw_data_fold0/.../imagesTr/`, and same for fold1, fold2).  
   - From each `*_0000.nii.gz` filename you get the **case_id** (strip `_0000.nii.gz`).  
   - Build a list of (case_id, fold). Only keep rows in your labels table for these case_ids.

5. **Attach fold**  
   - For each case_id, set **fold** = 0, 1, or 2 according to which folder you found it in.

6. **Save the CSV**  
   - Columns: `patient_id`, `image_id`, `cs_pca`, `fold` (and optionally `isup`, `case_id`).  
   - **image_id** = **case_id** (so the MIL loader knows which bag/case).  
   - Save as e.g. `data/picai_labels.csv`.

Then point your FPN-MIL config/dataloader to this CSV and to the preprocessed (and optionally ROI‑cropped) data.

---

## 4. Quick option: use the script

Use **`data/build_labels_from_preprocessed.py`**. It:

- Loads metadata from a path you set (e.g. Kaggle `Metadata(for ISUP).csv` or picai_labels marksheet).
- Scans your preprocessed `imagesTr` for folds 0, 1, 2 to get existing case_ids and their fold.
- Merges metadata with that list (by case_id if metadata has study_id, else by patient_id), computes cs_pca, and writes the labels CSV.

**Run after preprocessing** (once you have `nnUNet_raw_data_fold0/`, etc.):

```bash
python data/build_labels_from_preprocessed.py \
  --metadata "/path/to/Metadata(for ISUP).csv" \
  --preprocessed "/path/to/root/with/nnUNet_raw_data_fold0" \
  --out data/picai_labels.csv
```

On Kaggle you can use defaults (metadata from input, preprocessed from `/kaggle/working`). Optional: `--folds 0 1 2`.
