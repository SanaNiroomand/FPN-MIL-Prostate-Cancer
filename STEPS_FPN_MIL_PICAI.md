# FPN-MIL for PI-CAI: Steps (based on Multi-scale Attention-based MIL)

This guide adapts the [Multi-scale Attention-based MIL](https://github.com/marianamourao-37/Multi-scale-Attention-based-MIL) (MICCAI 2025) pipeline for the **PI-CAI** prostate MRI dataset. Their repo uses 2D mammograms and Mammo-CLIP; here we use 2D slices from 3D MRI and a generic backbone.

---

## 1. Map their repo to your project

| Their (VinDr-Mammo) | Your (PI-CAI) |
|---------------------|---------------|
| **Bag** = one mammogram (one image) | **Bag** = one **patient** (one 3D volume or set of 2D slices) |
| **Instances** = patches from that image | **Instances** = 2D slices (e.g. axial) or 2D patches from T2W (and optionally ADC/HBV stacked) |
| **Label** = Mass / Suspicious_Calcification | **Label** = `cs_pca` (ISUP ≥ 2) or ISUP grade |
| **Data** = `patient_id` / `image_id` in CSV + PNGs | **Data** = your `labels` table + MHA paths (T2W, ADC, HBV) |
| **Backbone** = Mammo-CLIP EfficientNet-B2 | **Backbone** = ResNet/EfficientNet (ImageNet or from scratch) |
| **Feature dir** = `feat_dir` with `patch_size-X` or `multi_scale` | Same idea: offline features under `feat_dir/multi_scale/patient_id/` with `C4_patch_features.pt`, `C5_patch_features.pt`, `info_patches.h5` |

Their pipeline supports:
- **Offline**: extract patch features once (backbone + FPN → save C4, C5 per bag), then train MIL on saved features.
- **Online**: load images in the dataloader, run backbone + FPN + MIL in one forward.

For PI-CAI you can do **offline** first (slice/patch → backbone+FPN → save features), then train MIL; or **online** with a `Dataset` that yields 2D slices per patient.

---

## 2. Steps to implement

### Step 1: Data and labels (you already have this in `main.ipynb`)

- Run your **preprocessing**: align ADC/HBV to T2W; save `t2w.mha`, `adc_to_t2w.mha`, `hbv_to_t2w.mha` per patient.
- Build **label table**: `patient_id`, `isup`, `cs_pca = (isup >= 2)`, `fold`, paths to the three modalities.
- Export a **CSV** that matches what the MIL code expects, e.g.:
  - `patient_id`, `image_id` (can equal `patient_id` for PI-CAI), `cs_pca`, `fold`.
  - Use `data/build_picai_labels_csv.py` to build `data/picai_labels.csv` from your label table.

### Step 2: Define “images” and patches for PI-CAI

- **Option A – 2D slices as instances**  
  For each patient, load T2W (and optionally ADC, HBV) and extract axial 2D slices (e.g. every slice or every 2nd). Each slice = one instance. Optionally stack 3 modalities as 3 channels (then use a 3-channel backbone).
- **Option B – 2D patches**  
  Extract a grid of 2D patches (e.g. 224×224 or 512×512) from each slice with optional overlap, similar to their `patch_size` and `overlap`. Each patch = one instance.

Their code expects either:
- **Offline**: one folder per bag with `C4_patch_features.pt`, `C5_patch_features.pt`, `info_patches.h5` (with `coords` and attributes like `patch_size`, `padding_*`).
- **Online**: a transform that, given an “image”, returns `(patches_tensor, coords, padding)`.

So you need a **PI-CAI slice/patch extractor** that outputs the same structure (list of patches + coordinates) so you can plug into their `BagDataset` / `Generic_MIL_Dataset` style.

### Step 3: Backbone + FPN (no Mammo-CLIP)

- **Backbone**: Use a standard 2D CNN (e.g. ResNet18/34, EfficientNet-B0) that outputs multi-scale feature maps (e.g. C4, C5). If you use 3-channel input (T2W+ADC+HBV), keep 3-channel first layer; if single-channel T2W, use 1-channel or repeat to 3 and use ImageNet pretrained.
- **FPN**: Copy or reimplement their **FeatureExtractors/FPN** logic: build FPN on top of backbone stages to get a pyramid, then for each patch you get C4 and C5 (or equivalent) feature maps. Their `offline_feature_extraction.py` saves `C4` and `C5` per bag; you do the same for each patient bag.

You don’t need Mammo-CLIP; only the **interface** must match: input = batch of patches, output = list of tensors `[C4_features, C5_features]` with consistent dimensions (they use `fpn_dim`).

### Step 4: Offline feature extraction for PI-CAI

- Write a script similar to their **offline_feature_extraction.py**:
  - Loop over patients (bags).
  - For each patient: load volume → extract 2D slices/patches → run backbone+FPN → get C4, C5 per patch.
  - Save:
    - `multi_scale/<patient_id>/C4_patch_features.pt`, `C5_patch_features.pt`
    - `info_patches.h5` with `coords` (e.g. slice index, row, col) and attributes (`patch_size`, `padding_*`, `img_height`, `img_width`, etc.).
- Use the same directory layout as they do so **Generic_MIL_Dataset** (or your PI-CAI variant) can load with `load_data(bag_dir, feat_pyramid_level='C4'/'C5')`.

### Step 5: MIL dataset and training (their `main.py` + MIL)

- **Dataset**: Either:
  - **Reuse their `Generic_MIL_Dataset`**: build a CSV with columns `patient_id`, `image_id` (can set `image_id = patient_id` for PI-CAI since one patient = one bag), and your label column (e.g. `cs_pca`). Point `args.data_dir`, `args.feat_dir`, `args.label` to your paths and label name.  
  Or
  - **Subclass** their dataset to read your CSV (patient_id, fold, cs_pca) and resolve paths to `feat_dir/multi_scale/<patient_id>/`.
- **Training**: Use their **MIL** setup:
  - `--mil_type pyramidal_mil`
  - `--multi_scale_model fpn`
  - `--scales 16 32 128` (or match what you used for patch sizes / FPN scales)
  - `--type_scale_aggregator gated-attention`
  - `--pooling_type gated-attention` or `pma`
  - `--type_mil_encoder isab` (or `mlp`)
  - `--deep_supervision`
  - `--label cs_pca` (or whatever column name you use)
  - `--feature_extraction offline`
- **Loss**: Binary (BCE) for `cs_pca`; they use `--weighted-BCE y` and dataset-specific weights—you can compute positive_fraction and set a weight for PI-CAI.

### Step 6: Evaluation and folds

- Use PI-CAI’s **5-fold** split: train on 4 folds, validate/test on 1; repeat and report mean ± std AUC (and sensitivity/specificity). Their `main.py` supports `--eval_scheme`, `--n_folds`, etc.; wire your fold column so the dataset only loads train or val/test according to the split.

### Step 7: Optional – ROI / heatmaps (lesion detection)

- Their **roi_eval** builds heatmaps from attention scores and evaluates detection (e.g. IoU with lesion boxes). For PI-CAI you could do the same if you have lesion annotations (e.g. from “Metadata with lesion info.csv”): map attention back to 3D or 2D and compare to ground-truth lesions.

---

## 3. Suggested file layout (mirroring their repo)

```
FPN-MIL Prostate Cancer/
├── main.ipynb                    # Your preprocessing + labels (existing)
├── STEPS_FPN_MIL_PICAI.md        # This file
├── config_picai.py               # PI-CAI defaults (paths, label, folds)
├── data/
│   └── picai_labels.csv           # patient_id, cs_pca, fold, (paths)
├── Datasets/
│   ├── dataset_concepts.py        # Copy/adapt: BagDataset, Generic_MIL_Dataset for PI-CAI
│   └── dataset_utils.py           # Bags dataloader, transforms (patch extraction)
├── FeatureExtractors/
│   ├── FPN.py                     # From their repo (or reimplement)
│   └── backbone_2d.py             # ResNet/EfficientNet for 2D slices (no Mammo-CLIP)
├── MIL/
│   ├── MIL_models.py              # Pyramidal MIL, scale aggregator
│   ├── AttentionModels.py         # Gated attention, PMA, ISAB
│   └── MIL_experiment.py          # do_experiments()
├── utils/
│   ├── generic_utils.py
│   ├── data_split_utils.py
│   └── metrics.py
├── main.py                        # Same entrypoint as theirs; add PI-CAI dataset branch
└── offline_feature_extraction_picai.py   # Extract C4/C5 from PI-CAI volumes
```

---

## 4. Quick reference – their best config (from README)

For **classification** (adapt for PI-CAI):

- `--mil_type pyramidal_mil --multi_scale_model fpn`
- `--fpn_dim 256 --fcl_encoder_dim 256 --fcl_dropout 0.25`
- `--type_mil_encoder isab --trans_layer_norm True` (calcifications) or keep default (masses)
- `--pooling_type pma` or `gated-attention`
- `--type_scale_aggregator gated-attention --deep_supervision`
- `--scales 16 32 128`

Use **offline** features first (`--feature_extraction offline`) once you have `multi_scale/<patient_id>/C4_patch_features.pt` and `C5_patch_features.pt` plus `info_patches.h5` for each patient.

---

## 5. References

- **Multi-scale Attention-based MIL (MICCAI 2025)**: [GitHub – marianamourao-37/Multi-scale-Attention-based-MIL](https://github.com/marianamourao-37/Multi-scale-Attention-based-MIL)
- **PI-CAI**: Use your existing `main.ipynb` for preprocessing and label construction; align CSV and folder layout with the table in Section 1 above.
