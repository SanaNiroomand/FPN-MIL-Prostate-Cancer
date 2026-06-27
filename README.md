# FPN-MIL for Prostate Cancer (PI-CAI)

Weakly supervised detection of **clinically significant prostate cancer (csPCa)** from biparametric MRI, using a **multi-scale attention-based Multiple Instance Learning** pipeline (FPN-MIL). Each exam is a *bag* of slice instances, and the model trains on **exam-level labels only** — no lesion masks.

> Adapted from the multi-scale attention MIL idea and applied to [PI-CAI](https://pi-cai.grand-challenge.org/). The focus here is the practical pipeline — preprocessing, ROI-aware cropping, and feature/instance design — rather than architectural novelty.

---

## Results

Our final ROI-aware, slice-based FPN-MIL reached **validation ROC-AUC ≈ 0.66** on PI-CAI fold 4 (case-level, weakly supervised).

| Metric | Score |
|---|---|
| ROC-AUC | **0.663** |
| Average Precision | 0.616 |
| PI-CAI score | 0.640 |
| Balanced accuracy | 0.630 |
| F1 | 0.581 |

### Comparison with related PI-CAI methods

| Method | Setting | AUROC | AP | PI-CAI score |
|---|---|---|---|---|
| **Ours** (ROI-aware slice FPN-MIL) | Validation, case-level, **weakly supervised** | 0.663 | 0.616 | 0.640 |
| Karagoz nnU-Net | Hidden test | 0.889 | 0.614 | 0.752 |
| Pooch nnU-Net (fully supervised) | Test (n=1000) | 0.890 | 0.650 | 0.770 |
| Z-SSMNet | Closed test | 0.909 | 0.690 | 0.800 |

> ⚠️ Not an apples-to-apples ranking. The other methods are **fully supervised** (lesion-level) and evaluated on **hidden/closed test** sets, while ours is **weakly supervised** and evaluated on the validation split. Notably, our **AP is competitive with strong nnU-Net baselines** despite using only exam-level labels — and the model produces interpretable instance-level attention, useful when lesion annotations aren't available.

---

## Key takeaways

- **Preprocessing > architecture (under weak supervision).** Instance quality and background removal mattered more than the aggregator. Moving to a **prostate ROI-aware** pipeline was the single biggest gain.
- **ROI-aware cropping** (whole-gland masks → crop around the prostate) lifted validation ROC-AUC from earlier internal variants to ≈ 0.66:

  | Variant | Val ROC-AUC |
  |---|---|
  | Earlier patch-based (no ROI) | ~0.56 |
  | Earlier slice-based (no ROI) | ~0.60 |
  | **ROI-aware slice-based (final)** | **~0.66** |

- **Attention is interpretable.** Per-slice attention can be plotted as bar charts or overlaid as heatmaps on T2W slices to see which slices drove each prediction.

**Future work:** stronger pretrained / domain-adapted feature extractors, systematic slice vs. ROI-patch instance comparison, repeated-run evaluation with better threshold selection, and deeper error analysis.

---

## Pipeline

```
PI-CAI (MHA) → [1] Preprocess (picai_prep) → nnU-Net raw (T2W/ADC/HBV)
                     ↓
             [2] ROI crop (whole-gland masks) + multi-scale feature extraction (ResNet18 + FPN)
                     ↓
             [3] FPN-MIL training (ISAB + gated attention, exam-level labels)
                     ↓
             [4] Attention visualization (heatmap overlay on T2W)
```

| Step | Notebook | Output |
|---|---|---|
| 1 | `pi-cai-preprocess.ipynb` | nnU-Net raw (T2W/ADC/HBV) |
| 2 | `pi-cai-feature-extraction.ipynb` | C4/C5 slice features + `picai_labels.csv` |
| 3 | `fpn-mil.ipynb` | Trained checkpoint + per-case attention |
| 4 | (in `fpn-mil.ipynb`) | Attention bar charts + T2W heatmaps |
| – | `fpn-mil-retina.ipynb` | Alternative/experimental setup |

**Setup at a glance:** labels are csPCa = `1[ISUP ≥ 2]` from PI-CAI metadata; each axial slice = one instance, each case = one bag; two feature scales (C4/C5); two ISAB blocks per scale + gated attention (instance & cross-scale); trained 150 epochs with AdamW, cosine schedule, weighted BCE for class imbalance, best checkpoint by validation ROC-AUC. Tested on Kaggle (Python 3.10+).

---

## Quick start (Kaggle)

1. **Preprocess** — add the [PI-CAI dataset](https://www.kaggle.com/datasets/varshithpsingh/prostate-cancer-pi-cai-dataset), run `pi-cai-preprocess.ipynb` (set `FOLDS`; run twice, e.g. `[0,1,2]` then `[3,4]`, due to Kaggle output limits).
2. **Extract features** — add both preprocess outputs, set `PREPROCESSED_ROOTS`, run `pi-cai-feature-extraction.ipynb`. Optionally clone [picai_labels](https://github.com/DIAGNijmegen/picai_labels) and set `MASKS_DIR` for prostate ROI cropping.
3. **Train** — set `INPUT_ROOT` to the feature output, run `fpn-mil.ipynb`. Set `FOLD_TO_IMAGE_ROOT` to the preprocess paths and run the attention cells for heatmaps.

---

## Data & credits

- **Dataset:** [PI-CAI](https://pi-cai.grand-challenge.org/) bpMRI (T2W/ADC/HBV).
- **Tools:** [picai_prep](https://github.com/DIAGNijmegen/picai_prep) (preprocessing), [picai_labels](https://github.com/DIAGNijmegen/picai_labels) (whole-gland masks, Bosma22b).
- **Authors:** Reyhaneh Khayaat-Zadeh-Mahani, Shaghayegh Mirjalili, Sana Niroumand, Sobhan Zamani — Dept. of Computer Engineering, Sharif University of Technology.

PI-CAI data and picai tools are subject to their own terms of use.
