"""
PI-CAI defaults for FPN-MIL pipeline (aligned with Multi-scale Attention-based MIL repo).
Override via argparse in main.py or set DATA_ROOT / FEAT_DIR for your environment.
"""
from pathlib import Path

# Paths (override in main or env)
DATA_ROOT = Path(".")  # e.g. /kaggle/input/prostate-cancer-pi-cai-dataset or local copy
FEAT_DIR = "picai_extracted_features"  # where offline C4/C5 features will be saved
LABELS_CSV = "data/picai_labels.csv"   # patient_id, cs_pca, fold [, t2w_path, adc_path, hbv_path ]

# Label column name (must match CSV)
LABEL_COL = "cs_pca"  # binary: 1 if ISUP >= 2

# FPN-MIL (from their best config)
MIL_TYPE = "pyramidal_mil"
MULTI_SCALE_MODEL = "fpn"
SCALES = [16, 32, 128]
FPN_DIM = 256
FCL_ENCODER_DIM = 256
FCL_DROPOUT = 0.25
POOLING_TYPE = "gated-attention"  # or "pma"
TYPE_SCALE_AGGREGATOR = "gated-attention"
TYPE_MIL_ENCODER = "isab"  # or "mlp"
DEEP_SUPERVISION = True

# For PI-CAI: one bag = one patient (no image_id; use patient_id as bag id)
# In CSV you need: patient_id, cs_pca, fold. Optionally image_id = patient_id for compatibility.
