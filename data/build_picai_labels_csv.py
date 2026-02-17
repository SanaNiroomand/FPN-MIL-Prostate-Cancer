"""
Build data/picai_labels.csv from your label table (e.g. from main.ipynb).
Run after you have built the labels dataframe with: patient_id, isup, cs_pca, fold, has_all_modalities.
Keeps only patients with all modalities and adds image_id = patient_id for MIL code compatibility.
"""
import pandas as pd
from pathlib import Path

# Example: if you have labels in memory from main.ipynb, save with:
#   labels[labels["has_all_modalities"]].assign(image_id=lambda df: df["patient_id"].astype(str))[
#       ["patient_id", "image_id", "cs_pca", "fold"]
#   ].to_csv("data/picai_labels.csv", index=False)

def build_from_dataframe(labels_df, out_path="data/picai_labels.csv"):
    """labels_df must have: patient_id, cs_pca, fold; optional has_all_modalities."""
    if "has_all_modalities" in labels_df.columns:
        df = labels_df[labels_df["has_all_modalities"]].copy()
    else:
        df = labels_df.copy()
    df["image_id"] = df["patient_id"].astype(str)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["patient_id", "image_id", "cs_pca", "fold"]].to_csv(out, index=False)
    print("Saved", out, "with", len(df), "rows")
    return df

# If run as script, you need to load your labels (e.g. from a pickle or CSV export from the notebook)
# if __name__ == "__main__":
#     labels = pd.read_csv("data/labels_export.csv")  # export from notebook
#     build_from_dataframe(labels)
