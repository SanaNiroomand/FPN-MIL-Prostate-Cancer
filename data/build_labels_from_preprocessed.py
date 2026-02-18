"""
Build data/picai_labels.csv from PI-CAI metadata and your preprocessed folds (0,1,2).

Usage:
  python data/build_labels_from_preprocessed.py \
    --metadata "/path/to/Metadata(for ISUP).csv" \
    --preprocessed "/path/to/nnUNet_raw_data" \
    --out data/picai_labels.csv

Or set paths below and run: python data/build_labels_from_preprocessed.py

Expects:
- Metadata CSV with patient id and ISUP (column names auto-detected).
- If CSV has study_id (or similar), case_id = patient_id_study_id; else case_id = patient_id.
- Preprocessed dirs: <preprocessed>/nnUNet_raw_data_fold0/.../imagesTr/*_0000.nii.gz (and fold1, fold2).
"""
import argparse
import pandas as pd
from pathlib import Path


def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def get_existing_case_ids_and_folds(preprocessed_root: Path, folds=(0, 1, 2)):
    """Scan imagesTr in each fold; return dict case_id -> fold."""
    case_to_fold = {}
    for fold in folds:
        images_tr = preprocessed_root / f"nnUNet_raw_data_fold{fold}" / f"Task2201_picai_fold{fold}" / "imagesTr"
        if not images_tr.exists():
            continue
        for p in images_tr.glob("*_0000.nii.gz"):
            case_id = p.name.replace("_0000.nii.gz", "")
            case_to_fold[case_id] = fold
    return case_to_fold


def main():
    p = argparse.ArgumentParser(description="Build picai_labels.csv from metadata + preprocessed case list")
    p.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV (e.g. Metadata(for ISUP).csv)")
    p.add_argument("--preprocessed", type=str, default=None, help="Root dir containing nnUNet_raw_data_fold0, fold1, fold2")
    p.add_argument("--out", type=str, default="data/picai_labels.csv", help="Output CSV path")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2], help="Folds to include")
    args = p.parse_args()

    # Default paths (Kaggle or local)
    data_root = Path("/kaggle/input/prostate-cancer-pi-cai-dataset") if Path("/kaggle/input").exists() else Path(".")
    preprocessed_root = Path(args.preprocessed or "/kaggle/working" if Path("/kaggle/working").exists() else Path("./picai_preprocessed"))
    metadata_path = Path(args.metadata or data_root / "Metadata(for ISUP).csv")
    if not metadata_path.exists():
        metadata_path = data_root / "Metadata(for ISUP without lesion info).csv"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Set --metadata to your PI-CAI ISUP CSV.")

    df = pd.read_csv(metadata_path)
    pid_col = find_col(df, ["patient_id", "patient", "case_id", "subject_id", "id"])
    isup_col = find_col(df, ["isup", "isup_grade", "isup_grade_group", "grade_group"])
    study_col = find_col(df, ["study_id", "study", "exam_id"])

    if pid_col is None or isup_col is None:
        raise RuntimeError(f"Could not find patient/id and ISUP columns. Columns: {list(df.columns)}")

    df = df.rename(columns={pid_col: "patient_id", isup_col: "isup"})
    df["patient_id"] = df["patient_id"].astype(str)
    df["isup"] = pd.to_numeric(df["isup"], errors="coerce")

    if study_col:
        df["study_id"] = df[study_col].astype(str)
        df["case_id"] = df["patient_id"] + "_" + df["study_id"]
    else:
        df["case_id"] = df["patient_id"]

    case_to_fold = get_existing_case_ids_and_folds(preprocessed_root, folds=args.folds)
    if not case_to_fold:
        raise FileNotFoundError(f"No preprocessed cases found under {preprocessed_root} (fold 0,1,2). Run picai_prep first.")

    # Build labels for each preprocessed case
    meta_by_case = df.set_index("case_id") if study_col else None
    meta_by_pid = df.drop_duplicates("patient_id", keep="first").set_index("patient_id")

    rows = []
    for case_id, fold in case_to_fold.items():
        pid = case_id.split("_")[0] if "_" in case_id else case_id
        if meta_by_case is not None and case_id in meta_by_case.index:
            row = meta_by_case.loc[case_id]
        elif pid in meta_by_pid.index:
            row = meta_by_pid.loc[pid]
        else:
            continue
        isup = pd.to_numeric(row.get("isup", -1), errors="coerce")
        cs_pca = 1 if isup >= 2 else 0
        rows.append({"patient_id": pid, "image_id": case_id, "cs_pca": cs_pca, "fold": fold})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print("Saved", out_path, "with", len(out_df), "rows (fold counts:", out_df["fold"].value_counts().sort_index().to_dict(), ")")


if __name__ == "__main__":
    main()
