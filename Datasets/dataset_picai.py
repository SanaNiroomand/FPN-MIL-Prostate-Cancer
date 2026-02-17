"""
PI-CAI dataset for FPN-MIL (compatible with Multi-scale Attention-based MIL pipeline).
One bag = one patient. Expects offline features under:
  feat_dir/multi_scale/<patient_id>/C4_patch_features.pt, C5_patch_features.pt, info_patches.h5
CSV columns: patient_id, <label_col> (e.g. cs_pca), fold (optional).
"""
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_bag_data(bag_dir, feat_pyramid_level=None):
    """
    Load pre-extracted instance features for one patient (bag).
    Same interface as Generic_MIL_Dataset.load_data in the reference repo.
    """
    if feat_pyramid_level is None:
        x = torch.load(os.path.join(bag_dir, "patch_features.pt"))
    else:
        x = torch.load(os.path.join(bag_dir, f"{feat_pyramid_level}_patch_features.pt"))

    with h5py.File(os.path.join(bag_dir, "info_patches.h5"), "r") as f:
        bag_coords = np.array(f["coords"])

    # Sort by y then x (or by slice index for 3D)
    if bag_coords.ndim == 2 and bag_coords.shape[1] >= 2:
        sorted_indices = np.lexsort((bag_coords[:, 0], bag_coords[:, 1]))
    else:
        sorted_indices = np.arange(len(bag_coords))
    x = x[sorted_indices]
    return x


class PI_CAI_MIL_Dataset(Dataset):
    """
    MIL dataset for PI-CAI: one sample = one patient (bag).
    Expects CSV with patient_id, label_col (e.g. cs_pca), and optional fold.
    Expects offline FPN features in data_dir/feat_dir/multi_scale/<patient_id>/.
    """

    def __init__(self, df, data_dir, feat_dir, label_col="cs_pca", multi_scale_model="fpn"):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.feat_dir = Path(feat_dir)
        self.label_col = label_col
        self.multi_scale_model = multi_scale_model
        self.dir_path = self.data_dir / self.feat_dir / "multi_scale"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["patient_id"])
        bag_dir = self.dir_path / patient_id

        if self.multi_scale_model in ["fpn", "backbone_pyramid"]:
            c4 = load_bag_data(bag_dir, feat_pyramid_level="C4")
            c5 = load_bag_data(bag_dir, feat_pyramid_level="C5")
            x = [c4, c5]
        else:
            x = load_bag_data(bag_dir)

        y = torch.tensor(row[self.label_col], dtype=torch.long)
        return {"x": x, "y": y}


def collate_MIL_patches(batch):
    """Collate for FPN-MIL: batch list of bags into [C4_batch, C5_batch] and labels."""
    if isinstance(batch[0]["x"], list):
        x = [torch.stack([item["x"][i] for item in batch], dim=0) for i in range(len(batch[0]["x"]))]
    else:
        x = torch.stack([item["x"] for item in batch])
    y = torch.from_numpy(np.array([item["y"] for item in batch], dtype=np.float32))
    return {"x": x, "y": y}
