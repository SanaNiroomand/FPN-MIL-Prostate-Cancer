# Ready-to-use preprocessing for PI-CAI

Yes. The **official**, MammoCLIP-style preprocessing for PI-CAI is **picai_prep**: a pip-installable pipeline maintained by the challenge organizers (DIAGNijmegen).

## What it does (vs MammoCLIP)

| | **MammoCLIP (VinDr)** | **PI-CAI (picai_prep)** |
|---|------------------------|--------------------------|
| **Output** | Preprocessed 2D PNGs (you download) | You run the pipeline locally/Docker |
| **Format** | 2D images per view | 3D MHA → nnU-Net raw (resampled, optional crop) |
| **Tool** | External preprocessing by Ghosh et al. | **picai_prep** (official PI-CAI) |

There is **no** single “download preprocessed PI-CAI images” bundle like VinDr’s MammoCLIP PNGs. You get the public MHA from Zenodo/Kaggle and run **picai_prep** to get a standardized, resampled (and optionally cropped) dataset.

## picai_prep pipeline

1. **DICOM → MHA**  
   If you have DICOM, convert to MHA (PI-CAI public data is usually already MHA).

2. **MHA → nnU-Net raw** (main step)  
   - Resamples T2W, ADC, HBV to a **shared voxel spacing per patient** (aligned volumes).  
   - Optionally: **uniform spacing** and **centre crop** (e.g. `matrix_size`, `spacing` in settings).  
   - Output: nnU-Net-style folder with 4D NIfTI (T2W+ADC+HBV) per case.

3. **nnU-Net → nnDetection raw**  
   Optional, for detection baselines (nnDetection).

## Install and use

```bash
pip install picai_prep
```

- **Generate settings** for the PI-CAI MHA layout (per fold or merged archive):

```bash
python -m picai_prep mha2nnunet_settings --structure picai_archive \
  --input /path/to/picai_public_images_fold0 \
  --output /workdir/mha2nnunet_settings_fold0.json
```

- **Run MHA → nnU-Net** conversion (resampling + optional crop):

```bash
python -m picai_prep mha2nnunet \
  --input /path/to/mha/archive \
  --output /path/to/nnUNet_raw_data \
  --json /workdir/mha2nnunet_settings.json
```

Or from Python:

```python
from picai_prep import MHA2nnUNetConverter
from picai_prep.examples.mha2nnunet.picai_archive import generate_mha2nnunet_settings

# 1) Generate settings (optional: add annotations_dir for cases with labels)
generate_mha2nnunet_settings(
    archive_dir="/path/to/picai_public_images_fold0",
    output_path="/workdir/mha2nnunet_settings.json",
)

# 2) Convert MHA → nnU-Net raw (resampled, aligned)
archive = MHA2nnUNetConverter(
    scans_dir="/path/to/mha/archive",
    output_dir="/path/to/nnUNet_raw_data",
    mha2nnunet_settings="/workdir/mha2nnunet_settings.json",
)
archive.convert()
archive.create_dataset_json()
```

## PI-CAI folder layout

**picai_prep** expects MHA archives like:

```
/path/to/archive/
├── [patient_id]/
│   ├── [patient_id]_[study_id]_t2w.mha
│   ├── [patient_id]_[study_id]_adc.mha
│   └── [patient_id]_[study_id]_hbv.mha
```

The Kaggle PI-CAI layout is **per-fold**:  
`picai_public_images_fold0/10000/10000_1000000_t2w.mha`  
→ so `archive_dir` should point at **one fold** (e.g. `picai_public_images_fold0`) and `patient_id` = `10000`, `study_id` = `1000000`. Run the pipeline once per fold or merge into one archive and run once.

## Using preprocessed data for FPN-MIL

After **picai_prep** (MHA → nnU-Net):

- Use the **nnU-Net raw** NIfTIs (resampled T2W/ADC/HBV) as input to your **slice/patch extraction** for FPN-MIL (2D slices or 2D patches from the 3D volumes).
- Your existing “align ADC/HBV to T2W” step is effectively replaced by picai_prep’s resampling; you can keep your label table and fold logic and point paths to the nnU-Net output.

## References

- **picai_prep**: [https://github.com/DIAGNijmegen/picai_prep](https://github.com/DIAGNijmegen/picai_prep)  
- **PI-CAI challenge**: [https://pi-cai.grand-challenge.org/](https://pi-cai.grand-challenge.org/)  
- **picai_baseline** (data prep + baselines): `pip install picai_baseline`, [https://github.com/DIAGNijmegen/picai_baseline](https://github.com/DIAGNijmegen/picai_baseline)
