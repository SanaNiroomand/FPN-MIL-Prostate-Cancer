"""
Run official PI-CAI preprocessing (picai_prep): MHA → nnU-Net raw.
Resamples T2W/ADC/HBV to shared voxel spacing (and optional uniform spacing + crop).

Requires: pip install picai_prep

Usage:
  python run_picai_prep.py --input /path/to/picai_public_images_fold0 --output /path/to/nnUNet_raw
  # Or for all folds, call in a loop from your notebook/shell.
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="PI-CAI preprocessing via picai_prep (MHA → nnU-Net raw)")
    p.add_argument("--input", type=str, required=True, help="Path to MHA archive (e.g. picai_public_images_fold0)")
    p.add_argument("--output", type=str, required=True, help="Path to nnU-Net raw data output")
    p.add_argument("--annotations", type=str, default=None, help="Optional: path to annotations (skips cases without)")
    p.add_argument("--task", type=str, default="Task2201_picai_baseline", help="nnUNet task name in dataset.json")
    p.add_argument("--json", type=str, default=None, help="Optional: reuse existing mha2nnunet_settings.json")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = output_dir.parent / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    settings_path = Path(args.json) if args.json else workdir / "mha2nnunet_settings.json"

    from picai_prep import MHA2nnUNetConverter

    if not settings_path.exists() or args.json is None:
        from picai_prep.examples.mha2nnunet.picai_archive import generate_mha2nnunet_settings
        generate_mha2nnunet_settings(
            archive_dir=str(input_dir),
            output_path=str(settings_path),
            annotations_dir=args.annotations,
            task=args.task,
        )

    archive = MHA2nnUNetConverter(
        scans_dir=str(input_dir),
        annotations_dir=args.annotations or str(input_dir),
        output_dir=str(output_dir),
        mha2nnunet_settings=str(settings_path),
    )
    archive.convert()
    archive.create_dataset_json()
    print("Done. nnU-Net raw data at:", output_dir)


if __name__ == "__main__":
    main()
