"""Check real checkpoint inference and sparse prompts on one AMOS scan.

The reference labels only choose test clicks. The model receives the scan and sparse annotations;
it never receives the reference segmentation. All generated files go into a separate output folder.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
import yaml
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from fiery_flame import FieryFlameRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    image = nib.load(args.volume)
    reference = nib.load(args.reference)
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
        raise ValueError("Scan and reference geometry must match")
    truth = np.asarray(reference.dataobj)
    annotation = np.zeros(image.shape, np.uint8)
    config_path = Path(__file__).resolve().parents[1] / "models" / "fiery-flame" / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    runner = FieryFlameRunner()
    cache = SimpleNamespace()
    records = []
    # AMOS IDs: 6 liver, 2 right kidney, 3 left kidney; these seed locations are test-only.
    for user_id, source_id in [(1, 6), (2, 2), (3, 3)]:
        membership = truth == source_id
        z = int(membership.sum(axis=(0, 1)).argmax())
        center = np.unravel_index(distance_transform_edt(membership[:, :, z]).argmax(), image.shape[:2])
        x, y = map(int, center)
        annotation[x, y, z] = user_id
        if user_id == 3:
            for dx in range(-8, 9):
                if 0 <= x + dx < image.shape[0] and membership[x + dx, y, z]:
                    annotation[x + dx, y, z] = user_id
        # Explicit background evidence is distinct from untouched annotation voxels.
        annotation[20, 20, z] = 255
        annot_path = args.output / f"annotations_{user_id}.nii.gz"
        nib.save(nib.Nifti1Image(annotation, image.affine), annot_path)
        started = time.perf_counter()
        prediction = runner.predict(args.volume, annot_path, cache, config, "ct")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        values, counts = np.unique(np.asarray(prediction.dataobj), return_counts=True)
        if prediction.shape != image.shape or not np.allclose(prediction.affine, image.affine):
            raise AssertionError("Prediction did not preserve original scan geometry")
        if not set(values).issubset(set(range(user_id + 1))):
            raise AssertionError("Prediction contains an unknown user class")
        if user_id not in values:
            raise AssertionError("Newly prompted class is absent from the prediction")
        nib.save(prediction, args.output / f"prediction_{user_id}.nii.gz")
        record = {
            "foreground_classes": user_id, "seconds": elapsed,
            "new_annotation_voxel": [x, y, z],
            "voxel_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "validation.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
