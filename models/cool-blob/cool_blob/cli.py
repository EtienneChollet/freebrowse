"""
Run COOL-BLOB on one scan and successive cumulative annotation NIfTI files.
"""
import argparse
from pathlib import Path

import nibabel as nib

from cool_blob.session import InteractiveSession


def main() -> None:
    """
    Save one native-space prediction per annotation file while retaining correction context.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--modality", default="ct", choices=("ct", "mri", "other"))
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    if len(arguments.annotations) != len(arguments.output):
        parser.error("Supply one --output for each --annotations file")
    sources = {arguments.volume.resolve(), *(path.resolve() for path in arguments.annotations)}
    outputs = [path.resolve() for path in arguments.output]
    if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
        parser.error("Output files must be distinct and must not already exist")
    if sources.intersection(outputs):
        parser.error("Output files must not overwrite the scan or annotations")
    session = InteractiveSession(arguments.model_dir, arguments.volume, arguments.modality)
    for annotation, output in zip(arguments.annotations, outputs):
        prediction = session.predict(annotation)
        nib.save(prediction, output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
