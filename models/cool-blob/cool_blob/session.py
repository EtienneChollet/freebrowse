"""
Use cumulative annotation files for successive Segmento v1 predictions on one scan.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import nibabel as nib
import yaml

from cool_blob.inference import runner


class InteractiveSession:
    """
    Retain private correction context for a single target volume.

    Parameters
    ----------
    model_directory : pathlib.Path
        Directory containing config.yml, model.yml, and the tracked checkpoint parts.
    volume_path : pathlib.Path
        Existing 3D NIfTI scan. The source file is never modified.
    modality : str, default="ct"
        "ct" uses Hounsfield clipping; other modalities use percentile clipping.

    Notes
    -----
    Annotation files use 0 for unmarked, 255 for background, and 1–254 for foreground IDs.
    Keep this object across cumulative corrections; separate objects never share prior logits.
    """

    def __init__(self, model_directory: Path, volume_path: Path, modality: str = "ct") -> None:
        """
        Resolve package-local model files and start with no previous prediction.
        """
        directory = Path(model_directory).resolve()
        with (directory / "config.yml").open() as stream:
            self.config: Dict[str, Any] = yaml.safe_load(stream)
        for name in ("training_config_path", "checkpoint_path"):
            self.config[name] = str((directory / self.config[name]).resolve())
        self.volume_path = Path(volume_path).resolve(strict=True)
        self.modality = modality
        self.cache = SimpleNamespace()

    def predict(self, annotation_path: Path) -> nib.Nifti1Image:
        """
        Return native-space class IDs and retain compatible raw logits for the next correction.

        Parameters
        ----------
        annotation_path : pathlib.Path
            Cumulative annotation NIfTI matching the scan's shape and affine.

        Returns
        -------
        nibabel.Nifti1Image
            Segmentation on the original scan grid, with background 0 and stable class IDs.
        """
        return runner.predict(
            self.volume_path, Path(annotation_path).resolve(strict=True), self.cache,
            self.config, self.modality,
        )
