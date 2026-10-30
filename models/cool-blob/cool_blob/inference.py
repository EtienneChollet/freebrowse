"""
Amix inference with sparse multiclass prompts and native-space output.

Annotation value 0 is unmarked, 255 is explicit background, and 1–254 are stable user class IDs.
Only foreground classes with prompts enter the model. The shared model's episode cache is locked
across image changes and inference; prepared images remain immutable between interactions.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from fastapi import HTTPException

logger = logging.getLogger(__name__)


@dataclass
class PreparedImage:
    """
    Immutable model input and its mapping to the original scan's voxel grid.
    """

    tensor: torch.Tensor
    affine: np.ndarray
    native_affine: np.ndarray
    native_shape: Tuple[int, int, int]


@dataclass
class PredictionContext:
    """
    Keep raw model-grid logits on CPU for one session's next correction.

    The key identifies the scan, checkpoint, class order, and editing context. Native prompt
    codes distinguish removed or relabeled marks even when they map to the same model voxel.
    """

    key: Tuple[Any, ...]
    prompt_codes: np.ndarray
    logits: torch.Tensor


def prepare_image(image: nib.Nifti1Image, modality: str) -> PreparedImage:
    """
    Resample a 3D image to the training grid, preserving the exact voxel-to-world mapping.

    CT uses [-1024, 1024] clipping; other modalities use the 0th and 99.8th percentiles.
    Normalize after interpolation and before center padding, as in the training data preparation.
    """
    import voxel

    if len(image.shape) != 3:
        raise HTTPException(400, "This model requires a 3D scan")
    canonical = nib.as_closest_canonical(image)
    array = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise HTTPException(400, "The scan contains non-finite intensities")
    low, high = (-1024, 1024) if modality == "ct" else np.percentile(array, [0, 99.8])
    array = np.clip(array, low, high)
    volume = voxel.Volume(torch.from_numpy(array), torch.as_tensor(canonical.affine))
    extent = volume.geometry.spacing.double() * torch.tensor(volume.baseshape)
    spacing = float(extent.max()) / 128
    volume = volume.resample(spacing, mode="linear", antialias=False, padding_mode="border")
    tensor = volume.tensor
    span = tensor.max() - tensor.min()
    tensor = (tensor - tensor.min()) / span.clamp_min(1e-8)
    # Match MONAI ResizeWithPadOrCrop: odd padding puts the extra voxel at the high end.
    shape = np.asarray(tensor.shape[-3:])
    crop = np.maximum((shape - 128) // 2, 0)
    tensor = tensor[:, crop[0]:crop[0] + 128, crop[1]:crop[1] + 128, crop[2]:crop[2] + 128]
    difference = np.maximum(128 - np.asarray(tensor.shape[-3:]), 0)
    before = difference // 2
    after = difference - before
    padding = tuple(int(value) for pair in zip(before[::-1], after[::-1]) for value in pair)
    tensor = functional.pad(tensor, padding)
    affine = volume.geometry.tensor.numpy().copy()
    affine[:3, 3] += affine[:3, :3] @ (crop - before)
    return PreparedImage(
        tensor.unsqueeze(0).float(), affine,
        image.affine.copy(), tuple(image.shape),
    )


def map_prompts(
    annotation: nib.Nifti1Image, prepared: PreparedImage,
    allowed_label_ids: Optional[List[int]] = None,
) -> Tuple[Dict[int, torch.Tensor], List[int]]:
    """
    Map every marked native voxel to a model voxel and densely remap prompted class IDs.

    Coordinates are transformed directly instead of resampling a sparse bitmap, so isolated
    clicks survive downsampling. Duplicates within each class are removed; no prompts are sampled.
    """
    values = annotation.get_fdata()
    if values.shape != prepared.native_shape or not np.allclose(
        annotation.affine, prepared.native_affine, atol=1e-4,
    ):
        raise HTTPException(400, "Annotations must match the scan's shape and affine")
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise HTTPException(400, "Annotation values must be integers from 0 through 255")
    if np.any((values < 0) | (values > 255)):
        raise HTTPException(400, "Annotation values must be integers from 0 through 255")
    foreground = [int(label) for label in np.unique(values) if 0 < label < 255]
    if allowed_label_ids is not None:
        foreground = [label for label in foreground if label in allowed_label_ids]
    labels = [0, *foreground]
    transform = np.linalg.inv(prepared.affine) @ annotation.affine
    shape = tuple(prepared.tensor.shape[-3:])
    indices = {}
    for dense_id, user_id in enumerate([255, *foreground]):
        coordinates = np.argwhere(values == user_id)
        mapped = nib.affines.apply_affine(transform, coordinates)
        mapped = np.rint(mapped).astype(np.int64)
        mapped = np.clip(mapped, 0, np.asarray(shape) - 1)
        flat = np.ravel_multi_index(mapped.T, shape) if mapped.size else np.empty(0, np.int64)
        indices[dense_id] = torch.from_numpy(np.unique(flat)).long()
    return indices, labels


@torch.inference_mode()
def restore_prediction(
    logits: torch.Tensor, prepared: PreparedImage, labels: List[int],
) -> nib.Nifti1Image:
    """
    Interpolate logits to the original scan before argmax, in bounded native-space slabs.

    The output preserves original scan orientation, shape, affine, and user class IDs. Trilinear
    interpolation with border padding and align_corners=False matches the native-grid evaluator.
    """
    device = logits.device
    transform = np.linalg.inv(prepared.affine) @ prepared.native_affine
    matrix = torch.as_tensor(transform, dtype=torch.float64, device=device)
    shape = prepared.native_shape
    cube_shape = torch.tensor(logits.shape[-3:], dtype=torch.float64, device=device)
    y = torch.arange(shape[1], dtype=torch.float64, device=device)
    z = torch.arange(shape[2], dtype=torch.float64, device=device)
    result = np.zeros(shape, np.uint8)
    lookup = np.asarray(labels, dtype=np.uint8)
    # Bound temporary sampled logits to roughly 32 MiB even for many classes or large scans.
    slab = max(1, 8_000_000 // (len(labels) * shape[1] * shape[2]))
    for start in range(0, shape[0], slab):
        stop = min(start + slab, shape[0])
        x = torch.arange(start, stop, dtype=torch.float64, device=device)
        coordinates = (
            x[:, None, None, None] * matrix[:3, 0]
            + y[None, :, None, None] * matrix[:3, 1]
            + z[None, None, :, None] * matrix[:3, 2] + matrix[:3, 3]
        )
        grid = ((2 * coordinates + 1) / cube_shape - 1).flip(-1).float().unsqueeze(0)
        sampled = functional.grid_sample(
            logits.float(), grid, mode="bilinear", padding_mode="border", align_corners=False,
        )
        result[start:stop] = lookup[sampled.argmax(1).squeeze(0).cpu().numpy()]
    return nib.Nifti1Image(result, prepared.native_affine)


class CoolBlobRunner:
    """
    Serialize access to one checkpoint and its active image-feature cache.
    """

    def __init__(self) -> None:
        """
        Defer checkpoint loading until the first real prediction.
        """
        self.lock = Lock()
        self.model = None
        self.checkpoint = None
        self.prepared = None

    def load_model(self, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
        """
        Strictly load the complete trained model, including the fine-tuned Anatomix backbone.
        """
        from cool_blob.model import AmixCrossAttentionClickCachedFixedTotalMultiLabel

        checkpoint = Path(config["checkpoint_path"])
        if self.model is not None and self.checkpoint == checkpoint:
            return self.model
        if self.model is not None:
            self.model.end_cached_episode()
        with Path(config["training_config_path"]).open() as stream:
            options = dict(yaml.safe_load(stream)["model"])
        options.pop("_class", None)
        options.pop("compile", None)
        options.update(anatomix_compile=False, anatomix_random_init=True)
        model = AmixCrossAttentionClickCachedFixedTotalMultiLabel(**options)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        cleaned = {}
        for key, value in state.items():
            if key.startswith("anatomix._orig_mod."):
                key = "anatomix." + key.removeprefix("anatomix._orig_mod.")
            cleaned[key] = value
        model.load_state_dict(cleaned, strict=True)
        self.model = model.to(device).eval()
        self.checkpoint = checkpoint
        self.prepared = None
        return self.model

    @torch.inference_mode()
    def predict(
        self, volume_path: Path, annotation_path: Path, cache_entry: Any, config: Dict[str, Any],
        modality: str, context_id: Optional[str] = None,
        active_label_id: Optional[int] = None,
        allowed_label_ids: Optional[List[int]] = None,
    ) -> nib.Nifti1Image:
        """
        Predict in native space, using the last compatible prediction for iterative models.

        A new context ID, image, checkpoint, class order, or removed mark starts without a prior.
        Prior probabilities are never reconstructed from the displayed hard segmentation.
        """
        with self.lock:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = self.load_model(config, device)
            key = (str(volume_path), volume_path.stat().st_mtime_ns, modality)
            if getattr(cache_entry, "fiery_image_key", None) != key:
                prepared = prepare_image(nib.load(volume_path), modality)
                prepared.tensor = prepared.tensor.to(device)
                cache_entry.fiery_image = prepared
                cache_entry.fiery_image_key = key
            prepared = cache_entry.fiery_image
            if self.prepared is not prepared:
                model.end_cached_episode()
                model.begin_cached_episode()
                self.prepared = prepared
            annotation = nib.load(annotation_path)
            if active_label_id is not None:
                values = np.asarray(annotation.dataobj)
                if not np.isin(values, [0, active_label_id, 255]).all():
                    raise HTTPException(400, "Other labels cannot enter binary inference")
                binary = np.where(values == active_label_id, 1, values).astype(np.uint8)
                annotation = nib.Nifti1Image(binary, annotation.affine)
            indices, labels = map_prompts(annotation, prepared, allowed_label_ids)
            if active_label_id is not None and len(labels) != 2:
                raise HTTPException(400, "Add foreground marks for this label first")
            if len(labels) == 1:
                cache_entry.prediction_context = None
                empty = np.zeros(prepared.native_shape, np.uint8)
                return nib.Nifti1Image(empty, prepared.native_affine)
            indices = {label: values.to(device) for label, values in indices.items()}
            checkpoint_time = self.checkpoint.stat().st_mtime_ns if self.checkpoint.exists() else None
            context_key = (str(self.checkpoint), checkpoint_time, key, tuple(labels), context_id)
            prediction = self.predict_from_prompts(
                model, prepared, annotation, indices, labels, cache_entry, context_key,
            )
            if active_label_id is not None:
                values = (np.asarray(prediction.dataobj) != 0).astype(np.uint8) * active_label_id
                prediction = nib.Nifti1Image(values, prediction.affine)
            return prediction

    @torch.inference_mode()
    def predict_from_prompts(
        self, model: torch.nn.Module, prepared: PreparedImage, annotation: nib.Nifti1Image,
        indices: Dict[int, torch.Tensor], labels: List[int], cache_entry: Any,
        context_key: Tuple[Any, ...],
    ) -> nib.Nifti1Image:
        """
        Sample prior evidence through the model's own sampler and cache only successful results.

        Prior state is private to a session and stays on CPU between requests. Removing any
        native mark discards the prior so erased or undone evidence cannot influence corrections.
        """
        device = prepared.tensor.device
        previous = getattr(cache_entry, "prediction_context", None)
        values = np.asarray(annotation.dataobj).reshape(-1)
        marked = np.flatnonzero(values)
        prompt_codes = marked * 256 + values[marked].astype(np.int64)
        previous_logits = None
        if model.supports_prev_prediction and previous is not None:
            has_same_context = previous.key == context_key
            has_all_marks = np.isin(previous.prompt_codes, prompt_codes).all()
            if has_same_context and has_all_marks:
                previous_logits = previous.logits.to(device)
        # Dense raw logits let the trained sampler choose tokens with the checkpoint's settings.
        logits = model(
            prepared.tensor, support_images=None, support_labels=None,
            labels=torch.arange(len(labels), device=device), click_indices=[indices],
            prev_logits=previous_logits,
        )
        prediction = restore_prediction(logits, prepared, labels)
        cache_entry.prediction_context = None
        if model.supports_prev_prediction:
            cache_entry.prediction_context = PredictionContext(
                context_key, prompt_codes, logits.detach().cpu(),
            )
            logger.info(
                "Prediction context: reused=%s classes=%s", previous_logits is not None, labels,
            )
        return prediction


runner = CoolBlobRunner()
