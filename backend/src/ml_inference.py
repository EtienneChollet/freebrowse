"""AI inference engine for the /ai/session/{id}/infer/{ml_id} endpoint.

Adapted from freebrowse-eti/backend/src/ml_inference.py:
- Sessions are on-disk (SessionManager-owned); inference reads paths from the
  session manifest rather than accepting base64-encoded bodies.
- Annotations arrive as a uint8 NIfTI label mask (1 = positive, 2 = negative)
  rather than flat click-index lists.
- Model cache is guarded by a threading.Lock since FastAPI runs sync endpoints
  on a threadpool.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from fastapi import HTTPException
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as label_components

import utils

logger = logging.getLogger(__name__)

SP3D_BACKEND = "sp3d"
NNINTERACTIVE_BACKEND = "nninteractive"
DEFAULT_MAX_INTERACTION_POINTS = 512

if not torch.cuda.is_available():
    logger.warning(
        "CUDA not available — ML inference will run on CPU "
        "(install CUDA-enabled pytorch for GPU acceleration)"
    )


_model_cache: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
_model_cache_lock = threading.Lock()


@dataclass
class InferenceArtifacts:
    """Populated lazily by run_inference and stashed on the session cache entry."""
    volume_tensor: torch.Tensor
    affine_ras: np.ndarray
    ras_dims: tuple[int, int, int]
    shape_before_pad: tuple[int, int, int]


@dataclass
class RawImageArtifacts:
    """Raw RAS image artifacts used by nnInteractive."""
    image: np.ndarray
    affine_ras: np.ndarray
    ras_dims: tuple[int, int, int]
    volume_path: Path


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Pad 3d tensor so each dimension is divisible by `multiple`."""
    d, h, w = tensor.shape
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if pd == 0 and ph == 0 and pw == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pw, 0, ph, 0, pd))


def smooth_click_components(
    mask: torch.Tensor,
    sigma: float | None,
    truncate: float = 3.0,
    normalize: str | None = None,
) -> torch.Tensor:
    """Smooth isolated click components while preserving scribble components.

    Parameters
    ----------
    mask : torch.Tensor
        Binary prompt mask with shape `(*spatial)`.
    sigma : float or None
        Gaussian smoothing sigma. `None` leaves the mask unchanged.
    truncate : float
        Number of standard deviations used to truncate the Gaussian kernel.
    normalize : str or None
        Kernel normalization from the training config. `None` restores peak scaling used by
        `scipy.ndimage.gaussian_filter`; other values keep SciPy's default sum-preserving output.

    Returns
    -------
    torch.Tensor
        Prompt mask with single-voxel click components smoothed and larger components unchanged.
    """
    if sigma is None:
        return mask

    mask_np = mask.detach().cpu().numpy().astype(np.float32)
    labeled, num_components = label_components(mask_np > 0)
    if num_components == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    click_mask = np.zeros_like(mask_np, dtype=np.float32)
    scribble_mask = np.zeros_like(mask_np, dtype=np.float32)
    for component_id in range(1, num_components + 1):
        component_mask = labeled == component_id
        if component_sizes[component_id] == 1:
            click_mask[component_mask] = 1.0
        else:
            scribble_mask[component_mask] = 1.0

    if not np.any(click_mask):
        return torch.from_numpy(scribble_mask).to(dtype=mask.dtype)

    blurred = gaussian_filter(click_mask, sigma=sigma, truncate=truncate)
    if normalize is None:
        radius = int(truncate * sigma + 0.5)
        axis = np.arange(-radius, radius + 1)
        kernel_1d = np.exp(-0.5 * (axis / sigma) ** 2)
        blurred *= kernel_1d.sum() ** mask_np.ndim

    smoothed = click_mask + blurred + scribble_mask
    return torch.from_numpy(smoothed.astype(np.float32)).to(dtype=mask.dtype)


def load_model_config(config_path: Path) -> dict[str, Any]:
    """Load a model YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_model_class(module_path: Path):
    """Dynamic import of SegModel from the model.py next to weights.pt."""
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module
    assert spec.loader is not None, f"No module loader for {module_path}"
    spec.loader.exec_module(module)

    model_class = getattr(module, "SegModel", None)
    if model_class is None:
        raise ValueError(f"No SegModel class found in {module_path}")
    return model_class


def load_model(
    module_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate, load state_dict, move to device, eval()."""
    model = load_model_class(module_path=module_path)()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def get_model(
    ml_id: str,
    models_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Lazy cached SP3D model loader. Thread-safe."""
    with _model_cache_lock:
        cached = _model_cache.get(ml_id)
        if cached is not None:
            return cached

        model_dir = models_dir / ml_id
        module_file = model_dir / "model.py"
        checkpoint_file = model_dir / "weights.pt"
        config_file = model_dir / "config.yml"

        if not module_file.exists() or not checkpoint_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{ml_id}' is missing model.py or weights.pt",
            )

        model = load_model(module_file, checkpoint_file, device)
        config = load_model_config(config_file) if config_file.exists() else {}

        _model_cache[ml_id] = (model, config)
        logger.info(f"Cached model '{ml_id}' on {device}")
        return model, config


def create_mask_nifti(
    mask: np.ndarray,
    affine: np.ndarray,
    label_value: int = 1,
) -> nib.Nifti1Image:
    """Build a uint8 NIfTI where foreground voxels carry `label_value`."""
    arr = np.where(mask > 0, np.uint8(label_value), np.uint8(0))
    nii = nib.Nifti1Image(arr, affine)
    nii.header.set_data_dtype(np.uint8)
    nii.header["scl_slope"] = 1.0
    nii.header["scl_inter"] = 0.0
    nii.header["cal_min"] = 0.0
    nii.header["cal_max"] = float(label_value)
    return nii


def _model_config(model_info: dict[str, Any] | None) -> dict[str, Any]:
    if not model_info:
        return {}
    config = model_info.get("config")
    return config if isinstance(config, dict) else {}


def _model_backend(model_info: dict[str, Any] | None) -> str:
    config = _model_config(model_info)
    return str(config.get("backend", SP3D_BACKEND)).lower()


def _resolve_volume_path(
    manifest: dict[str, Any],
    session_dir: Path,
    data_dir: Path,
) -> Path:
    """Resolve manifest.volume_path against DATA_DIR or session dir, guard traversal."""
    rel = manifest.get("volume_path")
    if not rel:
        raise HTTPException(status_code=400, detail="Session has no volume_path")
    root = data_dir if manifest.get("volume_path_root") == "data" else session_dir
    full = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        full.relative_to(root_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid volume_path: {rel}")
    if not full.exists():
        raise HTTPException(status_code=404, detail=f"Volume not found: {rel}")
    return full


def _resolve_annotation_path(session_dir: Path, manifest: dict[str, Any]) -> Path:
    """Resolve manifest.annotation_path under the session directory."""
    rel = manifest.get("annotation_path")
    if not rel:
        raise HTTPException(status_code=400, detail="Session has no annotation_path")
    full = (session_dir / rel).resolve()
    session_resolved = session_dir.resolve()
    try:
        full.relative_to(session_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid annotation_path: {rel}")
    if not full.exists():
        raise HTTPException(status_code=404, detail=f"Annotations not found: {rel}")
    return full


def prepare_session_tensors(
    manifest: dict[str, Any],
    session_dir: Path,
    data_dir: Path,
) -> InferenceArtifacts:
    """Load the session volume and run SP3D preprocessing.

    Mirrors eti's _load_and_store_volume_from_path: reorient to RAS, clip
    [0.5, 99.5] percentile, normalize to [0, 1], pad to multiple of 32.
    """
    volume_path = _resolve_volume_path(manifest, session_dir, data_dir)

    img = nib.load(str(volume_path))
    img_ras = nib.as_closest_canonical(img)
    volume_ras = img_ras.get_fdata().astype(np.float32)
    affine_ras = img_ras.affine
    ras_dims = volume_ras.shape

    tensor = torch.from_numpy(volume_ras).float()
    tensor = utils.clip_volume(tensor, "percentile", [0.5, 99.5])
    tensor = utils.relative_norm(tensor)
    shape_before_pad = tuple(tensor.shape)
    tensor = pad_to_multiple(tensor=tensor, multiple=32)

    return InferenceArtifacts(
        volume_tensor=tensor,
        affine_ras=affine_ras,
        ras_dims=ras_dims,
        shape_before_pad=shape_before_pad,
    )


def load_raw_ras_image(
    manifest: dict[str, Any],
    session_dir: Path,
    data_dir: Path,
) -> RawImageArtifacts:
    """Load the session volume in RAS orientation without intensity normalization.

    nnInteractive performs its own cropping and z-score normalization internally,
    so this function intentionally avoids the SP3D percentile clipping and
    min-max normalization pipeline.
    """
    volume_path = _resolve_volume_path(manifest, session_dir, data_dir)
    img = nib.load(str(volume_path))
    img_ras = nib.as_closest_canonical(img)
    image = np.asanyarray(img_ras.dataobj)

    return RawImageArtifacts(
        image=np.ascontiguousarray(image),
        affine_ras=img_ras.affine,
        ras_dims=tuple(image.shape),
        volume_path=volume_path,
    )


def annotation_mask_to_pos_neg(
    session_dir: Path,
    manifest: dict[str, Any],
    ras_dims: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load annotations NIfTI, reorient to RAS, split by value (1=pos, 2=neg)."""
    full = _resolve_annotation_path(session_dir, manifest)

    img = nib.load(str(full))
    img_ras = nib.as_closest_canonical(img)
    arr = np.asarray(img_ras.get_fdata())
    if arr.shape != ras_dims:
        raise HTTPException(
            status_code=400,
            detail=f"Annotation shape {arr.shape} != volume RAS dims {ras_dims}",
        )
    arr_int = arr.astype(np.int32)
    pos = torch.from_numpy((arr_int == 1).astype(np.float32))
    neg = torch.from_numpy((arr_int == 2).astype(np.float32))
    return pos, neg


def annotation_mask_to_points(
    session_dir: Path,
    manifest: dict[str, Any],
    ras_dims: tuple[int, int, int],
    max_interaction_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sparse annotation labels into nnInteractive point coordinates."""
    full = _resolve_annotation_path(session_dir, manifest)
    img = nib.load(str(full))
    img_ras = nib.as_closest_canonical(img)
    arr = np.asarray(img_ras.get_fdata())
    if arr.shape != ras_dims:
        raise HTTPException(
            status_code=400,
            detail=f"Annotation shape {arr.shape} != volume RAS dims {ras_dims}",
        )

    arr_int = arr.astype(np.int32)
    positive_points = np.argwhere(arr_int == 1)
    negative_points = np.argwhere(arr_int == 2)
    total_points = positive_points.shape[0] + negative_points.shape[0]
    if positive_points.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail="Annotation bitmap contains no positive points",
        )
    if total_points > max_interaction_points:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Annotation bitmap contains {total_points} interaction voxels, "
                f"which exceeds the nnInteractive limit of {max_interaction_points}. "
                "Use sparse pen points instead of filled or magic-wand regions."
            ),
        )
    return positive_points, negative_points


def _nninteractive_model_path(config: dict[str, Any]) -> Path:
    model_path = config.get("model_path")
    if not model_path:
        raise HTTPException(
            status_code=500,
            detail="nnInteractive model config is missing model_path",
        )
    resolved = Path(str(model_path)).expanduser().resolve()
    if not resolved.exists():
        raise HTTPException(
            status_code=500,
            detail=f"nnInteractive model_path does not exist: {resolved}",
        )
    return resolved


def _max_interaction_points(config: dict[str, Any]) -> int:
    value = config.get("max_interaction_points", DEFAULT_MAX_INTERACTION_POINTS)
    try:
        max_points = int(value)
    except (TypeError, ValueError):
        max_points = DEFAULT_MAX_INTERACTION_POINTS
    return max(1, max_points)


def _load_nninteractive_session_class():
    """Import nnInteractive lazily so SP3D inference works without it installed."""
    try:
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "nnInteractive is not installed in the backend environment. "
                "Install the local nnInteractive checkout into backend/.venv."
            ),
        ) from exc
    return nnInteractiveInferenceSession


def _get_or_create_nninteractive_session(
    cache_entry: Any,
    ml_id: str,
    config: dict[str, Any],
    artifacts: RawImageArtifacts,
):
    model_path = _nninteractive_model_path(config)
    cached_session = getattr(cache_entry, "nninteractive_session", None)
    cached_ml_id = getattr(cache_entry, "nninteractive_ml_id", None)
    cached_volume_path = getattr(cache_entry, "nninteractive_volume_path", None)

    if (
        cached_session is not None
        and cached_ml_id == ml_id
        and cached_volume_path == str(artifacts.volume_path)
    ):
        cached_session.reset_interactions()
        return cached_session

    session_class = _load_nninteractive_session_class()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    session = session_class(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count() or 1,
        do_autozoom=True,
        use_pinned_memory=device.type == "cuda",
    )
    session.initialize_from_trained_model_folder(str(model_path))
    session.set_image(artifacts.image[None])
    session.set_target_buffer(torch.zeros(artifacts.ras_dims, dtype=torch.uint8))

    cache_entry.nninteractive_session = session
    cache_entry.nninteractive_ml_id = ml_id
    cache_entry.nninteractive_volume_path = str(artifacts.volume_path)
    cache_entry.nninteractive_affine_ras = artifacts.affine_ras
    cache_entry.nninteractive_ras_dims = artifacts.ras_dims
    return session


def run_nninteractive_inference(
    *,
    ml_id: str,
    label_value: int,
    cache_entry: Any,
    session_dir: Path,
    manifest: dict[str, Any],
    data_dir: Path,
    model_info: dict[str, Any],
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Run nnInteractive from annotation bitmap-derived point interactions."""
    config = _model_config(model_info)
    artifacts = load_raw_ras_image(manifest, session_dir, data_dir)
    max_points = _max_interaction_points(config)
    positive_points, negative_points = annotation_mask_to_points(
        session_dir=session_dir,
        manifest=manifest,
        ras_dims=artifacts.ras_dims,
        max_interaction_points=max_points,
    )
    session = _get_or_create_nninteractive_session(
        cache_entry=cache_entry,
        ml_id=ml_id,
        config=config,
        artifacts=artifacts,
    )

    all_points = []
    for point in negative_points:
        all_points.append((tuple(int(v) for v in point), False))
    for point in positive_points:
        all_points.append((tuple(int(v) for v in point), True))

    last_index = len(all_points) - 1
    for index, (point, include_interaction) in enumerate(all_points):
        # nnInteractive centers autozoom on the interaction that triggers prediction.
        # Register negatives first so the single prediction is centered on a positive click.
        session.add_point_interaction(
            point,
            include_interaction=include_interaction,
            run_prediction=index == last_index,
        )

    target_buffer = session.target_buffer
    if isinstance(target_buffer, torch.Tensor):
        mask_np = target_buffer.detach().cpu().numpy()
    else:
        mask_np = np.asarray(target_buffer)
    nii = create_mask_nifti(mask_np, artifacts.affine_ras, label_value=label_value)
    return nii, artifacts.affine_ras


def run_sp3d_inference(
    *,
    session_id: str,
    ml_id: str,
    label_value: int,
    cache_entry: Any,
    session_dir: Path,
    manifest: dict[str, Any],
    data_dir: Path,
    models_dir: Path,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Run the existing SP3D-style SegModel inference path."""
    if cache_entry.volume_tensor is None:
        artifacts = prepare_session_tensors(manifest, session_dir, data_dir)
        cache_entry.volume_tensor = artifacts.volume_tensor
        cache_entry.affine_ras = artifacts.affine_ras
        cache_entry.ras_dims = artifacts.ras_dims
        cache_entry.shape_before_pad = artifacts.shape_before_pad

    volume_tensor: torch.Tensor = cache_entry.volume_tensor
    affine_ras: np.ndarray = cache_entry.affine_ras
    ras_dims = cache_entry.ras_dims
    shape_before_pad = cache_entry.shape_before_pad

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = get_model(ml_id=ml_id, models_dir=models_dir, device=device)
    prompts_config = config.get("prompts", {}) if isinstance(config, dict) else {}

    click_sigma = prompts_config.get("click_smoothing_sigma")
    click_truncate = prompts_config.get("click_smoothing_truncate", 3.0)
    click_normalize = prompts_config.get("normalize_smooth_kernel")

    pos_mask, neg_mask = annotation_mask_to_pos_neg(session_dir, manifest, ras_dims)
    pos_mask = smooth_click_components(pos_mask, click_sigma, click_truncate, click_normalize)
    neg_mask = smooth_click_components(neg_mask, click_sigma, click_truncate, click_normalize)
    pos_mask = pad_to_multiple(pos_mask, multiple=32)
    neg_mask = pad_to_multiple(neg_mask, multiple=32)

    input_tensor = torch.zeros((1, 5, *volume_tensor.shape), dtype=torch.float32)
    input_tensor[0, 0] = volume_tensor
    input_tensor[0, 2] = pos_mask
    input_tensor[0, 3] = neg_mask

    prev_logits = cache_entry.previous_logits
    if prev_logits is not None:
        include_pred = prompts_config.get("include_previous_prediction", False)
        include_logits = prompts_config.get("include_previous_logits", False)
        if include_logits:
            input_tensor[0, 1] = pad_to_multiple(prev_logits, multiple=32)
        elif include_pred:
            input_tensor[0, 1] = pad_to_multiple(
                (torch.sigmoid(prev_logits) > 0.5).float(), multiple=32,
            )

    with torch.no_grad():
        output = model(input_tensor.to(device))
        logits = output.squeeze().cpu()

    d, h, w = shape_before_pad
    logits = logits[:d, :h, :w]
    cache_entry.previous_logits = logits

    mask_np = (logits.sigmoid() > 0.5).to(torch.uint8).numpy()
    nii = create_mask_nifti(mask_np, affine_ras, label_value=label_value)
    logger.debug("SP3D inference completed for session %s", session_id)
    return nii, affine_ras


def run_inference(
    *,
    session_id: str,
    ml_id: str,
    label_value: int,
    cache_entry: Any,
    session_dir: Path,
    manifest: dict[str, Any],
    data_dir: Path,
    models_dir: Path,
    model_info: dict[str, Any] | None = None,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Run the configured model on the session volume and annotation mask.

    SP3D models use the existing `SegModel` path. nnInteractive models use the
    official `nnInteractiveInferenceSession` and receive raw image values plus
    bitmap-derived point interactions.
    """
    backend = _model_backend(model_info)
    if backend == NNINTERACTIVE_BACKEND:
        assert model_info is not None, "nnInteractive dispatch requires model_info"
        return run_nninteractive_inference(
            ml_id=ml_id,
            label_value=label_value,
            cache_entry=cache_entry,
            session_dir=session_dir,
            manifest=manifest,
            data_dir=data_dir,
            model_info=model_info,
        )
    if backend != SP3D_BACKEND:
        raise HTTPException(status_code=400, detail=f"Unsupported model backend: {backend}")
    return run_sp3d_inference(
        session_id=session_id,
        ml_id=ml_id,
        label_value=label_value,
        cache_entry=cache_entry,
        session_dir=session_dir,
        manifest=manifest,
        data_dir=data_dir,
        models_dir=models_dir,
    )
