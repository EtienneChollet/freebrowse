"""Resume the saved FreeBrowse debug-eval script for EPC-2141 skeleton models."""

import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom

MODEL_NAMES = ["skel-dilate", "skel-only", "skel-baseline"]
DEBUG_ROOT = Path("debug_output")
RESAMPLE_ISOTROPIC = True
MAX_SLICES = 6
MIN_THICKNESS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AXIS_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Axial"}


def pad_to_multiple(tensor, multiple=32):
    """Pad spatial dims to nearest multiple."""
    d, h, w = tensor.shape[-3:]
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if pd == 0 and ph == 0 and pw == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pw, 0, ph, 0, pd))


def load_model(model_name):
    """Load a FreeBrowse model folder."""
    model_dir = Path("models") / model_name
    module_path = model_dir / "model.py"
    spec = importlib.util.spec_from_file_location(f"{model_name}_model", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{model_name}_model"] = module
    spec.loader.exec_module(module)

    model = module.SegModel()
    checkpoint = torch.load(model_dir / "weights.pt", map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


def run_inference(model, volume, previous_prediction, positive, negative):
    """Build 5-channel input, pad, run model, unpad, and return a binary mask."""
    shape = volume.shape
    channels = [volume, previous_prediction, positive, negative, np.zeros_like(volume)]
    tensors = [pad_to_multiple(torch.from_numpy(channel).float()) for channel in channels]
    input_tensor = torch.stack(tensors).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor.to(DEVICE)).squeeze().cpu()

    d, h, w = shape
    logits = logits[:d, :h, :w]
    return (logits.sigmoid() > 0.5).to(torch.uint8).numpy()


def select_slices(positive, negative):
    """Rank (axis, slice) pairs by in-plane scribble density."""
    combined = positive + negative
    binary = combined > 0
    other_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    candidates = []
    for axis in range(3):
        densities = combined.sum(axis=other_axes[axis])
        nonzero_idxs = np.nonzero(densities)[0]
        for slice_idx in nonzero_idxs:
            sl = np.take(binary, slice_idx, axis=axis)
            rows = sl.any(axis=1).sum()
            cols = sl.any(axis=0).sum()
            if min(rows, cols) >= MIN_THICKNESS:
                candidates.append((axis, int(slice_idx), float(densities[slice_idx])))

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected = [(axis, slice_idx) for axis, slice_idx, _ in candidates[:MAX_SLICES]]
    selected.sort(key=lambda item: (item[0], item[1]))

    if selected:
        return selected

    fallback = []
    for axis in range(3):
        densities = combined.sum(axis=other_axes[axis])
        for slice_idx in np.nonzero(densities)[0]:
            fallback.append((axis, int(slice_idx), float(densities[slice_idx])))
    fallback.sort(key=lambda item: item[2], reverse=True)
    selected = [(axis, slice_idx) for axis, slice_idx, _ in fallback[:MAX_SLICES]]
    selected.sort(key=lambda item: (item[0], item[1]))
    print("  WARNING: no in-plane scribbles, using fallback")
    return selected


def resample_isotropic(arrays, voxel_spacing, orders):
    """Resample arrays to isotropic spacing. Returns None if already isotropic."""
    min_spacing = voxel_spacing.min()
    zoom_factors = tuple(voxel_spacing / min_spacing)
    if np.allclose(zoom_factors, 1.0):
        return None
    return [
        scipy_zoom(array, zoom_factors, order=order).astype(np.float32)
        for array, order in zip(arrays, orders)
    ]


def plot_prediction(volume, positive, negative, prediction, slices, title, save_path):
    """Two-row figure: scribble overlay on top and prediction on bottom."""
    nb_cols = len(slices)
    fig, axes = plt.subplots(2, nb_cols, figsize=(4 * nb_cols, 8))
    if nb_cols == 1:
        axes = axes[:, np.newaxis]

    for col, (axis, slice_idx) in enumerate(slices):
        volume_slice = np.take(volume, slice_idx, axis=axis).T
        positive_slice = np.take(positive, slice_idx, axis=axis).T
        negative_slice = np.take(negative, slice_idx, axis=axis).T
        prediction_slice = np.take(prediction, slice_idx, axis=axis).T

        ax = axes[0, col]
        ax.imshow(volume_slice, cmap="gray", origin="lower")
        overlay = np.zeros((*volume_slice.shape, 4), dtype=np.float32)
        overlay[positive_slice > 0] = [0, 1, 0, 1]
        overlay[negative_slice > 0] = [1, 0, 0, 1]
        ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(f"{AXIS_NAMES[axis]} #{slice_idx}", fontsize=10)
        ax.axis("off")

        ax = axes[1, col]
        ax.imshow(volume_slice, cmap="gray", origin="lower")
        prediction_overlay = np.zeros((*volume_slice.shape, 4), dtype=np.float32)
        prediction_overlay[prediction_slice > 0] = [0, 0.5, 1, 0.5]
        ax.imshow(prediction_overlay, origin="lower", interpolation="nearest")
        ax.set_title(f"{AXIS_NAMES[axis]} #{slice_idx}", fontsize=10)
        ax.axis("off")

    axes[0, 0].set_ylabel("Input + Scribbles", fontsize=12)
    axes[1, 0].set_ylabel("Prediction", fontsize=12)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def iter_prompt_dirs():
    """Yield prompt iteration folders recursively under the debug root."""
    if not DEBUG_ROOT.exists():
        return
    yield from sorted(
        path for path in DEBUG_ROOT.rglob("prompt_iter_*")
        if path.is_dir()
    )


def load_prompt_iteration(prompt_dir):
    """Load saved FreeBrowse debug tensors for one prompt iteration."""
    metadata = json.loads((prompt_dir / "metadata.json").read_text())
    volume_img = nib.load(str(prompt_dir / "0_volume.nii.gz"))
    volume = volume_img.get_fdata().astype(np.float32)
    voxel_spacing = np.array(volume_img.header.get_zooms()[:3])
    previous_prediction = nib.load(str(prompt_dir / "1_previous_pred.nii.gz")).get_fdata().astype(np.float32)
    positive = nib.load(str(prompt_dir / "2_positive.nii.gz")).get_fdata().astype(np.float32)
    negative = nib.load(str(prompt_dir / "3_negative.nii.gz")).get_fdata().astype(np.float32)
    return metadata, volume, voxel_spacing, previous_prediction, positive, negative


def needs_isotropic_output(prompt_dir, model_name):
    """Return whether this prompt still needs the isotropic overlay."""
    volume_img = nib.load(str(prompt_dir / "0_volume.nii.gz"))
    voxel_spacing = np.array(volume_img.header.get_zooms()[:3])
    is_isotropic = np.allclose(voxel_spacing / voxel_spacing.min(), 1.0)
    output_path = prompt_dir / f"prediction_overlay_isotropic_{model_name}.png"
    return not is_isotropic and not output_path.exists()


def needs_any_output(prompt_dir, model_name):
    """Return whether this prompt still needs any output for a model."""
    output_path = prompt_dir / f"prediction_overlay_{model_name}.png"
    return not output_path.exists() or needs_isotropic_output(prompt_dir, model_name)


def render_outputs(model, model_name, prompt_dir):
    """Render missing standard and isotropic outputs for one prompt iteration."""
    standard_path = prompt_dir / f"prediction_overlay_{model_name}.png"
    metadata, volume, voxel_spacing, previous_prediction, positive, negative = load_prompt_iteration(prompt_dir)
    prediction = run_inference(model, volume, previous_prediction, positive, negative)
    slices = select_slices(positive, negative)
    title = (
        f"Model: {model_name}  |  {metadata['volume_name']}\n"
        f"{prompt_dir.name}  |  "
        f"+{int(positive.sum())} / -{int(negative.sum())} scribble voxels"
    )

    if not standard_path.exists():
        plot_prediction(volume, positive, negative, prediction, slices, title, standard_path)

    if not RESAMPLE_ISOTROPIC:
        return

    iso_path = prompt_dir / f"prediction_overlay_isotropic_{model_name}.png"
    if iso_path.exists():
        return

    resampled = resample_isotropic(
        [volume, positive, negative, prediction.astype(np.float32)],
        voxel_spacing,
        orders=[1, 0, 0, 0],
    )
    if resampled is None:
        return

    iso_volume, iso_positive, iso_negative, iso_prediction = resampled
    zoom_factors = voxel_spacing / voxel_spacing.min()
    iso_slices = [
        (axis, min(round(slice_idx * zoom_factors[axis]), iso_volume.shape[axis] - 1))
        for axis, slice_idx in slices
    ]
    plot_prediction(
        iso_volume,
        iso_positive,
        iso_negative,
        iso_prediction.astype(np.uint8),
        iso_slices,
        title + "  (isotropic)",
        iso_path,
    )


def run_model(model_name, prompt_dirs):
    """Run one model over saved debug prompt iterations, skipping existing outputs."""
    missing = [prompt_dir for prompt_dir in prompt_dirs if needs_any_output(prompt_dir, model_name)]
    print(f"{model_name}: {len(missing)} prompt iterations need output")
    if not missing:
        return 0

    model = load_model(model_name)
    for idx, prompt_dir in enumerate(missing, start=1):
        print(f"{model_name}: {idx}/{len(missing)} {prompt_dir}")
        render_outputs(model, model_name, prompt_dir)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return len(missing)


def main():
    """Run all configured skeleton models."""
    prompt_dirs = list(iter_prompt_dirs())
    print(f"Found {len(prompt_dirs)} prompt iterations in {DEBUG_ROOT}")
    print(f"Device: {DEVICE}")
    counts = {model_name: run_model(model_name, prompt_dirs) for model_name in MODEL_NAMES}
    print("Done.")
    print(counts)


if __name__ == "__main__":
    main()
