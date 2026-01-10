# %% [markdown]
# # Predict on Saved Debug Interactions
#
# Load saved FreeBrowse interaction data (volume + scribble masks),
# run inference with a specified model, and plot input overlays
# alongside model predictions.
#
# Each session in `debug_output/` has prompt iterations with 5-channel
# NIfTI files. The notebook processes first-iteration manual interactions
# and saves `prediction_overlay.png` outputs into `prompt_iter_000`.

# %%
import sys
import json
import importlib
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom as scipy_zoom, label as scipy_label
from scipy.ndimage import gaussian_filter

# %% [markdown]
# ## Configuration

# %%
MODEL_NAMES = [
    'neel_synth-only',
    'neel_synth-real',
]
DEBUG_ROOT = Path('debug_output')
PROMPT_ITER_NAME = 'prompt_iter_000'
RESAMPLE_ISOTROPIC = True
MAX_SLICES = 6
MIN_THICKNESS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name):
    """Load model, config, and gaussian click params by name."""
    model_dir = Path('models') / model_name

    module_path = model_dir / 'model.py'
    spec = importlib.util.spec_from_file_location(
        f'{module_path.stem}_{model_name}', module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    net = module.SegModel()
    checkpoint = torch.load(
        model_dir / 'weights.pt', map_location=DEVICE, weights_only=False,
    )
    state_dict = (
        checkpoint['model']
        if isinstance(checkpoint, dict) and 'model' in checkpoint
        else checkpoint
    )
    net.load_state_dict(state_dict)
    net = net.to(DEVICE).eval()
    print(f'Model loaded: {model_name} on {DEVICE}')

    config_path = model_dir / 'config.yml'
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    prompts_config = config.get('prompts', {})
    click_sigma = prompts_config.get('click_smoothing_sigma', None)
    click_truncate = prompts_config.get('click_smoothing_truncate', 3)
    click_normalize = prompts_config.get('normalize_smooth_kernel', None)
    if click_sigma is not None:
        print(f'  Gaussian clicks: sigma={click_sigma}, '
              f'truncate={click_truncate}, normalize={click_normalize}')

    return net, click_sigma, click_truncate, click_normalize

# %% [markdown]
# ## Helpers

# %%
AXIS_NAMES = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}
CMAP_POS = LinearSegmentedColormap.from_list(
    'green_alpha', [(0, 1, 0, 0), (0, 1, 0, 1)],
)
CMAP_NEG = LinearSegmentedColormap.from_list(
    'red_alpha', [(1, 0, 0, 0), (1, 0, 0, 1)],
)


def separate_and_smooth_clicks(mask, sigma, truncate=3, normalize=None):
    """Separate clicks from scribbles via CC analysis, smooth clicks only.

    Parameters
    ----------
    mask : np.ndarray
        Binary interaction mask (positive or negative channel).
    sigma : float
        Gaussian smoothing sigma for click blobs.
    truncate : float
        Number of standard deviations to truncate the kernel.
    normalize : {'sum'} or None
        Kernel normalization. None = peak of 1 (unnormalized).

    Returns
    -------
    np.ndarray
        Combined mask: gaussian-smoothed clicks + binary scribbles.
    """
    labeled, n_components = scipy_label(mask > 0)
    if n_components == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())

    click_mask = np.zeros_like(mask, dtype=np.float32)
    scribble_mask = np.zeros_like(mask, dtype=np.float32)
    for comp_id in range(1, n_components + 1):
        if component_sizes[comp_id] == 1:
            click_mask[labeled == comp_id] = 1.0
        else:
            scribble_mask[labeled == comp_id] = 1.0

    n_clicks = int(click_mask.sum())
    n_scribble_ccs = n_components - n_clicks
    print(f'      CC separation: {n_clicks} clicks, '
          f'{n_scribble_ccs} scribble components')

    if n_clicks == 0:
        return scribble_mask

    blurred = gaussian_filter(click_mask, sigma=sigma, truncate=truncate)
    if normalize is None:
        radius = int(truncate * sigma + 0.5)
        ax = np.arange(-radius, radius + 1)
        kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel_sum = kernel_1d.sum() ** 3
        blurred *= kernel_sum

    smoothed_clicks = click_mask + blurred
    return smoothed_clicks + scribble_mask


def pad_to_multiple(tensor, multiple=32):
    """Pad spatial dims to nearest multiple."""
    d, h, w = tensor.shape[-3:]
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if pd == 0 and ph == 0 and pw == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pw, 0, ph, 0, pd))


def run_inference(net, vol, prev, pos, neg):
    """Build 5-channel input, pad, run model, unpad, return binary mask."""
    shape = vol.shape
    channels = [vol, prev, pos, neg, np.zeros_like(vol)]
    tensors = [
        pad_to_multiple(torch.from_numpy(c).float()) for c in channels
    ]
    input_tensor = torch.stack(tensors).unsqueeze(0)

    with torch.no_grad():
        logits = net(input_tensor.to(DEVICE)).squeeze().cpu()

    d, h, w = shape
    logits = logits[:d, :h, :w]
    return (logits.sigmoid() > 0.5).to(torch.uint8).numpy()


def select_slices(positive, negative):
    """Rank (axis, slice) pairs by in-plane scribble density."""
    combined = positive + negative
    binary = combined > 0
    other_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    # Vectorized per-slice density for all axes
    candidates = []
    for axis in range(3):
        densities = combined.sum(axis=other_axes[axis])
        nonzero_idxs = np.nonzero(densities)[0]
        for s in nonzero_idxs:
            sl = np.take(binary, s, axis=axis)
            rows = sl.any(axis=1).sum()
            cols = sl.any(axis=0).sum()
            if min(rows, cols) >= MIN_THICKNESS:
                candidates.append((axis, int(s), float(densities[s])))

    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = [(a, s) for a, s, _ in candidates[:MAX_SLICES]]
    selected.sort(key=lambda x: (x[0], x[1]))

    if not selected:
        fallback = []
        for axis in range(3):
            densities = combined.sum(axis=other_axes[axis])
            for s in np.nonzero(densities)[0]:
                fallback.append((axis, int(s), float(densities[s])))
        fallback.sort(key=lambda x: x[2], reverse=True)
        selected = [(a, s) for a, s, _ in fallback[:MAX_SLICES]]
        selected.sort(key=lambda x: (x[0], x[1]))
        print('  WARNING: no in-plane scribbles, using fallback')

    return selected


def resample_isotropic(arrays, voxel_spacing, orders):
    """Resample arrays to isotropic spacing. Returns None if already isotropic."""
    min_spacing = voxel_spacing.min()
    zoom_factors = tuple(voxel_spacing / min_spacing)
    if np.allclose(zoom_factors, 1.0):
        return None
    return [
        scipy_zoom(arr, zoom_factors, order=order).astype(np.float32)
        for arr, order in zip(arrays, orders)
    ]


def plot_prediction(volume, positive, negative, prediction, slices, title,
                    save_path):
    """Two-row figure: scribble overlay (top), prediction (bottom)."""
    n_cols = len(slices)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, (axis, sl_idx) in enumerate(slices):
        vol_slice = np.take(volume, sl_idx, axis=axis).T
        pos_slice = np.take(positive, sl_idx, axis=axis).T
        neg_slice = np.take(negative, sl_idx, axis=axis).T
        pred_slice = np.take(prediction, sl_idx, axis=axis).T

        # Top row: volume + interaction overlay
        ax = axes[0, col]
        ax.imshow(vol_slice, cmap='gray', origin='lower')
        if pos_slice.max() > 0:
            ax.imshow(
                pos_slice, cmap=CMAP_POS, vmin=0, vmax=pos_slice.max(),
                origin='lower', interpolation='nearest',
            )
        if neg_slice.max() > 0:
            ax.imshow(
                neg_slice, cmap=CMAP_NEG, vmin=0, vmax=neg_slice.max(),
                origin='lower', interpolation='nearest',
            )
        ax.set_title(f'{AXIS_NAMES[axis]} #{sl_idx}', fontsize=10)
        ax.axis('off')

        # Bottom row: volume + prediction overlay
        ax = axes[1, col]
        ax.imshow(vol_slice, cmap='gray', origin='lower')
        pred_overlay = np.zeros((*vol_slice.shape, 4), dtype=np.float32)
        pred_overlay[pred_slice > 0] = [0, 0.5, 1, 0.5]
        ax.imshow(pred_overlay, origin='lower', interpolation='nearest')
        ax.set_title(f'{AXIS_NAMES[axis]} #{sl_idx}', fontsize=10)
        ax.axis('off')

    axes[0, 0].set_ylabel('Input + Interactions', fontsize=12)
    axes[1, 0].set_ylabel('Prediction', fontsize=12)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# %% [markdown]
# ## Process all debug sessions

# %%
sessions = sorted(DEBUG_ROOT.iterdir()) if DEBUG_ROOT.exists() else []
print(f'Found {len(sessions)} sessions in {DEBUG_ROOT}')

for model_name in MODEL_NAMES:
    print(f'\n{"#" * 60}')
    print(f'# Model: {model_name}')
    print('#' * 60)
    net, click_sigma, click_truncate, click_normalize = load_model(model_name)

    for session_dir in sessions:
        if not session_dir.is_dir():
            continue
        print(f'\n{"=" * 60}')
        print(f'Session: {session_dir.name}')
        print('=' * 60)

        first_iter_dir = session_dir / PROMPT_ITER_NAME
        if not first_iter_dir.is_dir():
            print(f'  No {PROMPT_ITER_NAME} found, skipping')
            continue

        for iter_dir in [first_iter_dir]:
            print(f'\n  {iter_dir.name}:')

            with open(iter_dir / 'metadata.json') as f:
                metadata = json.load(f)

            vol_img = nib.load(str(iter_dir / '0_volume.nii.gz'))
            volume = vol_img.get_fdata().astype(np.float32)
            voxel_spacing = np.array(vol_img.header.get_zooms()[:3])
            prev_pred = nib.load(
                str(iter_dir / '1_previous_pred.nii.gz'),
            ).get_fdata().astype(np.float32)
            positive = nib.load(
                str(iter_dir / '2_positive.nii.gz'),
            ).get_fdata().astype(np.float32)
            negative = nib.load(
                str(iter_dir / '3_negative.nii.gz'),
            ).get_fdata().astype(np.float32)

            print(f'    Volume: {volume.shape}, spacing: {voxel_spacing}')
            print(f'    +{int(positive.sum())} / -{int(negative.sum())} voxels')

            slices = select_slices(positive, negative)

            if click_sigma is not None:
                print('    Positive:')
                positive = separate_and_smooth_clicks(
                    positive, click_sigma, click_truncate, click_normalize,
                )
                print('    Negative:')
                negative = separate_and_smooth_clicks(
                    negative, click_sigma, click_truncate, click_normalize,
                )

            prediction = run_inference(net, volume, prev_pred, positive, negative)
            print(f'    Prediction: {int(prediction.sum())} voxels')
            print(f'    Selected {len(slices)} slices')

            title = (
                f'Model: {model_name}  |  {metadata["volume_name"]}\n'
                f'{iter_dir.name}  |  '
                f'+{int(positive.sum())} / -{int(negative.sum())} interaction voxels'
            )
            plot_prediction(
                volume, positive, negative, prediction, slices, title,
                iter_dir / f'prediction_overlay_{model_name}.png',
            )

            if RESAMPLE_ISOTROPIC:
                resampled = resample_isotropic(
                    [volume, positive, negative, prediction.astype(np.float32)],
                    voxel_spacing,
                    orders=[1, 0, 0, 0],
                )
                if resampled is not None:
                    iso_vol, iso_pos, iso_neg, iso_pred = resampled
                    zoom_factors = voxel_spacing / voxel_spacing.min()
                    iso_slices = [
                        (axis, min(round(s * zoom_factors[axis]),
                                   iso_vol.shape[axis] - 1))
                        for axis, s in slices
                    ]
                    print(f'    Isotropic: {iso_vol.shape}')
                    iso_title = title + '  (isotropic)'
                    plot_prediction(
                        iso_vol, iso_pos, iso_neg, iso_pred.astype(np.uint8),
                        iso_slices, iso_title,
                        iter_dir / f'prediction_overlay_{model_name}_isotropic.png',
                    )

print('\nDone.')
