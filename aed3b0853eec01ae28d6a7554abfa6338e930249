# %% [markdown]
# # Debug Inference Viewer
#
# Load debug subfolders from `debug_output/` and overlay interaction
# channels on the volume. Uses smart slice selection: ranks all
# (axis, slice) pairs by in-plane scribble density so each scribble
# is shown in its drawn plane.
#
# When `RESAMPLE_ISOTROPIC` is True, saves both native-spacing and
# isotropic-resampled overlay figures for each subfolder.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as scipy_zoom

# %% [markdown]
# ## Configuration

# %%
RESAMPLE_ISOTROPIC = True
MIN_THICKNESS = 3
MAX_SLICES = 8

debug_root = Path('debug_output')
available = sorted(debug_root.iterdir()) if debug_root.exists() else []
for d in available:
    print(d.name)

# %% [markdown]
# ## Helpers

# %%
AXIS_NAMES = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}
POS_COLOR = np.array([0.0, 1.0, 0.0])
NEG_COLOR = np.array([1.0, 0.0, 0.0])


def select_slices(channels):
    """Rank (axis, slice) pairs by in-plane scribble density."""
    volume = channels['0_volume']
    positive = channels.get('2_positive', np.zeros_like(volume))
    negative = channels.get('3_negative', np.zeros_like(volume))
    combined = positive + negative

    candidates = []
    for axis in range(3):
        for s in range(combined.shape[axis]):
            mask = np.take(combined, s, axis=axis)
            if not mask.any():
                continue
            density = mask.sum()
            binary = mask > 0
            rows = binary.any(axis=1).sum()
            cols = binary.any(axis=0).sum()
            if min(rows, cols) >= MIN_THICKNESS:
                candidates.append((axis, s, density))

    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = [(a, s) for a, s, _ in candidates[:MAX_SLICES]]
    selected.sort(key=lambda x: (x[0], x[1]))

    # Fallback: top density slices regardless of thickness
    if not selected:
        fallback = []
        for axis in range(3):
            for s in range(combined.shape[axis]):
                mask = np.take(combined, s, axis=axis)
                d = mask.sum()
                if d > 0:
                    fallback.append((axis, s, d))
        fallback.sort(key=lambda x: x[2], reverse=True)
        selected = [(a, s) for a, s, _ in fallback[:MAX_SLICES]]
        selected.sort(key=lambda x: (x[0], x[1]))
        print("WARNING: no in-plane scribbles found, using fallback")

    return selected


def resample_isotropic(channels, voxel_spacing):
    """Resample all channels to isotropic spacing (min voxel dim)."""
    min_spacing = voxel_spacing.min()
    zoom_factors = tuple(voxel_spacing / min_spacing)
    if np.allclose(zoom_factors, 1.0):
        return None

    print(f"Resampling to isotropic ({min_spacing:.2f}mm), "
          f"zoom factors: {zoom_factors}")
    resampled = {}
    for name, data in channels.items():
        order = 1 if name == '0_volume' else 0
        resampled[name] = scipy_zoom(
            data, zoom_factors, order=order,
        ).astype(np.float32)
        print(f"  {name}: {data.shape} -> {resampled[name].shape}")
    return resampled


def plot_overlay(channels, selected, metadata, save_path):
    """Plot volume slices with scribble overlay and save to disk."""
    volume = channels['0_volume']
    positive = channels.get('2_positive', np.zeros_like(volume))
    negative = channels.get('3_negative', np.zeros_like(volume))

    n_cols = len(selected)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax_idx, (axis, sl) in enumerate(selected):
        ax = axes[ax_idx]
        vol_slice = np.take(volume, sl, axis=axis).T
        pos_slice = np.take(positive, sl, axis=axis).T
        neg_slice = np.take(negative, sl, axis=axis).T

        h, w = vol_slice.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)

        pos_mask = pos_slice > 0
        overlay[pos_mask, :3] = POS_COLOR
        overlay[pos_mask, 3] = 1.0

        neg_mask = neg_slice > 0
        overlay[neg_mask, :3] = NEG_COLOR
        overlay[neg_mask, 3] = 1.0

        ax.imshow(vol_slice, cmap='gray', origin='lower')
        ax.imshow(overlay, origin='lower', interpolation='nearest')

        n_pos = int(pos_slice.sum())
        n_neg = int(neg_slice.sum())
        ax.set_title(
            f'{AXIS_NAMES[axis]} #{sl}\n+{n_pos} / -{n_neg}', fontsize=10,
        )
        ax.axis('off')

    fig.suptitle(
        f"Model: {metadata['model_name']}  |  {metadata['timestamp']}\n"
        f"Green = positive, Red = negative  |  "
        f"{int(positive.sum())} pos, {int(negative.sum())} neg voxels total",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# %% [markdown]
# ## Process all debug subfolders

# %%
for session_dir in available:
    iter_dirs = sorted(session_dir.glob('prompt_iter_*'))
    if not iter_dirs:
        iter_dirs = [session_dir]

    for debug_dir in iter_dirs:
        label = f"{session_dir.name}/{debug_dir.name}" if debug_dir != session_dir else session_dir.name
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print('='*60)

        metadata_path = debug_dir / 'metadata.json'
        if not metadata_path.exists():
            print("No metadata.json found, skipping.")
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load channels and extract voxel spacing from the first image
        channels = {}
        voxel_spacing = None
        for nii_path in sorted(debug_dir.glob('*.nii.gz')):
            name = nii_path.name.replace('.nii.gz', '')
            img = nib.load(str(nii_path))
            channels[name] = img.get_fdata().astype(np.float32)
            if voxel_spacing is None:
                voxel_spacing = np.array(img.header.get_zooms()[:3])
            print(f"{name}: shape={channels[name].shape}, "
                  f"range=[{channels[name].min():.4f}, {channels[name].max():.4f}]")
        print(f"Voxel spacing: {voxel_spacing}")

        # Anisotropic (native) overlay
        selected = select_slices(channels)
        print(f"Selected {len(selected)} slices (native):")
        for axis, s in selected:
            print(f"  {AXIS_NAMES[axis]} #{s}")
        plot_overlay(
            channels, selected, metadata,
            debug_dir / 'overlay_anisotropic.png',
        )

        # Isotropic overlay (same anatomical slices, scaled indices)
        if RESAMPLE_ISOTROPIC:
            iso_channels = resample_isotropic(channels, voxel_spacing)
            if iso_channels is not None:
                zoom_factors = voxel_spacing / voxel_spacing.min()
                iso_selected = [
                    (axis, round(s * zoom_factors[axis]))
                    for axis, s in selected
                ]
                iso_selected = [
                    (a, min(s, iso_channels['0_volume'].shape[a] - 1))
                    for a, s in iso_selected
                ]
                print(f"Mapped {len(iso_selected)} slices (isotropic):")
                for axis, s in iso_selected:
                    print(f"  {AXIS_NAMES[axis]} #{s}")
                plot_overlay(
                    iso_channels, iso_selected, metadata,
                    debug_dir / 'overlay_isotropic.png',
                )
            else:
                print("Already isotropic, skipping resample.")

# %% [markdown]
# ## Previous prediction (latest subfolder only)

# %%
if available:
    latest = available[-1]
    latest_iters = sorted(latest.glob('prompt_iter_*'))
    latest_iter = latest_iters[-1] if latest_iters else latest
    prev_nii = latest_iter / '1_previous_pred.nii.gz'
    if prev_nii.exists():
        prev_pred = nib.load(str(prev_nii)).get_fdata().astype(np.float32)
        if prev_pred.max() > 0:
            mid_z = prev_pred.shape[2] // 2
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, offset in zip(axes, [-5, 0, 5]):
                sl = min(max(mid_z + offset, 0), prev_pred.shape[2] - 1)
                ax.imshow(prev_pred[:, :, sl].T, cmap='hot', origin='lower')
                ax.set_title(f'prev_pred z={sl}')
                ax.axis('off')
            plt.tight_layout()
            save_path = latest_iter / 'prev_pred.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            print("No previous prediction data (channel is all zeros).")
