# %% [markdown]
# # Brainstem Bbox Crop Experiment (EPC-1780)
#
# Test whether cropping the input volume to a bounding box around
# positive scribble annotations improves brainstem segmentation.

# %%
import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# %%
DEBUG_DIR = Path('debug_output/20260331_181259_263605')
MODEL_DIR = Path('models/sp3d-scribbles-halfres')
PADDING = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(DEBUG_DIR / 'metadata.json') as f:
    metadata = json.load(f)
print(json.dumps(metadata, indent=2))

# %% [markdown]
# ## Load model

# %%
sys.path.insert(0, str(MODEL_DIR))
from model import SegModel

model = SegModel(
    ndim=3, in_channels=5, out_channels=1,
    nb_features=(4, 16, 64, 256, 256),
    order='ca', final_activation=None,
)
checkpoint = torch.load(
    MODEL_DIR / 'weights.pt', map_location=DEVICE, weights_only=False,
)
state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()
print(f'Model loaded on {DEVICE}')

# %% [markdown]
# ## Load debug tensors

# %%
volume = nib.load(str(DEBUG_DIR / '0_volume.nii.gz')).get_fdata().astype(np.float32)
prev_pred = nib.load(str(DEBUG_DIR / '1_previous_pred.nii.gz')).get_fdata().astype(np.float32)
positive = nib.load(str(DEBUG_DIR / '2_positive.nii.gz')).get_fdata().astype(np.float32)
negative = nib.load(str(DEBUG_DIR / '3_negative.nii.gz')).get_fdata().astype(np.float32)

print(f'Volume: {volume.shape}')
print(f'Positive scribbles: {int(positive.sum())} voxels')
print(f'Negative scribbles: {int(negative.sum())} voxels')

# %% [markdown]
# ## Compute bounding box around positive scribbles

# %%
coords = np.argwhere(positive > 0)
bbox_min = coords.min(axis=0)
bbox_max = coords.max(axis=0)
print(f'Positive bbox: min={bbox_min}, max={bbox_max}')
print(f'Positive bbox span: {bbox_max - bbox_min + 1}')

# Apply padding and clamp to volume bounds
crop_min = np.maximum(bbox_min - PADDING, 0)
crop_max = np.minimum(bbox_max + PADDING + 1, volume.shape)
crop_shape = crop_max - crop_min
print(f'Crop region: min={crop_min}, max={crop_max}, shape={crop_shape}')

# %% [markdown]
# ## Helper: pad to multiple of 16 and run inference

# %%
def pad_to_multiple(tensor, multiple=32):
    d, h, w = tensor.shape[-3:]
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if pd == 0 and ph == 0 and pw == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pw, 0, ph, 0, pd))


def run_inference(vol, pos, neg, prev):
    """Build 5-channel input, pad, run model, unpad, return logits + mask."""
    shape_before_pad = vol.shape
    vol_t = pad_to_multiple(torch.from_numpy(vol).float())
    pos_t = pad_to_multiple(torch.from_numpy(pos).float())
    neg_t = pad_to_multiple(torch.from_numpy(neg).float())
    prev_t = pad_to_multiple(torch.from_numpy(prev).float())
    unused_t = torch.zeros_like(vol_t)

    input_tensor = torch.stack(
        [vol_t, prev_t, pos_t, neg_t, unused_t]
    ).unsqueeze(0)
    print(f'  Input tensor shape: {list(input_tensor.shape)}')

    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
        logits = output.squeeze().cpu()

    d, h, w = shape_before_pad
    logits = logits[:d, :h, :w]
    mask = (logits.sigmoid() > 0.5).to(torch.uint8).numpy()
    return logits.numpy(), mask

# %% [markdown]
# ## Run inference: FULL volume (no crop)

# %%
print('Running FULL volume inference...')
logits_full, mask_full = run_inference(volume, positive, negative, prev_pred)
print(f'  Full prediction: {mask_full.sum()} voxels')

# %% [markdown]
# ## Run inference: CROPPED to positive scribble bbox

# %%
sl = tuple(slice(lo, hi) for lo, hi in zip(crop_min, crop_max))
vol_crop = volume[sl]
pos_crop = positive[sl]
neg_crop = negative[sl]
prev_crop = prev_pred[sl]

print(f'Cropped shapes: vol={vol_crop.shape}, pos={int(pos_crop.sum())} voxels, neg={int(neg_crop.sum())} voxels')
print('Running CROPPED inference...')
logits_crop, mask_crop = run_inference(vol_crop, pos_crop, neg_crop, prev_crop)
print(f'  Cropped prediction: {mask_crop.sum()} voxels')

# %% [markdown]
# ## Compare predictions

# %%
# Embed cropped prediction back into full volume for comparison
mask_crop_full = np.zeros_like(volume, dtype=np.uint8)
mask_crop_full[sl] = mask_crop

# Dice between the two predictions (treating full as reference)
intersection = (mask_full & mask_crop_full).sum()
dice = 2 * intersection / (mask_full.sum() + mask_crop_full.sum() + 1e-8)
print(f'\nDice between full vs cropped predictions: {dice:.4f}')
print(f'Full prediction voxels: {mask_full.sum()}')
print(f'Cropped prediction voxels: {mask_crop_full.sum()}')

# %% [markdown]
# ## Visual comparison
#
# Show axial slices through the brainstem region for both predictions.

# %%
AXIS_NAMES = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}
MIN_THICKNESS = 3
MAX_SLICES = 6

# Smart slice selection: rank by in-plane scribble density
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
slices_to_show = [(a, s) for a, s, _ in candidates[:MAX_SLICES]]
slices_to_show.sort(key=lambda x: (x[0], x[1]))

if not slices_to_show:
    print("WARNING: no in-plane scribbles found, falling back to center slices")
    center = (bbox_min + bbox_max) // 2
    slices_to_show = [(0, center[0]), (1, center[1]), (2, center[2])]

print(f"Selected {len(slices_to_show)} slices:")
for axis, s in slices_to_show:
    mask = np.take(combined, s, axis=axis)
    print(f"  {AXIS_NAMES[axis]} #{s}: density={mask.sum():.0f}")

n_rows = len(slices_to_show)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

for row, (axis, sl_idx) in enumerate(slices_to_show):
    vol_slice = np.take(volume, sl_idx, axis=axis).T
    pos_slice = np.take(positive, sl_idx, axis=axis).T
    neg_slice = np.take(negative, sl_idx, axis=axis).T
    full_slice = np.take(mask_full, sl_idx, axis=axis).T
    crop_slice = np.take(mask_crop_full, sl_idx, axis=axis).T

    # Volume + scribbles
    ax = axes[row, 0]
    ax.imshow(vol_slice, cmap='gray', origin='lower')
    overlay = np.zeros((*vol_slice.shape, 4), dtype=np.float32)
    overlay[pos_slice > 0] = [0, 1, 0, 1]
    overlay[neg_slice > 0] = [1, 0, 0, 1]
    ax.imshow(overlay, origin='lower', interpolation='nearest')
    ax.set_title(f'{AXIS_NAMES[axis]} #{sl_idx} — Input')
    ax.axis('off')

    # Full prediction
    ax = axes[row, 1]
    ax.imshow(vol_slice, cmap='gray', origin='lower')
    pred_overlay = np.zeros((*vol_slice.shape, 4), dtype=np.float32)
    pred_overlay[full_slice > 0] = [0, 0.5, 1, 0.5]
    ax.imshow(pred_overlay, origin='lower', interpolation='nearest')
    ax.set_title(f'{AXIS_NAMES[axis]} #{sl_idx} — Full')
    ax.axis('off')

    # Cropped prediction + bbox outline
    ax = axes[row, 2]
    ax.imshow(vol_slice, cmap='gray', origin='lower')
    pred_overlay = np.zeros((*vol_slice.shape, 4), dtype=np.float32)
    pred_overlay[crop_slice > 0] = [0, 0.5, 1, 0.5]
    ax.imshow(pred_overlay, origin='lower', interpolation='nearest')

    # Draw bounding box projected onto this slice plane
    other_axes = [a for a in range(3) if a != axis]
    a0, a1 = other_axes
    rect = plt.Rectangle(
        (crop_min[a0], crop_min[a1]),
        crop_max[a0] - crop_min[a0],
        crop_max[a1] - crop_min[a1],
        linewidth=1.5, edgecolor='hotpink', facecolor='none',
    )
    ax.add_patch(rect)

    ax.set_title(f'{AXIS_NAMES[axis]} #{sl_idx} — Cropped')
    ax.axis('off')

fig.suptitle(
    f'Brainstem Segmentation: Full Volume vs Bbox Crop (padding={PADDING})\n'
    f'Dice(full vs crop) = {dice:.4f}',
    fontsize=14,
)
plt.tight_layout()
out_path = DEBUG_DIR / 'bbox_crop_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved comparison to {out_path}')
plt.show()
