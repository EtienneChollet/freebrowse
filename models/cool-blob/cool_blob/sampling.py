"""
Anatomix padding and class sampling, preserved for checkpoint-compatible inference.
"""
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

def pad_to_multiple(x: torch.Tensor, multiple: int):
    """Right-pad the trailing three dims of `x` with zeros so each is
    divisible by `multiple`.

    Returns `(x_padded, (d_pad, h_pad, w_pad))`.
    """
    D, H, W = x.shape[-3:]
    d_pad = (multiple - D % multiple) % multiple
    h_pad = (multiple - H % multiple) % multiple
    w_pad = (multiple - W % multiple) % multiple
    # F.pad order: (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi)
    x_padded = F.pad(x, (0, w_pad, 0, h_pad, 0, d_pad))
    return x_padded, (d_pad, h_pad, w_pad)



def crop_from_pad(x: torch.Tensor, original_shape):
    """Undo `pad_to_multiple` by slicing the trailing three dims."""
    D, H, W = original_shape
    return x[..., :D, :H, :W]



def _pick_with_replacement(
    pool: torch.Tensor, n: int, rng: Optional[torch.Generator]
) -> torch.Tensor:
    """Draw `n` entries from 1-D `pool` uniformly with replacement."""
    kwargs = {"generator": rng} if rng is not None else {}
    pick = torch.randint(0, pool.numel(), (n,), device=pool.device, **kwargs)
    return pool[pick]



def _border_pad_shape(kernel_shape):
    """Asymmetric 3-D padding for same-size output (handles even kernels)."""
    kd, kh, kw = kernel_shape
    return (
        kw // 2, kw - 1 - kw // 2,
        kh // 2, kh - 1 - kh // 2,
        kd // 2, kd - 1 - kd // 2,
    )



def _border_kernel_shape(kernel_size, spacing):
    """Convert an isotropic box width to per-axis voxel kernel sizes."""
    assert kernel_size >= 1, f"kernel_size must be >= 1, got {kernel_size}"
    if spacing is None:
        return (kernel_size,) * 3
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.tolist()
    assert len(spacing) == 3, f"spacing must have 3 values, got {spacing}"
    return (
        max(2, math.ceil(kernel_size / spacing[0])),
        max(2, math.ceil(kernel_size / spacing[1])),
        max(2, math.ceil(kernel_size / spacing[2])),
    )



def _resolve_border_kernel_size(
    border_kernel_size: Union[int, Sequence[int]],
    rng: Optional[torch.Generator],
) -> int:
    """Resolve an int or `(min, max)` range to a single box width.

    A range is sampled uniformly per call (like RandBorderClick), giving a
    random-width band each time prompts are drawn.
    """
    if isinstance(border_kernel_size, int):
        assert border_kernel_size >= 1, f"invalid border_kernel_size {border_kernel_size}"
        return border_kernel_size
    lo, hi = int(border_kernel_size[0]), int(border_kernel_size[1])
    assert 1 <= lo <= hi, f"invalid border_kernel_size {border_kernel_size}"
    if lo == hi:
        return lo
    kwargs = {"generator": rng} if rng is not None else {}
    return int(torch.randint(lo, hi + 1, (1,), **kwargs).item())



def class_border_band(
    seg: torch.Tensor,
    cls: int,
    kernel_size: int,
    spacing: Optional[Union[torch.Tensor, Sequence[float]]] = None,
) -> torch.Tensor:
    """Boolean mask of `cls` voxels whose box neighborhood touches another class.

    Box-blur the complement of the `cls` region with a (replicate-padded,
    optionally spacing-aware) `kernel_size` box kernel and keep `cls` voxels
    where the blur is > 0 — i.e. at least one voxel in the box window belongs to
    a different class. `seg` must be a 3-D spatial volume; returns a mask of the
    same shape.
    """
    class_mask = seg == cls
    kernel_shape = _border_kernel_shape(kernel_size, spacing)
    comp = (~class_mask).to(torch.float32)[None, None]  # (1, 1, D, H, W)
    comp = F.pad(comp, _border_pad_shape(kernel_shape), mode="replicate")
    blurred = F.avg_pool3d(comp, kernel_shape, stride=1)  # (1, 1, D, H, W)
    return class_mask & (blurred[0, 0] > 0)



def sample_prompts_in_class(
    seg: torch.Tensor,
    cls: int,
    k: int,
    rng: Optional[torch.Generator] = None,
    border_fraction: float = 0.0,
    border_kernel_size: Union[int, Sequence[int]] = (3, 7),
    border_spacing: Optional[Union[torch.Tensor, Sequence[float]]] = None,
) -> torch.Tensor:
    """Draw `k` flat indices into DHW from the voxels of `cls`.

    Samples with replacement. By default (``border_fraction == 0``) samples
    uniformly over all `cls` voxels — training uses `torch.multinomial` on the
    full flat mask, but that path caps at 2**24 categories; for big volumes we
    instead locate the class voxels and uniformly pick among them (equivalent
    for binary masks).

    If ``border_fraction > 0``, roughly that fraction of the `k` samples is
    drawn from the *border band* of the `cls` region and the rest from the
    interior, biasing prompts toward boundaries. The band is computed with a box
    kernel exactly like `RandBorderClick` (see `class_border_band`):
    ``border_kernel_size`` is the box width — an int for a fixed band, or a
    ``(min, max)`` range sampled per call for a random-width band (default
    ``(3, 7)``). ``border_spacing`` optionally makes the box anisotropic in
    physical units. The border path requires `seg` to be a 3-D spatial volume.

    Empty-band fallbacks preserve the contract that exactly `k` indices are
    returned whenever the class is non-empty: if the `cls` region has no border
    band or no interior (e.g. a thin region that is entirely border), all `k`
    samples come from whichever band is non-empty.

    Raises if the class is empty in `seg`.
    """
    flat = seg.reshape(-1)
    class_idx = torch.nonzero(flat == cls, as_tuple=False).flatten()
    if class_idx.numel() == 0:
        raise ValueError(f"Class {cls} has no voxels in seg.")

    if border_fraction <= 0.0:
        return _pick_with_replacement(class_idx, k, rng)

    if seg.ndim < 3:
        raise ValueError(
            "border_fraction > 0 requires a 3-D `seg` volume to compute the "
            f"border band; got seg.ndim={seg.ndim}."
        )

    kernel_size = _resolve_border_kernel_size(border_kernel_size, rng)
    band = class_border_band(seg, cls, kernel_size, spacing=border_spacing).reshape(-1)
    border_idx = torch.nonzero(band, as_tuple=False).flatten()
    interior_idx = torch.nonzero((flat == cls) & ~band, as_tuple=False).flatten()

    # Fallbacks when a band is empty — always return exactly k indices.
    if border_idx.numel() == 0:
        return _pick_with_replacement(class_idx, k, rng)
    if interior_idx.numel() == 0:
        return _pick_with_replacement(border_idx, k, rng)

    k_border = max(0, min(k, int(round(k * border_fraction))))
    k_interior = k - k_border

    parts = []
    if k_border > 0:
        parts.append(_pick_with_replacement(border_idx, k_border, rng))
    if k_interior > 0:
        parts.append(_pick_with_replacement(interior_idx, k_interior, rng))
    return torch.cat(parts, dim=0)
