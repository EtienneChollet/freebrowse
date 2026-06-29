from pathlib import Path
import sys

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from ml_inference import smooth_click_components


def test_smooth_click_components_smooths_isolated_click() -> None:
    """A single-voxel prompt is treated as a click and Gaussian-smoothed."""
    mask = torch.zeros((24, 24, 24), dtype=torch.float32)
    mask[18, 18, 18] = 1.0

    smoothed = smooth_click_components(mask, sigma=2, truncate=3.0, normalize=None)

    assert smoothed.shape == mask.shape
    assert smoothed[18, 18, 18] > 1.0
    assert smoothed[18, 18, 19] > 0.0


def test_smooth_click_components_preserves_scribble_components() -> None:
    """A multi-voxel connected prompt is treated as a scribble and left binary."""
    mask = torch.zeros((24, 24, 24), dtype=torch.float32)
    mask[1, 1, 1:4] = 1.0

    smoothed = smooth_click_components(mask, sigma=2, truncate=3.0, normalize=None)

    assert torch.equal(smoothed, mask)


def test_smooth_click_components_can_be_disabled() -> None:
    """A missing smoothing sigma returns the prompt mask unchanged."""
    mask = torch.zeros((8, 8, 8), dtype=torch.float32)
    mask[4, 4, 4] = 1.0

    smoothed = smooth_click_components(mask, sigma=None)

    assert smoothed is mask
