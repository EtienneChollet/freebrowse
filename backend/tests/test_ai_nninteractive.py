from pathlib import Path
import sys
from typing import Any, Dict

import nibabel as nib
import numpy as np
import pytest
from fastapi import HTTPException

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ai_session
import ml_inference


class CacheEntry:
    volume_tensor = None
    affine_ras = None
    ras_dims = None
    shape_before_pad = None
    previous_logits = None
    nninteractive_session = None
    nninteractive_ml_id = None
    nninteractive_volume_path = None
    nninteractive_affine_ras = None
    nninteractive_ras_dims = None


class FakeNNInteractiveSession:
    instances = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model_path = None
        self.image = None
        self.target_buffer = None
        self.points = []
        FakeNNInteractiveSession.instances.append(self)

    def initialize_from_trained_model_folder(self, model_path: str) -> None:
        self.model_path = model_path

    def set_image(self, image: np.ndarray) -> None:
        self.image = image.copy()

    def set_target_buffer(self, target_buffer: Any) -> None:
        self.target_buffer = target_buffer

    def reset_interactions(self) -> None:
        self.points = []
        if self.target_buffer is not None:
            self.target_buffer.zero_()

    def add_point_interaction(
        self,
        point: tuple[int, int, int],
        include_interaction: bool,
        run_prediction: bool = True,
    ) -> None:
        self.points.append((point, include_interaction, run_prediction))
        if run_prediction:
            for recorded_point, is_positive, _runs_prediction in self.points:
                if is_positive:
                    self.target_buffer[recorded_point] = 1


def test_model_discovery_accepts_nninteractive_config_without_weights(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "nninteractive-v1"
    model_dir.mkdir()
    (model_dir / "config.yml").write_text(
        "backend: nninteractive\n"
        "name: nnInteractive v1.0\n"
        "model_path: /tmp/nnInteractive_v1.0\n",
    )

    models = ai_session._list_models(tmp_path)

    assert len(models) == 1
    assert models[0]["ml_id"] == "nninteractive-v1"
    assert models[0]["name"] == "nnInteractive v1.0"
    assert models[0]["backend"] == "nninteractive"
    assert models[0]["model_module_path"] is None
    assert models[0]["checkpoint_path"] is None


def test_run_inference_dispatches_nninteractive_without_segmodel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: Dict[str, Any] = {}

    def fake_nninteractive_inference(**kwargs: Any):
        called.update(kwargs)
        return nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)), np.eye(4)

    def fail_sp3d_loader(*args: Any, **kwargs: Any):
        raise AssertionError("SP3D loader should not run for nnInteractive models")

    monkeypatch.setattr(
        ml_inference,
        "run_nninteractive_inference",
        fake_nninteractive_inference,
    )
    monkeypatch.setattr(ml_inference, "get_model", fail_sp3d_loader)

    model_info = {
        "ml_id": "nninteractive-v1",
        "config": {"backend": "nninteractive", "model_path": "/tmp/nnInteractive_v1.0"},
    }
    nii, _affine = ml_inference.run_inference(
        session_id="session",
        ml_id="nninteractive-v1",
        label_value=7,
        cache_entry=CacheEntry(),
        session_dir=tmp_path,
        manifest={"volume_path": "volume.nii.gz", "annotation_path": "annotations.nii.gz"},
        data_dir=tmp_path,
        models_dir=tmp_path,
        model_info=model_info,
    )

    assert nii.get_fdata().shape == (2, 2, 2)
    assert called["ml_id"] == "nninteractive-v1"
    assert called["label_value"] == 7
    assert called["model_info"] is model_info


def test_run_nninteractive_inference_uses_raw_image_and_point_interactions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "nnInteractive_v1.0"
    model_path.mkdir()
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 100.0
    annotation = np.zeros((3, 3, 3), dtype=np.uint8)
    annotation[1, 1, 1] = 1
    annotation[2, 1, 1] = 2
    nib.save(nib.Nifti1Image(volume, np.eye(4)), tmp_path / "volume.nii.gz")
    nib.save(nib.Nifti1Image(annotation, np.eye(4)), tmp_path / "annotations.nii.gz")

    FakeNNInteractiveSession.instances.clear()
    monkeypatch.setattr(
        ml_inference,
        "_load_nninteractive_session_class",
        lambda: FakeNNInteractiveSession,
    )

    nii, _affine = ml_inference.run_inference(
        session_id="session",
        ml_id="nninteractive-v1",
        label_value=3,
        cache_entry=CacheEntry(),
        session_dir=tmp_path,
        manifest={
            "volume_path": "volume.nii.gz",
            "volume_path_root": "session",
            "annotation_path": "annotations.nii.gz",
        },
        data_dir=tmp_path,
        models_dir=tmp_path,
        model_info={
            "ml_id": "nninteractive-v1",
            "config": {"backend": "nninteractive", "model_path": str(model_path)},
        },
    )

    session = FakeNNInteractiveSession.instances[0]
    assert session.model_path == str(model_path)
    assert session.image.shape == (1, 3, 3, 3)
    assert float(session.image[0, 1, 1, 1]) == 100.0
    assert session.points == [
        ((2, 1, 1), False, False),
        ((1, 1, 1), True, True),
    ]
    result = np.asarray(nii.get_fdata(), dtype=np.uint8)
    assert result[1, 1, 1] == 3


def test_annotation_mask_to_points_rejects_dense_bitmap(tmp_path: Path) -> None:
    annotation = np.zeros((3, 3, 3), dtype=np.uint8)
    annotation[0, 0, 0] = 1
    annotation[0, 0, 1] = 1
    annotation[0, 0, 2] = 2
    nib.save(nib.Nifti1Image(annotation, np.eye(4)), tmp_path / "annotations.nii.gz")

    with pytest.raises(HTTPException) as exc_info:
        ml_inference.annotation_mask_to_points(
            session_dir=tmp_path,
            manifest={"annotation_path": "annotations.nii.gz"},
            ras_dims=(3, 3, 3),
            max_interaction_points=2,
        )

    assert exc_info.value.status_code == 400
    assert "exceeds the nnInteractive limit" in exc_info.value.detail


def test_annotation_mask_to_points_splits_positive_and_negative(tmp_path: Path) -> None:
    annotation = np.zeros((3, 3, 3), dtype=np.uint8)
    annotation[1, 0, 0] = 1
    annotation[2, 0, 0] = 2
    nib.save(nib.Nifti1Image(annotation, np.eye(4)), tmp_path / "annotations.nii.gz")

    positive, negative = ml_inference.annotation_mask_to_points(
        session_dir=tmp_path,
        manifest={"annotation_path": "annotations.nii.gz"},
        ras_dims=(3, 3, 3),
        max_interaction_points=2,
    )

    assert positive.tolist() == [[1, 0, 0]]
    assert negative.tolist() == [[2, 0, 0]]
