"""
Verify that iterative segmentation receives only compatible, session-local raw predictions.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ai_session
import fiery_flame


class RecordingModel:
    """
    Record prior inputs at the model boundary while returning distinct raw class logits.
    """

    supports_prev_prediction = True

    def __init__(self) -> None:
        """
        Start with no recorded model calls.
        """
        self.priors: List[Optional[torch.Tensor]] = []

    def __call__(
        self, image: torch.Tensor, support_images: Any, support_labels: Any,
        labels: torch.Tensor, click_indices: Any, prev_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Preserve the received prior and return raw logits on the image grid.
        """
        self.priors.append(prev_logits)
        shape = (1, len(labels), *image.shape[-3:])
        return torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape) / 10


@pytest.fixture
def prediction() -> Any:
    """
    Exercise the production prior-selection and native-output path with a recording model.
    """
    runner = fiery_flame.FieryFlameRunner()
    model = RecordingModel()
    prepared = fiery_flame.PreparedImage(
        torch.zeros(1, 1, 2, 2, 2), np.eye(4), np.eye(4), (2, 2, 2),
    )
    cache = SimpleNamespace()

    def predict(
        marks: Dict[int, int], key: Tuple[Any, ...] = ("scan", "checkpoint", (0, 1), "edit"),
        session: Any = cache,
    ) -> nib.Nifti1Image:
        """
        Apply flat native marks and run a prediction in the supplied session context.
        """
        bitmap = np.zeros((2, 2, 2), np.uint8)
        for index, label in marks.items():
            bitmap.flat[index] = label
        annotation = nib.Nifti1Image(bitmap, np.eye(4))
        indices, labels = fiery_flame.map_prompts(annotation, prepared)
        return runner.predict_from_prompts(
            model, prepared, annotation, indices, labels, session, key,
        )

    return predict, model, cache


def test_next_edit_receives_exact_raw_logits(prediction: Any) -> None:
    """
    Appending a mark feeds back the previous raw prediction, not its hard output or softmax.
    """
    predict, model, cache = prediction
    predict({0: 1})
    previous = cache.prediction_context.logits.clone()
    result = predict({0: 1, 1: 1})
    assert model.priors[0] is None
    assert torch.equal(model.priors[1], previous)
    assert cache.prediction_context.logits.device.type == "cpu"
    assert result.shape == (2, 2, 2)


@pytest.mark.parametrize("changed_part", range(4))
def test_scan_checkpoint_class_or_edit_change_discards_prior(
    prediction: Any, changed_part: int,
) -> None:
    """
    Context changes cannot reuse logits with a different scan, model, class order, or edit ID.
    """
    predict, model, _ = prediction
    predict({0: 1})
    key = ["scan", "checkpoint", (0, 1), "edit"]
    key[changed_part] = "changed"
    predict({0: 1, 1: 1}, tuple(key))
    assert model.priors[-1] is None


@pytest.mark.parametrize("marks", [{1: 1}, {0: 255, 1: 1}])
def test_removed_or_relabeled_marks_discard_prior(prediction: Any, marks: Dict[int, int]) -> None:
    """
    Undoing or relabeling evidence cannot leave its influence in the next prediction.
    """
    predict, model, _ = prediction
    predict({0: 1, 1: 1})
    predict(marks)
    assert model.priors[-1] is None


def test_sessions_do_not_share_predictions(prediction: Any) -> None:
    """
    Two sessions editing identical marks on the same scan retain independent priors.
    """
    predict, model, _ = prediction
    predict({0: 1})
    predict({0: 1}, session=SimpleNamespace())
    assert model.priors[-1] is None
    predict({0: 1, 1: 1})
    assert model.priors[-1] is not None


def test_noniterative_model_does_not_reuse_or_store_prior(prediction: Any) -> None:
    """
    FIERY-FLAME retains its annotation-only inference behavior.
    """
    predict, model, cache = prediction
    predict({0: 1})
    model.supports_prev_prediction = False
    predict({0: 1, 1: 1})
    assert model.priors[-1] is None
    assert cache.prediction_context is None


def test_annotation_upload_preserves_iterative_context(tmp_path: Path) -> None:
    """
    The upload route's invalidation keeps iterative context available for the next correction.
    """
    manager = ai_session.SessionManager(tmp_path / "sessions", tmp_path, 60)
    session = manager.create("iterative")
    session_id = session["session_id"]
    directory, _ = manager.find_by_id(session_id)
    (directory / "annotations.nii.gz").touch()
    cache = manager.touch(session_id)
    context = object()
    cache.prediction_context = context
    manager.set_annots(session_id, "annotations.nii.gz")
    assert manager.touch(session_id).prediction_context is context
    manager.invalidate_all(session_id)
    assert manager.touch(session_id).prediction_context is None


def test_inference_route_passes_context_id_to_runner(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Browser context IDs reach the iterative runner through the real session and dispatch routes.
    """
    model_directory = tmp_path / "models" / "segmento-v1"
    model_directory.mkdir(parents=True)
    (model_directory / "config.yml").write_text("backend: fiery_flame\n")
    image = nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.eye(4))
    nib.save(image, tmp_path / "scan.nii.gz")
    app = FastAPI()
    app.include_router(ai_session.build_router(
        ai_dir=tmp_path / "sessions", data_dir=tmp_path, models_dir=model_directory.parent,
        ttl_seconds=60, enabled=True,
    ))
    client = TestClient(app)
    session = client.post("/ai/session/new", json={"session_name": "browser"}).json()
    url = f"/ai/session/{session['session_id']}"
    client.post(f"{url}/set_volume", json={"volume_path": "scan.nii.gz"})
    nib.save(image, tmp_path / "sessions" / "browser" / "annotations.nii.gz")
    client.post(f"{url}/set_annots", json={"annotation_path": "annotations.nii.gz"})
    received = []

    def predict(*args: Any, context_id: Optional[str] = None) -> nib.Nifti1Image:
        """
        Observe the runner boundary without loading a checkpoint in an HTTP unit test.
        """
        received.append(context_id)
        return image

    monkeypatch.setattr(fiery_flame.runner, "predict", predict)
    response = client.post(f"{url}/infer/segmento-v1", json={
        "annotation_mode": "multiclass", "prediction_context_id": "browser-edit-1",
    })
    assert response.status_code == 200, response.text
    assert received == ["browser-edit-1"]
