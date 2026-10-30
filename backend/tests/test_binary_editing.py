"""
Check binary model inputs, independent saved edits, accepted masks, and stale request rejection.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any, Dict

import nibabel as nib
import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ai_session
import fiery_flame


CLASSES = [
    {"id": 3, "name": "A", "color": "#00ff00", "locked": False},
    {"id": 8, "name": "B", "color": "#00ffff", "locked": False},
]


class RecordingModel:
    """
    Record production binary model calls and return a controllable overlapping foreground mask.
    """

    supports_prev_prediction = True

    def __init__(self) -> None:
        """
        Prepare model-call evidence and optional synchronization for in-flight requests.
        """
        self.calls = []
        self.foreground = [0, 1, 2]
        self.started = None
        self.release = None

    def begin_cached_episode(self) -> None:
        """
        Accept the runner's production image-cache lifecycle.
        """

    def end_cached_episode(self) -> None:
        """
        Accept the runner's production image-cache lifecycle.
        """

    def __call__(self, image: torch.Tensor, **options: Any) -> torch.Tensor:
        """
        Keep exact class, prompt, and prior inputs without loading an expensive checkpoint.
        """
        self.calls.append(options)
        if self.started is not None:
            self.started.set()
            assert self.release.wait(10)
        logits = torch.zeros((1, 2, 2, 2, 2))
        logits[:, 0] = 1
        logits[0, 1].view(-1)[self.foreground] = 3
        return logits


class EditingSession:
    """
    Drive the actual session routes and inspect only disposable synthetic session files.
    """

    def __init__(self, directory: Path, client: TestClient, state: Dict[str, Any]) -> None:
        """
        Retain one test session's authoritative HTTP state.
        """
        self.directory = directory
        self.client = client
        self.state = state
        self.url = f"/ai/session/{state['session_id']}"

    def settings(self, **changes: Any) -> Any:
        """
        Submit complete settings with the latest known revision.
        """
        settings = {
            "model_id": "segmento-v1", "annotation_mode": "multiclass", "classes": CLASSES,
            "editing_mode": "per_label", "active_label_id": 3, "modality": "ct",
        }
        settings.update({key: self.state[key] for key in settings if key in self.state})
        settings.update(state_version=self.state.get("state_version", 0), **changes)
        response = self.client.post(f"{self.url}/settings", json=settings)
        if response.status_code == 200:
            self.state = response.json()
        return response

    def drawing(self, label: int, marks: Dict[int, int]) -> Any:
        """
        Save native foreground/background marks without triggering prediction.
        """
        values = np.zeros((2, 2, 2), np.uint8)
        for index, value in marks.items():
            values.flat[index] = value
        data = base64.b64encode(nib.Nifti1Image(values, np.eye(4)).to_bytes()).decode()
        response = self.client.post(f"{self.url}/binary/annotations", json={
            "label_id": label, "state_version": self.state["state_version"], "data": data,
        })
        if response.status_code == 200:
            self.state = response.json()
        return response

    def predict(self) -> Any:
        """
        Predict the selected label against the saved prompts and latest session revision.
        """
        response = self.client.post(f"{self.url}/infer/{self.state['model_id']}", json={
            "editing_mode": "per_label", "active_label_id": self.state["active_label_id"],
            "state_version": self.state["state_version"],
        })
        if response.status_code == 200:
            self.state = response.json()
        return response

    def mask(self) -> np.ndarray:
        """
        Read the current combined result on its original native voxel grid.
        """
        return np.asarray(nib.load(self.directory / self.state["result_path"]).dataobj)


@pytest.fixture(params=["segmento-v1", "open-whale", "grey-plant"])
def editing(tmp_path: Path, monkeypatch: Any, request: Any) -> Any:
    """
    Exercise production routing, prompt remapping, priors, composition, and persistence together.
    """
    models = tmp_path / "models"
    model_id = request.param
    (models / model_id).mkdir(parents=True)
    (models / model_id / "config.yml").write_text("backend: fiery_flame\n")
    scan = nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), np.eye(4))
    nib.save(scan, tmp_path / "scan.nii.gz")
    app = FastAPI()
    app.include_router(ai_session.build_router(
        ai_dir=tmp_path / "sessions", data_dir=tmp_path, models_dir=models,
        ttl_seconds=60, enabled=True,
    ))
    client = TestClient(app)
    state = client.post("/ai/session/new", json={"session_name": "binary"}).json()
    directory = tmp_path / "sessions" / "binary"
    client.post(f"/ai/session/{state['session_id']}/set_volume", json={
        "volume_path": "scan.nii.gz",
    })
    model = RecordingModel()
    runner = fiery_flame.FieryFlameRunner()
    runner.checkpoint = tmp_path / "checkpoint.pt"
    runner.checkpoint.touch()
    monkeypatch.setattr(runner, "load_model", lambda config, device: model)
    monkeypatch.setattr(fiery_flame, "runner", runner)
    monkeypatch.setattr(fiery_flame, "prepare_image", lambda image, modality:
                        fiery_flame.PreparedImage(
                            torch.zeros(1, 1, 2, 2, 2), np.eye(4), np.eye(4), (2, 2, 2),
                        ))
    session = EditingSession(directory, client, state)
    assert session.settings(model_id=model_id).status_code == 200
    return session, model


def test_binary_prompts_priors_and_reopening_are_isolated(editing: Any) -> None:
    """
    Each structure has two model classes, private background marks, and its own saved raw prior.
    """
    session, model = editing
    assert session.drawing(3, {0: 3, 7: 255}).status_code == 200
    assert session.predict().status_code == 200
    assert session.drawing(3, {0: 3, 1: 3, 7: 255}).status_code == 200
    assert session.predict().status_code == 200
    assert session.settings(active_label_id=8).status_code == 200
    assert session.drawing(8, {2: 8, 6: 255}).status_code == 200
    assert session.predict().status_code == 200
    assert session.settings(active_label_id=3).status_code == 200
    assert session.predict().status_code == 200
    assert [call["labels"].tolist() for call in model.calls] == [[0, 1]] * 4
    assert [call["prev_logits"] is not None for call in model.calls] == [False, True, False, True]
    assert model.calls[2]["click_indices"][0][0].tolist() == [6]
    assert model.calls[2]["click_indices"][0][1].tolist() == [2]
    assert session.drawing(3, {0: 3, 1: 8}).status_code == 400
    for edit in session.state["label_edits"].values():
        assert (session.directory / edit["annotation_path"]).exists()
        assert (session.directory / edit["prior_path"]).exists()


def test_lock_freezes_visible_mask_and_rejects_edits_and_deletion(editing: Any) -> None:
    """
    Accepted masks keep both their foreground and negative space when another label overlaps.
    """
    session, model = editing
    session.drawing(3, {0: 3})
    session.predict()
    session.settings(active_label_id=8)
    session.drawing(8, {1: 8})
    model.foreground = [1, 2, 3, 4]
    session.predict()
    visible_b = session.mask() == 8
    locked = [{**entry, "locked": entry["id"] == 8} for entry in CLASSES]
    assert session.settings(classes=locked).status_code == 200
    assert np.array_equal(session.mask() == 8, visible_b)
    assert session.drawing(8, {1: 8}).status_code == 409
    assert session.client.post(f"{session.url}/binary/clear", json={
        "label_id": 8, "state_version": session.state["state_version"],
    }).status_code == 409
    assert session.settings(classes=[locked[0]], active_label_id=3).status_code == 409
    assert session.settings(editing_mode="joint").status_code == 409
    session.settings(active_label_id=3)
    model.foreground = list(range(8))
    session.predict()
    assert np.array_equal(session.mask() == 8, visible_b)
    visible_a = session.mask() == 3
    all_locked = [{**entry, "locked": True} for entry in CLASSES]
    assert session.settings(classes=all_locked).status_code == 200
    assert np.array_equal(session.mask() == 8, visible_b)
    assert np.array_equal(session.mask() == 3, visible_a)


def test_lock_invalidates_inflight_prediction(editing: Any) -> None:
    """
    A prediction started before accepting a mask cannot commit after the lock is saved.
    """
    session, model = editing
    session.drawing(3, {0: 3})
    session.predict()
    accepted = session.mask().copy()
    model.started, model.release = Event(), Event()
    model.foreground = list(range(8))
    with ThreadPoolExecutor(max_workers=1) as executor:
        prediction = executor.submit(session.predict)
        assert model.started.wait(10)
        locked = [{**entry, "locked": entry["id"] == 3} for entry in CLASSES]
        assert session.settings(classes=locked).status_code == 200
        model.release.set()
        assert prediction.result().status_code == 409
    assert np.array_equal(session.mask(), accepted)


def test_clear_preserves_other_label_and_saved_marks(editing: Any) -> None:
    """
    Saving and clearing in manual mode does not require an inference or erase another label.
    """
    session, model = editing
    session.drawing(3, {0: 3, 7: 255})
    first_path = session.state["label_edits"]["3"]["annotation_path"]
    session.settings(active_label_id=8)
    session.drawing(8, {1: 8, 6: 255})
    response = session.client.post(f"{session.url}/binary/clear", json={
        "label_id": 8, "state_version": session.state["state_version"],
    })
    assert response.status_code == 200
    state = response.json()
    assert state["label_edits"]["3"]["annotation_path"] == first_path
    assert state["label_edits"]["8"]["annotation_path"] is None
    assert model.calls == []
    stale = session.settings(active_label_id=3)
    assert stale.status_code == 409
    assert session.client.post(f"{session.url}/set_volume", json={
        "volume_path": "scan.nii.gz",
    }).status_code == 409


def test_joint_work_survives_opt_in_and_return_to_joint(editing: Any) -> None:
    """
    Switching workflows preserves both independent binary edits and the original joint drawing.
    """
    session, _ = editing
    assert session.settings(editing_mode="joint").status_code == 200
    values = np.zeros((2, 2, 2), np.uint8)
    values.flat[0], values.flat[1], values.flat[7] = 3, 8, 255
    nib.save(nib.Nifti1Image(values, np.eye(4)), session.directory / "annotations.nii.gz")
    assert session.client.post(f"{session.url}/set_annots", json={
        "annotation_path": "annotations.nii.gz",
    }).status_code == 200
    assert session.settings(editing_mode="per_label").status_code == 200
    session.drawing(3, {2: 3, 6: 255})
    assert session.settings(editing_mode="joint").status_code == 200
    assert session.state["annotation_path"] == "annotations.nii.gz"
    assert np.array_equal(nib.load(session.directory / "annotations.nii.gz").get_fdata(), values)
    assert session.settings(editing_mode="per_label").status_code == 200
    path = session.state["label_edits"]["3"]["annotation_path"]
    assert nib.load(session.directory / path).get_fdata().flat[2] == 3


def test_deleted_labels_cannot_reenter_joint_model_inputs(editing: Any) -> None:
    """
    Restored old joint files cannot supply prompts for classes deleted during binary editing.
    """
    session, model = editing
    session.settings(editing_mode="joint")
    values = np.zeros((2, 2, 2), np.uint8)
    values.flat[0], values.flat[1], values.flat[7] = 3, 8, 255
    nib.save(nib.Nifti1Image(values, np.eye(4)), session.directory / "annotations.nii.gz")
    session.client.post(f"{session.url}/set_annots", json={"annotation_path": "annotations.nii.gz"})
    session.settings(editing_mode="per_label")
    session.settings(classes=[CLASSES[1]], active_label_id=8)
    session.settings(editing_mode="joint")
    response = session.client.post(f"{session.url}/infer/{session.state['model_id']}", json={
        "annotation_mode": "multiclass",
    })
    assert response.status_code == 200
    assert model.calls[-1]["labels"].tolist() == [0, 1]
    assert model.calls[-1]["click_indices"][0][1].tolist() == [1]


def test_joint_prediction_survives_modality_and_display_metadata_autosave(editing: Any) -> None:
    """
    Ordinary joint settings saves do not cancel a compatible in-flight model prediction.
    """
    session, model = editing
    session.settings(editing_mode="joint")
    values = np.zeros((2, 2, 2), np.uint8)
    values.flat[0] = 3
    nib.save(nib.Nifti1Image(values, np.eye(4)), session.directory / "annotations.nii.gz")
    session.client.post(f"{session.url}/set_annots", json={"annotation_path": "annotations.nii.gz"})
    model.started, model.release = Event(), Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(
            session.client.post, f"{session.url}/infer/{session.state['model_id']}", json={
                "annotation_mode": "multiclass", "modality": "other",
            },
        )
        assert model.started.wait(10)
        renamed = [{**entry, "name": "Renamed", "color": "#ff00ff"} for entry in CLASSES]
        assert session.settings(modality="other", classes=renamed).status_code == 200
        model.release.set()
        assert response.result().status_code == 200


def test_joint_prediction_cannot_commit_after_opting_in(editing: Any) -> None:
    """
    Enabling binary editing protects its result from an older joint request.
    """
    session, model = editing
    session.settings(editing_mode="joint")
    values = np.zeros((2, 2, 2), np.uint8)
    values.flat[0] = 3
    nib.save(nib.Nifti1Image(values, np.eye(4)), session.directory / "annotations.nii.gz")
    session.client.post(f"{session.url}/set_annots", json={"annotation_path": "annotations.nii.gz"})
    model.started, model.release = Event(), Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(
            session.client.post, f"{session.url}/infer/{session.state['model_id']}", json={
                "annotation_mode": "multiclass",
            },
        )
        assert model.started.wait(10)
        assert session.settings(editing_mode="per_label").status_code == 200
        model.release.set()
        assert response.result().status_code == 409
