"""
Verify multiclass encoding, sparse coordinate preservation, and native-grid reconstruction.
"""
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ai_session
import fiery_flame


def test_sparse_prompt_survives_downsampling_and_preserves_class_identity() -> None:
    """
    A one-voxel mark survives a downsampling grid that would miss it with bitmap interpolation.
    """
    prepared = fiery_flame.PreparedImage(
        torch.zeros(1, 1, 4, 4, 4), np.diag([4., 4., 4., 1.]), np.eye(4), (16, 16, 16),
    )
    bitmap = np.zeros(prepared.native_shape, np.uint8)
    bitmap[3, 3, 3] = 17
    bitmap[4, 3, 3] = 17
    bitmap[10, 10, 10] = 2
    bitmap[0, 0, 0] = 255
    indices, labels = fiery_flame.map_prompts(nib.Nifti1Image(bitmap, np.eye(4)), prepared)
    assert labels == [0, 2, 17]
    assert indices[0].tolist() == [0]
    assert indices[2].tolist() == [21]
    assert len(indices[1]) == 1


def test_empty_background_is_distinct_from_explicit_background() -> None:
    """
    Unmarked voxels never become background clicks.
    """
    prepared = fiery_flame.PreparedImage(
        torch.zeros(1, 1, 2, 2, 2), np.eye(4), np.eye(4), (2, 2, 2),
    )
    annotation = nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.eye(4))
    indices, labels = fiery_flame.map_prompts(annotation, prepared)
    assert labels == [0]
    assert indices[0].numel() == 0


def test_annotation_affine_mismatch_is_rejected() -> None:
    """
    An annotation from a different scan cannot silently shift prompts.
    """
    prepared = fiery_flame.PreparedImage(
        torch.zeros(1, 1, 2, 2, 2), np.eye(4), np.eye(4), (2, 2, 2),
    )
    annotation = nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.diag([2., 1., 1., 1.]))
    with pytest.raises(HTTPException, match="shape and affine"):
        fiery_flame.map_prompts(annotation, prepared)


def test_prediction_interpolates_logits_before_argmax_in_original_orientation() -> None:
    """
    A flipped native axis and half-voxel sampling retain user IDs and continuous decision borders.
    """
    native_affine = np.eye(4)
    native_affine[0, 0] = -0.5
    native_affine[0, 3] = 1
    prepared = fiery_flame.PreparedImage(
        torch.zeros(1, 1, 2, 2, 2), np.eye(4), native_affine, (3, 2, 2),
    )
    logits = torch.zeros(1, 2, 2, 2, 2)
    logits[:, 1, 0] = -1
    logits[:, 1, 1] = 3
    result = fiery_flame.restore_prediction(logits, prepared, [0, 17])
    assert np.array_equal(result.affine, native_affine)
    assert result.get_fdata()[:, 0, 0].tolist() == [17, 17, 0]


def test_preparation_matches_monai_odd_center_padding() -> None:
    """
    Training pads odd differences on the high side and retains the matching world coordinates.
    """
    import voxel
    from monai.transforms import ResizeWithPadOrCrop

    array = np.linspace(-2000, 2000, 3 * 5 * 9, dtype=np.float32).reshape(3, 5, 9)
    image = nib.Nifti1Image(array, np.eye(4))
    prepared = fiery_flame.prepare_image(image, "ct")
    volume = voxel.Volume(torch.from_numpy(np.clip(array, -1024, 1024)), torch.eye(4))
    volume = volume.resample(9 / 128, mode="linear", antialias=False, padding_mode="border")
    tensor = (volume.tensor - volume.tensor.min()) / (volume.tensor.max() - volume.tensor.min())
    expected = ResizeWithPadOrCrop((128, 128, 128))(tensor)
    assert torch.equal(prepared.tensor[0], expected)
    assert prepared.tensor.shape == (1, 1, 128, 128, 128)
    before = (128 - np.array(volume.baseshape)) // 2
    expected_origin = nib.affines.apply_affine(volume.geometry.tensor.numpy(), -before)
    assert np.allclose(prepared.affine[:3, 3], expected_origin)


def test_session_settings_preserve_classes_and_reject_wrong_model_mode(tmp_path: Path) -> None:
    """
    Saved class names and IDs travel with a multiclass session, and binary inference is gated.
    """
    models = tmp_path / "models" / "fiery-flame"
    models.mkdir(parents=True)
    (models / "config.yml").write_text("backend: fiery_flame\n")
    app = FastAPI()
    app.include_router(ai_session.build_router(
        ai_dir=tmp_path / "sessions", data_dir=tmp_path, models_dir=models.parent,
        ttl_seconds=60, enabled=True,
    ))
    client = TestClient(app)
    session = client.post("/ai/session/new", json={"session_name": "saved"}).json()
    url = f"/ai/session/{session['session_id']}"
    settings = {
        "model_id": "fiery-flame", "annotation_mode": "multiclass", "modality": "ct",
        "classes": [{"id": 17, "name": "Liver", "color": "#b6ff00"}],
    }
    assert client.post(f"{url}/settings", json=settings).status_code == 200
    saved = client.get("/ai/session/list").json()[0]
    assert saved["classes"] == settings["classes"]
    assert saved["model_id"] == "fiery-flame"
    wrong_settings = client.post(f"{url}/settings", json={**settings, "annotation_mode": "binary"})
    wrong_inference = client.post(f"{url}/infer/fiery-flame", json={"annotation_mode": "binary"})
    assert wrong_settings.status_code == wrong_inference.status_code == 400


def test_read_only_scan_storage_allows_only_existing_session_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Separate AI storage accepts annotations while the source scan directory remains unwritable.
    """
    for directory in ["scans", "static", "sessions", "models"]:
        (tmp_path / directory).mkdir()
    (tmp_path / "sessions" / "demo").mkdir()
    (tmp_path / "sessions" / "demo" / "session.json").write_text("{}")
    for key, value in {
        "DATA_DIR": str(tmp_path / "scans"), "NIIVUE_BUILD_DIR": str(tmp_path / "static"),
        "AI_DIR": str(tmp_path / "sessions"), "MODELS_DIR": str(tmp_path / "models"),
        "ENABLE_AI": "true", "READ_ONLY_DATA": "true", "SERVERLESS_MODE": "false",
    }.items():
        monkeypatch.setenv(key, value)
    server_path = Path(ai_session.__file__).with_name("server.py")
    spec = importlib.util.spec_from_file_location("isolated_server", server_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = TestClient(module.app)
    assert client.post("/nvd", json={"filename": "scene.nvd", "data": {}}).status_code == 403
    forbidden = client.post("/data/nii", json={"filename": "scan.nii", "data": "AA=="})
    missing = client.post(
        "/data/nii", json={"filename": "ai-sessions/missing/a.nii", "data": "AA=="},
    )
    assert forbidden.status_code == 403
    assert missing.status_code == 400
    response = client.post("/data/nii", json={"filename": "ai-sessions/demo/a.nii", "data": "AA=="})
    assert response.status_code == 200
    assert (tmp_path / "sessions" / "demo" / "a.nii").read_bytes() == b"\0"
    assert not list((tmp_path / "scans").iterdir())
    assert client.get("/data/ai-sessions/demo/a.nii").status_code == 200


def test_uploaded_scan_binds_session_root_despite_filename_collision(tmp_path: Path) -> None:
    """
    A local upload must never bind the unrelated source scan with the same basename.
    """
    manager = ai_session.SessionManager(tmp_path / "sessions", tmp_path / "source", 60)
    manager.data_dir.mkdir()
    manifest = manager.create("upload")
    session_id = manifest["session_id"]
    session_dir, _ = manager.find_by_id(session_id)
    (manager.data_dir / "scan.nii.gz").write_bytes(b"source")
    (session_dir / "scan.nii.gz").write_bytes(b"uploaded")
    saved = manager.set_volume(session_id, "scan.nii.gz", root="session")
    assert saved["volume_path_root"] == "session"
    assert saved["volume_path"] == "scan.nii.gz"
