"""
Keep generic image uploads from bypassing label locks or corrupting binary session files.
"""
import importlib.util
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import ai_session


def test_generic_upload_cannot_overwrite_binary_files(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Normal initial and joint uploads remain usable; binary sessions and generated masks reject them.
    """
    for name in ["scans", "static", "sessions", "models"]:
        (tmp_path / name).mkdir()
    directory = tmp_path / "scans" / "ai-sessions" / "demo"
    directory.mkdir(parents=True)
    manifest = directory / "session.json"
    manifest.write_text(json.dumps({"editing_mode": "joint"}))
    settings = {
        "DATA_DIR": "scans", "NIIVUE_BUILD_DIR": "static", "AI_DIR": "scans/ai-sessions",
        "MODELS_DIR": "models",
    }
    for name, folder in settings.items():
        monkeypatch.setenv(name, str(tmp_path / folder))
    monkeypatch.setenv("ENABLE_AI", "true")
    monkeypatch.setenv("READ_ONLY_DATA", "true")
    monkeypatch.setenv("SERVERLESS_MODE", "false")
    specification = importlib.util.spec_from_file_location(
        "binary_upload_server", Path(ai_session.__file__).with_name("server.py"),
    )
    server = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(server)
    client = TestClient(server.app)
    for filename in ["scan.nii.gz", "annotations.nii.gz"]:
        assert client.post("/data/nii", json={
            "filename": f"ai-sessions/demo/{filename}", "data": "AA==",
        }).status_code == 200
    protected = directory / "locked-3-abc123.nii.gz"
    protected.write_bytes(b"accepted mask")
    for mode in ["joint", "per_label"]:
        manifest.write_text(json.dumps({"editing_mode": mode}))
        assert client.post("/data/nii", json={
            "filename": "ai-sessions/demo/locked-3-abc123.nii.gz", "data": "AA==",
        }).status_code == 409
    assert protected.read_bytes() == b"accepted mask"
    monkeypatch.setattr(server, "read_only_data", False)
    normalized = "unused/../ai-sessions/demo/locked-3-abc123.nii.gz"
    assert client.post("/data/nii", json={
        "filename": normalized, "data": "AA==",
    }).status_code == 409
    assert protected.read_bytes() == b"accepted mask"
    assert client.post("/data/nii", json={
        "filename": "ai-sessions/demo/annotations.nii.gz", "data": "AQ==",
    }).status_code == 409
    assert (directory / "annotations.nii.gz").read_bytes() == b"\0"
