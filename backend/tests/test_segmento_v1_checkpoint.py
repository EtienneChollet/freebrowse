"""
Verify complete tensor reconstruction and rejection of invalid Segmento V1 checkpoint parts.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models/segmento-v1"))

from segmento_v1.checkpoint import load_checkpoint


def _write_parts(directory: Path, state: Dict[str, Any]) -> Path:
    """
    Split one serialized tensor dictionary across small files for isolated loader tests.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    with io.BytesIO() as buffer:
        torch.save(state, buffer)
        payload = buffer.getvalue()
    for index, start in enumerate(range(0, len(payload), 200)):
        shard = directory / f"weights-{index:05d}.bin"
        shard.write_bytes(payload[start:start + 200])
        entries.append({
            "file": shard.name, "bytes": shard.stat().st_size,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        })
    manifest = directory / "weights.index.json"
    manifest.write_text(json.dumps({
        "format": "segmento-v1-torch-parts-v1", "source_checkpoint_sha256": "a" * 64,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "state_key_count": len(state), "shards": entries,
    }))
    return manifest


def test_parts_reconstruct_exact_raw_tensors(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Byte parts smaller than one tensor preserve keys and values through one safe torch.load.
    """
    first = {"anatomix._orig_mod.model.weight": torch.arange(600).reshape(20, 30)}
    second = {"attention.weight": torch.tensor([0.25, -2.0], dtype=torch.bfloat16)}
    manifest = _write_parts(tmp_path, first | second)
    original_load = torch.load
    calls = []

    def record_load(*args: Any, **options: Any) -> Any:
        """
        Confirm all verified bytes are deserialized once with safe tensor-only loading.
        """
        calls.append(options)
        return original_load(*args, **options)

    monkeypatch.setattr(torch, "load", record_load)
    restored = load_checkpoint(manifest)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert restored.keys() == (first | second).keys()
    for key, tensor in (first | second).items():
        assert torch.equal(restored[key], tensor)
        assert restored[key].dtype == tensor.dtype
        assert restored[key].device.type == "cpu"


def test_legacy_full_checkpoint_remains_supported(tmp_path: Path, monkeypatch: Any) -> None:
    """
    The original model dictionary still loads with weights_only=False and CPU mapping.
    """
    checkpoint = tmp_path / "legacy.pt"
    state = {"original.key": torch.tensor([3.0])}
    torch.save({"model": state, "optimizer": {"unused": 5}}, checkpoint)
    original_load = torch.load
    received = {}

    def record_load(*args: Any, **options: Any) -> Any:
        """
        Capture loading options without replacing tensor deserialization.
        """
        received.update(options)
        return original_load(*args, **options)

    monkeypatch.setattr(torch, "load", record_load)
    restored = load_checkpoint(checkpoint)
    assert received == {"map_location": "cpu", "weights_only": False}
    assert torch.equal(restored["original.key"], state["original.key"])


def test_missing_part_is_rejected(tmp_path: Path) -> None:
    """
    Missing files cannot produce a partially initialized model state.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    document["shards"][0]["file"] = "missing.pt"
    manifest.write_text(json.dumps(document))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(manifest)


@pytest.mark.parametrize("changed_field", ["bytes", "sha256"])
def test_size_and_checksum_are_verified_before_deserialization(
    tmp_path: Path, monkeypatch: Any, changed_field: str,
) -> None:
    """
    Tampered metadata is rejected before torch.load can deserialize the complete payload.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    entry = document["shards"][0]
    entry[changed_field] = entry["bytes"] + 1 if changed_field == "bytes" else "0" * 64
    manifest.write_text(json.dumps(document))

    def reject_load(*args: Any, **options: Any) -> Any:
        """
        Fail if checkpoint deserialization happens before metadata verification.
        """
        pytest.fail("Invalid part must be rejected before deserialization")

    monkeypatch.setattr(torch, "load", reject_load)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(manifest)


@pytest.mark.parametrize("filename", ["../outside.pt", "/outside.pt", "nested/file.pt", "..\\x.pt"])
def test_part_paths_must_be_plain_relative_basenames(tmp_path: Path, filename: str) -> None:
    """
    Manifest entries cannot traverse directories or name absolute paths.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    document["shards"][0]["file"] = filename
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="relative basename"):
        load_checkpoint(manifest)


def test_part_symlinks_cannot_escape_manifest_directory(tmp_path: Path) -> None:
    """
    A basename symlink cannot redirect checkpoint loading to files outside the model directory.
    """
    manifest = _write_parts(tmp_path / "model", {"weight": torch.ones(1)})
    outside = tmp_path / "outside.pt"
    torch.save({"weight": torch.ones(1)}, outside)
    (manifest.parent / "external.pt").symlink_to(outside)
    document = json.loads(manifest.read_text())
    document["shards"][0]["file"] = "external.pt"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="escapes"):
        load_checkpoint(manifest)


def test_changed_part_bytes_are_rejected(tmp_path: Path) -> None:
    """
    A same-size damaged part fails its checksum instead of reaching model deserialization.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    shard = tmp_path / "weights-00000.bin"
    contents = bytearray(shard.read_bytes())
    contents[-1] ^= 1
    shard.write_bytes(contents)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint(manifest)


def test_duplicate_part_filenames_are_rejected(tmp_path: Path) -> None:
    """
    A repeated part cannot silently duplicate bytes in the serialized payload.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    document["shards"].append(document["shards"][0])
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate"):
        load_checkpoint(manifest)


def test_incomplete_state_count_is_rejected(tmp_path: Path) -> None:
    """
    A missing model parameter cannot silently pass manifest reconstruction.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    document["state_key_count"] = 2
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="state count mismatch"):
        load_checkpoint(manifest)


def test_non_tensor_payload_contents_are_rejected(tmp_path: Path) -> None:
    """
    Even a checksum-valid payload must contain only named model tensors.
    """
    manifest = _write_parts(tmp_path, {"weight": "not a tensor"})
    with pytest.raises(ValueError, match="only named tensors"):
        load_checkpoint(manifest)


def test_payload_checksum_is_verified_before_deserialization(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """
    Correct individual parts cannot bypass a mismatched whole-payload checksum.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    document["payload_sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))

    def reject_load(*args: Any, **options: Any) -> Any:
        """
        Prevent deserialization when only the complete payload checksum is incorrect.
        """
        pytest.fail("Invalid payload must be rejected before deserialization")

    monkeypatch.setattr(torch, "load", reject_load)
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        load_checkpoint(manifest)


def test_empty_part_list_is_rejected(tmp_path: Path) -> None:
    """
    An empty manifest cannot initialize an incomplete network.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    document["shards"] = []
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="part list"):
        load_checkpoint(manifest)


def test_empty_part_is_rejected(tmp_path: Path) -> None:
    """
    Individual binary parts must contain at least one byte.
    """
    manifest = _write_parts(tmp_path, {"weight": torch.ones(1)})
    document = json.loads(manifest.read_text())
    entry = document["shards"][0]
    (tmp_path / entry["file"]).write_bytes(b"")
    entry.update(bytes=0, sha256=hashlib.sha256(b"").hexdigest())
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Invalid checkpoint part byte size"):
        load_checkpoint(manifest)
