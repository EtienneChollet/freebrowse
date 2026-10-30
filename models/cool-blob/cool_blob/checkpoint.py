"""
Load complete Segmento v1 tensors from a legacy checkpoint or verified local binary parts.
"""
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Dict

import torch


def _append_part(
    directory: Path, entry: Dict[str, Any], buffer: io.BytesIO, payload_hash: Any,
) -> None:
    """
    Verify one local part while appending its bytes and updating the complete payload checksum.
    """
    filename = entry["file"]
    if filename in ("", ".", "..") or "/" in filename or "\\" in filename:
        raise ValueError("Checkpoint part file must be a relative basename")
    part = (directory / filename).resolve()
    if part.parent != directory:
        raise ValueError(f"Checkpoint part escapes the model directory: {filename}")
    expected_size, expected_hash = entry.get("bytes"), entry.get("sha256")
    if type(expected_size) is not int or expected_size < 1:
        raise ValueError(f"Invalid checkpoint part byte size: {filename}")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError(f"Invalid checkpoint part SHA-256: {filename}")
    if part.stat().st_size != expected_size:
        raise ValueError(f"Checkpoint part byte size mismatch: {filename}")
    digest = hashlib.sha256()
    with part.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            payload_hash.update(block)
            buffer.write(block)
    if digest.hexdigest() != expected_hash.lower():
        raise ValueError(f"Checkpoint part SHA-256 mismatch: {filename}")


def load_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """
    Return the complete raw model state without renaming keys or changing tensor values.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Legacy full .pt checkpoint or a cool-blob-torch-parts-v1 JSON manifest. Parts must
        resolve inside the manifest directory and pass their recorded sizes and SHA-256 hashes.

    Returns
    -------
    dict of str to torch.Tensor
        CPU model tensors, including the fine-tuned backbone, ready for strict loading.

    Notes
    -----
    Legacy checkpoints preserve weights_only=False loading. Binary parts reconstruct one
    serialized tensor dictionary, verified before weights_only=True loading. Optimizer state
    is unused. Splitting serialized bytes preserves tensors larger than an individual part.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() != ".json":
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    with checkpoint_path.open() as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("format") != "cool-blob-torch-parts-v1":
        raise ValueError("Unsupported Segmento v1 checkpoint manifest format")
    for name in ("source_checkpoint_sha256", "payload_sha256"):
        digest = manifest.get(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"Invalid checkpoint manifest SHA-256: {name}")
    count, parts = manifest.get("state_key_count"), manifest.get("shards")
    if type(count) is not int or count < 1 or not isinstance(parts, list) or not parts:
        raise ValueError("Checkpoint manifest requires a positive state count and part list")
    directory = checkpoint_path.parent.resolve()
    filenames = set()
    payload_hash = hashlib.sha256()
    with io.BytesIO() as buffer:
        for entry in parts:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ValueError("Checkpoint part entries require a file basename")
            if entry["file"] in filenames:
                raise ValueError("Checkpoint manifest contains duplicate part filenames")
            filenames.add(entry["file"])
            _append_part(directory, entry, buffer, payload_hash)
        if payload_hash.hexdigest() != manifest["payload_sha256"].lower():
            raise ValueError("Checkpoint payload SHA-256 mismatch")
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint payload must contain a tensor dictionary")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint payload must contain only named tensors")
    if len(state) != count:
        raise ValueError(f"Checkpoint state count mismatch: expected {count}, found {len(state)}")
    return state
