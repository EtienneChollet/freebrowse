"""
Persist independent label edits and compose accepted masks without changing them.

Callers hold the session manager's manifest lock while reading or changing these files. Versioned
prompt files keep inference inputs immutable while newer edits or locks arrive.
"""
import base64
import gzip
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import torch
from fastapi import HTTPException

from fiery_flame import PredictionContext


def check_version(manifest: Dict[str, Any], version: Optional[int]) -> None:
    """
    Reject stale edits instead of overwriting newer label selection, prompts, or locks.
    """
    if version != manifest.get("state_version", 0):
        raise HTTPException(409, "The session changed. Reload its latest label edits and retry.")


def require_label(manifest: Dict[str, Any], label_id: int) -> Dict[str, Any]:
    """
    Resolve a saved binary label and reject editing accepted structures.
    """
    if manifest.get("editing_mode") != "per_label":
        raise HTTPException(400, "Enable One label at a time first")
    entry = next((item for item in manifest["classes"] if item["id"] == label_id), None)
    if entry is None:
        raise HTTPException(400, "Select an existing label")
    if entry.get("locked"):
        raise HTTPException(409, "Unlock the label before editing it")
    return manifest["label_edits"][str(label_id)]


def initialize_labels(directory: Path, manifest: Dict[str, Any]) -> None:
    """
    Keep joint work separately and initialize each new binary label from its existing marks.
    """
    if "joint_edit" not in manifest:
        manifest["joint_edit"] = {
            key: manifest.get(key) for key in ("annotation_path", "result_path")
        }
    joint = manifest["joint_edit"]
    annotation = None
    if joint["annotation_path"]:
        annotation = nib.load(directory / joint["annotation_path"])
    result = nib.load(directory / joint["result_path"]) if joint["result_path"] else None
    edits = manifest.setdefault("label_edits", {})
    for entry in manifest["classes"]:
        label_id = entry["id"]
        if str(label_id) in edits:
            continue
        edit = {"context_id": uuid.uuid4().hex, "annotation_path": None, "mask_path": None}
        if annotation is not None:
            values = np.asarray(annotation.dataobj)
            marks = np.where((values == label_id) | (values == 255), values, 0).astype(np.uint8)
            edit["annotation_path"] = f"label-{label_id}-initial.nii.gz"
            nib.save(nib.Nifti1Image(marks, annotation.affine), directory / edit["annotation_path"])
        if result is not None:
            edit["mask_path"] = f"label-{label_id}-initial-mask.nii.gz"
            mask = (np.asarray(result.dataobj) == label_id).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, result.affine), directory / edit["mask_path"])
        edits[str(label_id)] = edit


def compose_masks(directory: Path, manifest: Dict[str, Any]) -> Optional[nib.Nifti1Image]:
    """
    Give accepted labels priority, then resolve remaining overlaps by ascending stable label ID.
    """
    result = None
    affine = None
    classes = sorted(manifest["classes"], key=lambda item: (not item.get("locked"), item["id"]))
    for entry in classes:
        path = manifest["label_edits"].get(str(entry["id"]), {}).get("mask_path")
        if not path:
            continue
        image = nib.load(directory / path)
        mask = np.asarray(image.dataobj) != 0
        if result is None:
            result = np.zeros(mask.shape, np.uint8)
            affine = image.affine
        result[mask & (result == 0)] = entry["id"]
    return nib.Nifti1Image(result, affine) if result is not None else None


def save_composite(directory: Path, manifest: Dict[str, Any]) -> None:
    """
    Publish a versioned combined result without overwriting the joint editing result.
    """
    image = compose_masks(directory, manifest)
    manifest["result_path"] = None
    if image is not None:
        manifest["result_path"] = f"binary-result-{manifest['state_version']}.nii.gz"
        nib.save(image, directory / manifest["result_path"])


def save_annotations(
    directory: Path, manifest: Dict[str, Any], label_id: int, encoded: str,
) -> None:
    """
    Validate a native-space label drawing and save an immutable inference input.
    """
    edit = require_label(manifest, label_id)
    try:
        payload = base64.b64decode(encoded, validate=True)
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        image = nib.Nifti1Image.from_bytes(payload)
        values = np.asarray(image.dataobj)
    except (ValueError, OSError, EOFError, nib.filebasedimages.ImageFileError) as error:
        raise HTTPException(400, "The label drawing must be a valid NIfTI image") from error
    if not np.isin(values, [0, label_id, 255]).all():
        raise HTTPException(400, "A binary drawing may contain only its label and background marks")
    path = f"label-{label_id}-annotations-{manifest['state_version']}.nii.gz"
    nib.save(image, directory / path)
    edit["annotation_path"] = path


def freeze_key(value: Any) -> Any:
    """
    Restore tuple identity after loading a prediction compatibility key from JSON.
    """
    return tuple(freeze_key(item) for item in value) if isinstance(value, list) else value


def load_prior(directory: Path, edit: Dict[str, Any]) -> Optional[PredictionContext]:
    """
    Restore saved raw logits; the runner still checks image, checkpoint, marks, and modality.
    """
    if not edit.get("prior_path"):
        return None
    with np.load(directory / edit["prior_path"], allow_pickle=False) as saved:
        return PredictionContext(
            freeze_key(json.loads(str(saved["key"]))), saved["prompt_codes"].copy(),
            torch.from_numpy(saved["logits"].copy()),
        )


def save_prior(directory: Path, edit: Dict[str, Any], prior: Optional[PredictionContext]) -> None:
    """
    Store raw model logits independently from the visible hard mask for session reopening.
    """
    edit["prior_path"] = None
    if prior is None:
        return
    path = f"prior-{edit['context_id']}.npz"
    temporary = directory / f"prior-{edit['context_id']}.tmp.npz"
    np.savez_compressed(
        temporary, key=json.dumps(prior.key), prompt_codes=prior.prompt_codes,
        logits=prior.logits.float().numpy(),
    )
    os.replace(temporary, directory / path)
    edit["prior_path"] = path
