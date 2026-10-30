"""AI session router.

Persistent, server-keyed sessions backed by a folder under AI_DIR. Each session
carries a reference to a volume and (optionally) an annotation mask and a result
mask. A RAM cache with a TTL keeps the preprocessed volume tensor and last
inference logits available across rapid successive calls; the manifest on disk
is the source of truth.
"""
import json
import copy
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
import binary_editing
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MANIFEST_FILENAME = "session.json"
_VOLUME_ROOT_DATA = "data"
_VOLUME_ROOT_SESSION = "session"
_PER_LABEL_MODELS = {"cool-blob", "open-whale", "grey-plant"}


class NewSessionRequest(BaseModel):
    session_name: str


class SetVolumeRequest(BaseModel):
    volume_path: str
    volume_path_root: Optional[Literal["data", "session"]] = None


class SetAnnotsRequest(BaseModel):
    annotation_path: str


class AnnotationClass(BaseModel):
    """
    Stable foreground ID and its editable display name and RGB color.
    """

    id: int = Field(ge=1, le=254)
    name: str = Field(min_length=1, max_length=100)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    locked: bool = False


class SessionSettingsRequest(BaseModel):
    """
    Persist the annotation meaning with the session, independently of predictions.
    """

    model_id: str
    annotation_mode: Literal["binary", "multiclass"]
    classes: list[AnnotationClass] = Field(default_factory=list, max_length=254)
    modality: Literal["ct", "other"] = "ct"
    editing_mode: Literal["joint", "per_label"] = "joint"
    active_label_id: Optional[int] = Field(default=None, ge=1, le=254)
    state_version: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_class_ids(self) -> "SessionSettingsRequest":
        """
        Reject duplicate IDs, which would make saved annotations ambiguous.
        """
        ids = [entry.id for entry in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("Class IDs must be unique")
        return self


class InferRequest(BaseModel):
    """
    Request inference with an optional editing-context ID that isolates prediction feedback.
    """

    label_value: int = 1
    annotation_mode: Optional[Literal["binary", "multiclass"]] = None
    modality: Optional[Literal["ct", "other"]] = None
    prediction_context_id: Optional[str] = Field(default=None, max_length=64)
    editing_mode: Literal["joint", "per_label"] = "joint"
    active_label_id: Optional[int] = Field(default=None, ge=1, le=254)
    state_version: Optional[int] = Field(default=None, ge=0)


class LabelDrawingRequest(BaseModel):
    """
    Save one native-space label drawing against an explicit session revision.
    """

    label_id: int = Field(ge=1, le=254)
    state_version: int = Field(ge=0)
    data: Optional[str] = None


@dataclass
class _SessionCacheEntry:
    """
    Hold one session's image caches and model state until invalidation or inactivity expiry.

    Iterative Amix context survives annotation uploads; the runner checks mark removal and
    context identity before reuse. Legacy binary logits retain their existing invalidation.
    """

    last_touched: float = field(default_factory=time.monotonic)
    volume_tensor: Any = None
    affine_ras: Any = None
    ras_dims: tuple[int, int, int] | None = None
    shape_before_pad: tuple[int, int, int] | None = None
    previous_logits: Any = None
    prediction_context: Any = None
    nninteractive_session: Any = None
    nninteractive_ml_id: str | None = None
    nninteractive_volume_path: str | None = None
    nninteractive_affine_ras: Any = None
    nninteractive_ras_dims: tuple[int, int, int] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SessionManager:
    """Owns session manifests on disk and the RAM cache with lazy TTL eviction.

    Filesystem is authoritative. The RAM cache exists only to avoid repeated
    disk I/O and preprocessing inside an active editing session.
    """

    def __init__(
        self,
        ai_dir: Path,
        data_dir: Path,
        ttl_seconds: int,
        enable_history: bool = False,
    ):
        self.ai_dir = ai_dir
        self.data_dir = data_dir
        self.ttl_seconds = ttl_seconds
        self.enable_history = enable_history
        self._cache: dict[str, _SessionCacheEntry] = {}
        self._lock = threading.Lock()
        self._manifest_lock = threading.RLock()

    # ----- cache -----
    def _sweep_expired(self) -> None:
        now = time.monotonic()
        stale = [sid for sid, e in self._cache.items()
                 if now - e.last_touched > self.ttl_seconds]
        for sid in stale:
            self._cache.pop(sid, None)
            logger.info(f"AI session cache: evicted {sid} (TTL)")

    def touch(self, session_id: str) -> _SessionCacheEntry:
        with self._lock:
            self._sweep_expired()
            entry = self._cache.get(session_id)
            if entry is None:
                entry = _SessionCacheEntry()
                self._cache[session_id] = entry
            entry.last_touched = time.monotonic()
            return entry

    def invalidate_all(self, session_id: str) -> None:
        with self._lock:
            self._cache.pop(session_id, None)

    def invalidate_logits(self, session_id: str) -> None:
        with self._lock:
            entry = self._cache.get(session_id)
            if entry is not None:
                entry.previous_logits = None
                if entry.nninteractive_session is not None:
                    entry.nninteractive_session.reset_interactions()
                entry.last_touched = time.monotonic()

    # ----- manifest I/O -----
    def _session_dir_by_name(self, session_name: str) -> Path:
        return self.ai_dir / session_name

    def _read_manifest(self, manifest_path: Path) -> dict[str, Any]:
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _write_manifest(self, manifest_path: Path, manifest: dict[str, Any]) -> None:
        """
        Replace a complete manifest atomically, serializing temporary-file ownership.
        """
        with self._manifest_lock:
            tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp, manifest_path)

    def update_manifest(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """
        Merge session edits atomically so a prediction cannot overwrite newer class metadata.
        """
        with self._manifest_lock:
            session_dir, manifest = self.find_by_id(session_id)
            manifest.update(updates)
            self._write_manifest(session_dir / _MANIFEST_FILENAME, manifest)
            return manifest

    def _iter_session_manifests(self):
        if not self.ai_dir.exists():
            return
        for entry in sorted(self.ai_dir.iterdir()):
            if not entry.is_dir():
                continue
            manifest = entry / _MANIFEST_FILENAME
            if not manifest.exists():
                continue
            try:
                yield entry, self._read_manifest(manifest)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"Skipping unreadable session {entry.name}: {exc}")

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = [m for _dir, m in self._iter_session_manifests()]
        return sorted(sessions, key=lambda m: m.get("created_at", ""))

    def find_by_id(self, session_id: str) -> tuple[Path, dict[str, Any]]:
        for session_dir, manifest in self._iter_session_manifests():
            if manifest.get("session_id") == session_id:
                return session_dir, manifest
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    def create(self, session_name: str) -> dict[str, Any]:
        if not _SESSION_NAME_RE.match(session_name):
            raise HTTPException(
                status_code=400,
                detail="session_name must match [A-Za-z0-9_-]+",
            )
        session_dir = self._session_dir_by_name(session_name)
        if session_dir.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Session '{session_name}' already exists",
            )
        self.ai_dir.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=False)
        manifest: dict[str, Any] = {
            "session_id": uuid.uuid4().hex,
            "session_name": session_name,
            "created_at": _now_iso(),
            "volume_path": None,
            "volume_path_root": None,
            "annotation_path": None,
            "result_path": None,
            "last_inference_ml_id": None,
            "last_inference_at": None,
        }
        if self.enable_history:
            manifest["iteration_count"] = 0
            manifest["iterations"] = []
        self._write_manifest(session_dir / _MANIFEST_FILENAME, manifest)
        logger.info(f"AI session created: {session_name} ({manifest['session_id']})")
        return manifest

    def delete(self, session_id: str) -> None:
        session_dir, _manifest = self.find_by_id(session_id)
        shutil.rmtree(session_dir)
        self.invalidate_all(session_id)
        logger.info(f"AI session deleted: {session_id}")

    # ----- path resolution -----
    def _resolve_under(self, root: Path, rel: str) -> Path | None:
        try:
            candidate = (root / rel).resolve()
        except (OSError, ValueError):
            return None
        root_resolved = root.resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            return None
        if not candidate.exists():
            return None
        return candidate

    def resolve_path_for_session(
        self,
        rel: str,
        session_dir: Path,
        allow_data_root: bool,
    ) -> tuple[Path, str]:
        if allow_data_root:
            resolved = self._resolve_under(self.data_dir, rel)
            if resolved is not None:
                return resolved, _VOLUME_ROOT_DATA
        resolved = self._resolve_under(session_dir, rel)
        if resolved is not None:
            return resolved, _VOLUME_ROOT_SESSION
        raise HTTPException(
            status_code=400,
            detail=f"Path does not resolve to an existing file: {rel}",
        )

    # ----- mutations -----
    def set_volume(
        self, session_id: str, rel: str, root: Optional[Literal["data", "session"]] = None,
    ) -> dict[str, Any]:
        """
        Reference an existing scan and invalidate prepared image features for the session.
        """
        session_dir, manifest = self.find_by_id(session_id)
        if manifest.get("label_edits"):
            raise HTTPException(409, "Create a new session to edit another scan")
        if root is None:
            _abs, root_tag = self.resolve_path_for_session(rel, session_dir, allow_data_root=True)
        else:
            storage = self.data_dir if root == "data" else session_dir
            if self._resolve_under(storage, rel) is None:
                raise HTTPException(400, f"Volume not found in {root} storage: {rel}")
            root_tag = root
        manifest = self.update_manifest(session_id, {
            "volume_path": rel, "volume_path_root": root_tag,
            "workflow_version": manifest.get("workflow_version", 0) + 1,
            "annotation_path": None, "result_path": None,
            "last_inference_ml_id": None, "last_inference_at": None,
        })
        self.invalidate_all(session_id)
        return manifest

    def set_annots(self, session_id: str, rel: str) -> dict[str, Any]:
        """
        Reference an annotation file within the session and invalidate prior prediction state.
        """
        session_dir, manifest = self.find_by_id(session_id)
        if manifest.get("editing_mode") == "per_label":
            raise HTTPException(409, "Save annotations for the active label separately")
        self.resolve_path_for_session(
            rel=rel, session_dir=session_dir, allow_data_root=False,
        )
        manifest = self.update_manifest(session_id, {"annotation_path": rel})
        self.invalidate_logits(session_id)
        return manifest

    def record_inference(
        self,
        session_id: str,
        ml_id: str,
        result_rel_path: str,
    ) -> dict[str, Any]:
        """
        Record a completed prediction without overwriting concurrent class metadata changes.
        """
        return self.update_manifest(session_id, {
            "result_path": result_rel_path, "last_inference_ml_id": ml_id,
            "last_inference_at": _now_iso(),
        })

    def set_settings(self, session_id: str, settings: SessionSettingsRequest) -> dict[str, Any]:
        """
        Store the model, annotation encoding, classes, and scan modality together.
        """
        with self._manifest_lock:
            directory, manifest = self.find_by_id(session_id)
            old_mode = manifest.get("editing_mode", "joint")
            if old_mode == "per_label" or settings.editing_mode == "per_label":
                binary_editing.check_version(manifest, settings.state_version)
            incoming = settings.model_dump(exclude={"state_version"})
            incoming["classes"] = []
            for entry in settings.classes:
                incoming["classes"].append(entry.model_dump(exclude_unset=True))
            previous = {item["id"]: item for item in manifest.get("classes", [])}
            current = {item["id"]: item for item in incoming["classes"]}
            locked = [item for item in previous.values() if item.get("locked")]
            changing_model = settings.model_id != manifest.get("model_id")
            if locked and (settings.editing_mode != "per_label" or changing_model):
                raise HTTPException(409, "Unlock accepted labels before changing the model or mode")
            if any(item["id"] not in current for item in locked):
                raise HTTPException(409, "Unlock the label before deleting it")
            if settings.editing_mode == "per_label" and settings.active_label_id not in current:
                raise HTTPException(400, "Select an existing active label")
            has_requested_locks = any(item.get("locked") for item in current.values())
            if settings.editing_mode == "joint" and has_requested_locks:
                raise HTTPException(400, "Label locking requires One label at a time")
            if all(manifest.get(key, "joint" if key == "editing_mode" else None) == value
                   for key, value in incoming.items()):
                return manifest
            if old_mode == "joint" and settings.editing_mode == "per_label":
                manifest["joint_edit"] = {
                    key: manifest.get(key) for key in ("annotation_path", "result_path")
                }
            if old_mode == "per_label":
                composite = binary_editing.compose_masks(directory, manifest)
                for label_id, item in current.items():
                    if not item.get("locked") or previous.get(label_id, {}).get("locked"):
                        continue
                    edit = manifest["label_edits"].get(str(label_id), {})
                    if composite is None or not edit.get("mask_path"):
                        raise HTTPException(409, "Predict this label before locking it")
                    import nibabel as nib
                    import numpy as np
                    path = f"locked-{label_id}-{uuid.uuid4().hex}.nii.gz"
                    mask = (np.asarray(composite.dataobj) == label_id).astype(np.uint8)
                    nib.save(nib.Nifti1Image(mask, composite.affine), directory / path)
                    edit["mask_path"] = path
            manifest.update(incoming)
            if old_mode != settings.editing_mode or changing_model:
                manifest["workflow_version"] = manifest.get("workflow_version", 0) + 1
            for label_id in set(previous) - set(current):
                manifest.get("label_edits", {}).pop(str(label_id), None)
            manifest["state_version"] = manifest.get("state_version", 0) + 1
            if settings.editing_mode == "per_label":
                binary_editing.initialize_labels(directory, manifest)
                manifest["annotation_path"] = None
                binary_editing.save_composite(directory, manifest)
            elif old_mode == "per_label":
                manifest.update(manifest["joint_edit"])
            self._write_manifest(directory / _MANIFEST_FILENAME, manifest)
            return manifest

    def save_label_drawing(
        self, session_id: str, request: LabelDrawingRequest, clear: bool = False,
    ) -> dict[str, Any]:
        """
        Save or clear only an unlocked label and invalidate any older in-flight prediction.
        """
        with self._manifest_lock:
            directory, manifest = self.find_by_id(session_id)
            binary_editing.check_version(manifest, request.state_version)
            edit = binary_editing.require_label(manifest, request.label_id)
            manifest["state_version"] += 1
            if clear:
                edit.update(annotation_path=None, mask_path=None, prior_path=None)
                edit["context_id"] = uuid.uuid4().hex
                binary_editing.save_composite(directory, manifest)
            else:
                if request.data is None:
                    raise HTTPException(400, "A label drawing is required")
                binary_editing.save_annotations(directory, manifest, request.label_id, request.data)
                import nibabel as nib
                import numpy as np
                root = self.data_dir if manifest["volume_path_root"] == "data" else directory
                volume = nib.load(root / manifest["volume_path"])
                drawing = nib.load(directory / edit["annotation_path"])
                if drawing.shape != volume.shape or not np.allclose(drawing.affine, volume.affine):
                    raise HTTPException(400, "The drawing must match the scan shape and affine")
            self._write_manifest(directory / _MANIFEST_FILENAME, manifest)
            return manifest

    def infer_label(
        self, session_id: str, ml_id: str, request: InferRequest, model_info: dict[str, Any],
        models_dir: Path,
    ) -> dict[str, Any]:
        """
        Predict one binary structure and commit only if its session revision is still current.
        """
        from ml_inference import run_inference
        import nibabel as nib
        import numpy as np

        with self._manifest_lock:
            directory, manifest = self.find_by_id(session_id)
            binary_editing.check_version(manifest, request.state_version)
            label_id = request.active_label_id
            if label_id != manifest.get("active_label_id") or ml_id != manifest.get("model_id"):
                raise HTTPException(409, "The active label or model changed")
            edit = binary_editing.require_label(manifest, label_id)
            if not edit.get("annotation_path"):
                raise HTTPException(400, "Add foreground marks for this label first")
            cache = copy.copy(self.touch(session_id))
            cache.prediction_context = binary_editing.load_prior(directory, edit)
            inference_manifest = {
                **manifest, "annotation_path": edit["annotation_path"],
                "prediction_context_id": edit["context_id"], "active_label_id": label_id,
            }
        prediction, _ = run_inference(
            session_id=session_id, ml_id=ml_id, label_value=label_id, cache_entry=cache,
            session_dir=directory, manifest=inference_manifest, data_dir=self.data_dir,
            models_dir=models_dir, model_info=model_info,
        )
        with self._manifest_lock:
            directory, latest = self.find_by_id(session_id)
            image_cache = self.touch(session_id)
            if hasattr(cache, "fiery_image"):
                image_cache.fiery_image = cache.fiery_image
                image_cache.fiery_image_key = cache.fiery_image_key
            binary_editing.check_version(latest, request.state_version)
            edit = binary_editing.require_label(latest, label_id)
            latest["state_version"] += 1
            path = f"label-{label_id}-mask-{latest['state_version']}.nii.gz"
            values = (np.asarray(prediction.dataobj) == label_id).astype(np.uint8)
            nib.save(nib.Nifti1Image(values, prediction.affine), directory / path)
            edit["mask_path"] = path
            binary_editing.save_prior(directory, edit, cache.prediction_context)
            binary_editing.save_composite(directory, latest)
            latest.update(last_inference_ml_id=ml_id, last_inference_at=_now_iso())
            self._write_manifest(directory / _MANIFEST_FILENAME, latest)
            return latest

    def record_iteration(
        self,
        session_id: str,
        ml_id: str,
        result_rel_path: str,
        iteration_entry: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Append a history entry atomically with the latest class metadata and result reference.
        """
        with self._manifest_lock:
            session_dir, manifest = self.find_by_id(session_id)
            manifest["result_path"] = result_rel_path
            manifest["last_inference_ml_id"] = ml_id
            manifest["last_inference_at"] = _now_iso()
            iterations = manifest.get("iterations") or []
            iterations.append(iteration_entry)
            manifest["iterations"] = iterations
            manifest["iteration_count"] = manifest.get("iteration_count", 0) + 1
            self._write_manifest(session_dir / _MANIFEST_FILENAME, manifest)
            return manifest


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"Could not parse {path}: {exc}")
        return None


def _model_backend(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "sp3d"
    return str(config.get("backend", "sp3d")).lower()


def _model_display_name(model_dir: Path, config: dict[str, Any] | None) -> str:
    if isinstance(config, dict) and config.get("name"):
        return str(config["name"])
    return model_dir.name


def _list_models(models_dir: Path) -> list[dict[str, Any]]:
    """
    Discover model configurations and declare each model's annotation encoding.
    """
    if not models_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_file = model_dir / "model.py"
        weights_file = model_dir / "weights.pt"
        config_file = model_dir / "config.yml"
        config = _load_yaml(config_file) if config_file.exists() else None
        backend = _model_backend(config)

        if backend == "sp3d" and not (model_file.exists() and weights_file.exists()):
            continue
        if backend == "nninteractive" and not isinstance(config, dict):
            continue
        if backend == "fiery_flame" and not isinstance(config, dict):
            continue

        info: dict[str, Any] = {
            "ml_id": model_dir.name,
            "name": _model_display_name(model_dir, config),
            "backend": backend,
            "annotation_mode": "multiclass" if backend == "fiery_flame" else "binary",
            "model_module_path": str(model_file) if model_file.exists() else None,
            "checkpoint_path": str(weights_file) if weights_file.exists() else None,
            "config_path": str(config_file) if config_file.exists() else None,
            "config": config,
        }
        out.append(info)
    return out


def build_router(
    *,
    ai_dir: Path,
    data_dir: Path,
    models_dir: Path,
    ttl_seconds: int,
    enabled: bool,
    enable_history: bool = False,
) -> APIRouter:
    """Build the /ai/* router. When `enabled` is False, every route returns 404."""
    router = APIRouter(prefix="/ai")
    manager = SessionManager(
        ai_dir=ai_dir,
        data_dir=data_dir,
        ttl_seconds=ttl_seconds,
        enable_history=enable_history,
    )

    def _require_enabled() -> None:
        if not enabled:
            raise HTTPException(status_code=404, detail="AI endpoints disabled")

    def _require_ml_id(ml_id: str) -> dict[str, Any]:
        for m in _list_models(models_dir):
            if m["ml_id"] == ml_id:
                return m
        raise HTTPException(status_code=404, detail=f"Unknown ml_id: {ml_id}")

    @router.get("/model/list")
    def model_list():
        _require_enabled()
        return _list_models(models_dir)

    @router.get("/session/list")
    def session_list():
        _require_enabled()
        return manager.list_sessions()

    @router.post("/session/new")
    def session_new(request: NewSessionRequest):
        _require_enabled()
        manifest = manager.create(request.session_name)
        return {
            "session_id": manifest["session_id"],
            "session_name": manifest["session_name"],
        }

    @router.delete("/session/{session_id}")
    def session_delete(session_id: str):
        _require_enabled()
        manager.delete(session_id)
        return {"success": True}

    @router.post("/session/{session_id}/set_volume")
    def session_set_volume(session_id: str, request: SetVolumeRequest):
        _require_enabled()
        manifest = manager.set_volume(session_id, request.volume_path, request.volume_path_root)
        return {
            "success": True,
            "volume_path": manifest["volume_path"],
            "volume_path_root": manifest["volume_path_root"],
        }

    @router.post("/session/{session_id}/set_annots")
    def session_set_annots(session_id: str, request: SetAnnotsRequest):
        _require_enabled()
        with manager._manifest_lock:
            manifest = manager.set_annots(session_id, request.annotation_path)
        return {"success": True, "annotation_path": manifest["annotation_path"]}

    @router.get("/session/{session_id}")
    def session_state(session_id: str) -> dict[str, Any]:
        """
        Return authoritative label edits after reopening or a rejected stale edit.
        """
        _require_enabled()
        with manager._manifest_lock:
            return manager.find_by_id(session_id)[1]

    @router.post("/session/{session_id}/binary/annotations")
    def session_label_annotations(session_id: str, request: LabelDrawingRequest) -> dict[str, Any]:
        """
        Persist label prompts even when automatic prediction is disabled.
        """
        _require_enabled()
        return manager.save_label_drawing(session_id, request)

    @router.post("/session/{session_id}/binary/clear")
    def session_label_clear(session_id: str, request: LabelDrawingRequest) -> dict[str, Any]:
        """
        Clear an unlocked label without changing other labels' prompts or predictions.
        """
        _require_enabled()
        return manager.save_label_drawing(session_id, request, clear=True)

    @router.post("/session/{session_id}/infer/{ml_id}")
    def session_infer(
        session_id: str, ml_id: str, request: Optional[InferRequest] = None,
    ) -> dict[str, Any]:
        """
        Predict and save a session result, isolating iterative feedback by editing-context ID.
        """
        _require_enabled()
        session_dir, manifest = manager.find_by_id(session_id)
        model_info = _require_ml_id(ml_id)
        if manifest.get("editing_mode") == "per_label" or (
            request and request.editing_mode == "per_label"
        ):
            if not request or request.editing_mode != "per_label" or ml_id not in _PER_LABEL_MODELS:
                raise HTTPException(409, "Use One label at a time inference for the active label")
            result = manager.infer_label(session_id, ml_id, request, model_info, models_dir)
            return {"success": True, "ml_id": ml_id, **result}
        expected_mode = model_info["annotation_mode"]
        requested_mode = request.annotation_mode if request else None
        stored_mode = manifest.get("annotation_mode", "binary")
        if manifest.get("annotation_mode") and requested_mode and requested_mode != stored_mode:
            raise HTTPException(400, "The annotation mode does not match the saved session")
        if (requested_mode or stored_mode) != expected_mode:
            raise HTTPException(400, "The annotation mode does not match the selected model")
        if not manifest.get("volume_path"):
            raise HTTPException(status_code=400, detail="Session has no volume set")
        if not manifest.get("annotation_path"):
            raise HTTPException(status_code=400, detail="Session has no annotations set")
        label_value = 1 if request is None else max(0, min(255, request.label_value))

        from ml_inference import run_inference  # lazy; only when AI is enabled
        import nibabel as nib

        cache_entry = manager.touch(session_id)
        if request and request.modality:
            manifest = {**manifest, "modality": request.modality}
        manifest = {
            **manifest,
            "prediction_context_id": request.prediction_context_id if request else None,
        }
        nii, _affine = run_inference(
            session_id=session_id,
            ml_id=ml_id,
            label_value=label_value,
            cache_entry=cache_entry,
            session_dir=session_dir,
            manifest=manifest,
            data_dir=manager.data_dir,
            models_dir=models_dir,
            model_info=model_info,
        )

        result_rel = "result.nii.gz"
        result_abs = session_dir / result_rel
        with manager._manifest_lock:
            _directory, latest = manager.find_by_id(session_id)
            if (latest.get("editing_mode", "joint") != "joint" or
                    latest.get("workflow_version", 0) != manifest.get("workflow_version", 0)):
                raise HTTPException(409, "Editing mode changed while predicting")
            nib.save(nii, str(result_abs))

            if manager.enable_history:
                n = manifest.get("iteration_count", 0)
                numbered_result = f"result_{n:03d}.nii.gz"
                numbered_annot = f"annotations_{n:03d}.nii.gz"
                annot_rel = manifest["annotation_path"]
                annot_abs, _root = manager.resolve_path_for_session(
                    rel=annot_rel, session_dir=session_dir, allow_data_root=False,
                )
                annotated_at = datetime.fromtimestamp(
                    annot_abs.stat().st_mtime, tz=timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                shutil.copy2(result_abs, session_dir / numbered_result)
                shutil.copy2(annot_abs, session_dir / numbered_annot)
                iteration_entry = {
                    "iteration": n, "annotation_path": numbered_annot,
                    "result_path": numbered_result, "ml_id": ml_id,
                    "label_value": label_value, "annotated_at": annotated_at,
                    "inferred_at": _now_iso(),
                }
                manager.record_iteration(session_id, ml_id, result_rel, iteration_entry)
            else:
                manager.record_inference(session_id, ml_id, result_rel)

        return {
            "success": True,
            "ml_id": ml_id,
            "result_path": result_rel,
            "label_value": label_value,
        }

    @router.post("/session/{session_id}/settings")
    def session_settings(session_id: str, request: SessionSettingsRequest) -> dict[str, Any]:
        """
        Save class metadata and modality after checking the selected model's capability.
        """
        _require_enabled()
        model = _require_ml_id(request.model_id)
        if model["annotation_mode"] != request.annotation_mode:
            raise HTTPException(400, "The annotation mode does not match the selected model")
        if request.editing_mode == "per_label" and request.model_id not in _PER_LABEL_MODELS:
            raise HTTPException(400, "This model does not support One label at a time")
        manifest = manager.set_settings(session_id, request)
        return {"success": True, **manifest}

    return router
