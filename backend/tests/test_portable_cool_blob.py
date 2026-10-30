"""
Verify portable COOL-BLOB loading, dispatch, and session-local correction handling.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend/src"))
sys.path.insert(0, str(ROOT / "models/cool-blob"))

from cool_blob import inference, model
from cool_blob.session import InteractiveSession
import ml_inference


def test_imports_do_not_require_source_repositories() -> None:
    """
    Import bundled definitions with both original source packages explicitly unavailable.
    """
    script = """
import importlib.abc
import sys

class RejectOriginalRepositories(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'interseg3d', 'anatomix'}:
            raise ImportError('Original repository is forbidden: ' + fullname)

sys.meta_path.insert(0, RejectOriginalRepositories())
sys.path.insert(0, sys.argv[1])
import cool_blob.model
import cool_blob.inference
import cool_blob.session
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "models/cool-blob")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_load_is_complete_strict_and_normalizes_compilation_keys(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """
    Strict loading includes the fine-tuned backbone and preserves sampling settings.
    """
    received: Dict[str, Any] = {}

    class RecordingModel(torch.nn.Module):
        """
        Observe model construction and state loading without allocating the full network.
        """

        def __init__(self, **options: Any) -> None:
            """
            Record architecture options passed by the production checkpoint loader.
            """
            super().__init__()
            received["options"] = options

        def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
            """
            Record the complete state and the strictness of the load operation.
            """
            received["state"] = state
            received["strict"] = strict

    monkeypatch.setattr(model, "AmixCrossAttentionClickCachedFixedTotalMultiLabel", RecordingModel)
    state = {"anatomix._orig_mod.model.weight": torch.tensor([1]), "attention.weight": 2}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": state})
    config = {
        "checkpoint_path": str(tmp_path / "weights.pt"),
        "training_config_path": str(ROOT / "models/cool-blob/model.yml"),
    }
    inference.CoolBlobRunner().load_model(config, torch.device("cpu"))
    assert received["strict"] is True
    assert set(received["state"]) == {"anatomix.model.weight", "attention.weight"}
    assert received["options"]["anatomix_compile"] is False
    assert received["options"]["anatomix_random_init"] is True
    assert received["options"]["prediction_tokens_budget"] == 1024
    assert received["options"]["prediction_tokens_selection"] == "random"


def test_freebrowse_dispatch_resolves_model_local_files(tmp_path: Path, monkeypatch: Any) -> None:
    """
    The portable entry resolves paths beside config.yml and forwards all editing options.
    """
    received: Dict[str, Any] = {}
    directory = tmp_path / "models/cool-blob"
    session = tmp_path / "session"
    session.mkdir()
    volume = tmp_path / "scan.nii.gz"
    prediction = nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.eye(4))
    config = yaml.safe_load((ROOT / "models/cool-blob/config.yml").read_text())
    original = dict(config)

    def predict(*args: Any, **options: Any) -> nib.Nifti1Image:
        """
        Capture the selected runner's resolved settings and browser session options.
        """
        received["config"] = args[3]
        received["options"] = options
        return prediction

    monkeypatch.setattr(inference.runner, "predict", predict)
    monkeypatch.setattr(ml_inference, "_resolve_volume_path", lambda *args: volume)
    result, _ = ml_inference.run_inference(
        ml_id="cool-blob", label_value=1, session_id="portable",
        cache_entry=SimpleNamespace(), session_dir=session,
        manifest={"annotation_path": "marks.nii.gz", "classes": [{"id": 3}],
                  "prediction_context_id": "edit-2"},
        data_dir=tmp_path, models_dir=directory.parent,
        model_info={"config": config, "config_path": str(directory / "config.yml")},
    )
    assert result is prediction
    assert received["config"]["checkpoint_path"] == str(directory / "weights.pt")
    assert received["config"]["training_config_path"] == str(directory / "model.yml")
    assert received["options"] == {"allowed_label_ids": [3], "context_id": "edit-2"}
    assert config == original


def test_sessions_keep_independent_correction_state(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Successive API calls retain one session cache while another session starts fresh.
    """
    (tmp_path / "config.yml").write_text(
        "training_config_path: model.yml\ncheckpoint_path: weights.pt\n",
    )
    volume = tmp_path / "scan.nii.gz"
    marks = tmp_path / "marks.nii.gz"
    volume.touch()
    marks.touch()
    calls = []

    def predict(*args: Any) -> Any:
        """
        Observe cache identity and whether a previous API call populated the session.
        """
        cache = args[2]
        calls.append((cache, getattr(cache, "prediction_context", None)))
        cache.prediction_context = "raw logits"
        return cache

    monkeypatch.setattr(inference.runner, "predict", predict)
    first = InteractiveSession(tmp_path, volume)
    second = InteractiveSession(tmp_path, volume)
    first.predict(marks)
    first.predict(marks)
    second.predict(marks)
    assert calls[0][0] is calls[1][0]
    assert calls[0][0] is not calls[2][0]
    assert [call[1] for call in calls] == [None, "raw logits", None]


@pytest.mark.parametrize("changed_marks", [{0: 1, 1: 1}, {0: 255, 1: 1}, {1: 1}])
def test_only_cumulative_marks_reuse_raw_previous_logits(changed_marks: Dict[int, int]) -> None:
    """
    Adding marks reuses raw logits; erasing or relabeling evidence discards them.
    """
    priors: List[Optional[torch.Tensor]] = []

    class RecordingPrediction:
        """
        Return raw logits while recording the previous prediction supplied by the runner.
        """

        supports_prev_prediction = True

        def __call__(self, image: torch.Tensor, **options: Any) -> torch.Tensor:
            """
            Observe prior state at the actual model boundary.
            """
            priors.append(options["prev_logits"])
            return torch.arange(16).float().reshape(1, 2, 2, 2, 2)

    prepared = inference.PreparedImage(
        torch.zeros(1, 1, 2, 2, 2), np.eye(4), np.eye(4), (2, 2, 2),
    )
    runner = inference.CoolBlobRunner()
    cache = SimpleNamespace()
    for marks in ({0: 1}, changed_marks):
        values = np.zeros((2, 2, 2), np.uint8)
        for position, label in marks.items():
            values.flat[position] = label
        annotation = nib.Nifti1Image(values, np.eye(4))
        indices, labels = inference.map_prompts(annotation, prepared)
        runner.predict_from_prompts(
            RecordingPrediction(), prepared, annotation, indices, labels, cache, ("scan",),
        )
    assert priors[0] is None
    assert (priors[1] is not None) == (changed_marks == {0: 1, 1: 1})
