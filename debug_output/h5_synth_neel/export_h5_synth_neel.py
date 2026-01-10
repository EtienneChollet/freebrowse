import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np


REPO_DIR = Path('/autofs/cluster/dalcalab2/users/epc28/projects/freebrowse_bhim')
OUTPUT_DIR = REPO_DIR / 'debug_output' / 'h5_synth_neel'
H5_PATH = Path(
    '/autofs/cluster/dalcalab2/users/nd480/mid-projects/dataaug-real-pairing/'
    'multi_data_engine_h5/train_data.hdf5'
)
INDEX_PATH = Path(
    '/autofs/cluster/dalcalab2/users/epc28/projects/interseg3d/interseg3d/datasets/'
    'h5_with_context/index.py'
)
CONFIG_PATH = Path(
    '/autofs/cluster/dalcalab2/users/epc28/projects/interseg3d/interseg3d-notebooks/'
    'jupyter/scribbleprompt3d/2026-04-15_synth/configs/2026-04-17_h5-synth-only/'
    'config.yml'
)
ENGINES = ['amix', 'brains', 'WBCT', 'AnyStar', 'Vessels']
PROMPT_ITER_NAME = 'prompt_iter_000'
CHANNEL_FILES = {
    'volume': '0_volume.nii.gz',
    'previous_pred': '1_previous_pred.nii.gz',
    'positive': '2_positive.nii.gz',
    'negative': '3_negative.nii.gz',
    'unused': '4_unused.nii.gz',
    'ground_truth': '5_ground_truth.nii.gz',
}


def load_index_module(index_path: Path) -> Any:
    """
    Load the H5 task-index helper from an absolute source path.

    Parameters
    ----------
    index_path : Path
        Absolute path to `h5_with_context/index.py`.

    Returns
    -------
    Any
        Loaded Python module exposing `probe_engine_assignments`.

    Examples
    --------
    >>> module = load_index_module(INDEX_PATH)
    >>> hasattr(module, 'probe_engine_assignments')
    True

    Notes
    -----
    Importing by path keeps this artifact script independent of package install state.
    """
    spec = importlib.util.spec_from_file_location('h5_index_epc2027', index_path)
    assert spec is not None, f'could not build import spec for {index_path}'
    assert spec.loader is not None, f'import spec has no loader for {index_path}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_positive_prompt(target: np.ndarray) -> np.ndarray:
    """
    Create a small deterministic positive prompt inside the target mask.

    Parameters
    ----------
    target : np.ndarray
        Binary target mask with shape `(D, H, W)`.

    Returns
    -------
    np.ndarray
        Binary positive prompt mask with shape `(D, H, W)`.

    Examples
    --------
    >>> mask = np.zeros((5, 5, 5), dtype=np.float32)
    >>> mask[2, 2, 2] = 1
    >>> create_positive_prompt(mask).sum()
    np.float32(1.0)

    Notes
    -----
    The seed voxel is the foreground voxel nearest the foreground centroid.
    """
    assert target.ndim == 3, f'target must be 3d, got shape {target.shape}'
    assert np.any(target > 0), 'target must contain foreground'

    coordinates = np.argwhere(target > 0)
    centroid = coordinates.mean(axis=0)
    distances = np.sum((coordinates - centroid) ** 2, axis=1)
    seed_coordinate = coordinates[int(np.argmin(distances))]

    slices = []
    for coordinate, size in zip(seed_coordinate, target.shape):
        start = max(int(coordinate) - 1, 0)
        stop = min(int(coordinate) + 2, int(size))
        slices.append(slice(start, stop))

    positive = np.zeros_like(target, dtype=np.float32)
    prompt_region = tuple(slices)
    positive[prompt_region] = target[prompt_region].astype(np.float32)
    if positive.sum() == 0:
        positive[tuple(seed_coordinate)] = 1.0
    return positive


def load_foreground_sample(
    h5_file: h5py.File,
    sample_key: str,
    preferred_view: int,
) -> Optional[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Load a sample if it has foreground and an available image view.

    Parameters
    ----------
    h5_file : h5py.File
        Open aggregate H5 file.
    sample_key : str
        Top-level sample key inside the aggregate H5.
    preferred_view : int
        View to try first.

    Returns
    -------
    Optional[Tuple[int, int, np.ndarray, np.ndarray]]
        Tuple `(view, label_id, volume, target)` or `None` for background-only samples.

    Examples
    --------
    >>> result = load_foreground_sample(h5py.File(str(H5_PATH), 'r'), '000000', 0)
    >>> result is None or len(result) == 4
    True

    Notes
    -----
    The first nonzero label id matches the deterministic export requirement.
    """
    group = h5_file[sample_key]
    segmentation = group['seg'][()]
    labels = np.unique(segmentation)
    foreground_labels = labels[labels != 0]
    if len(foreground_labels) == 0:
        return None

    views = [preferred_view]
    fallback_view = 1 if preferred_view == 0 else 0
    if fallback_view not in views:
        views.append(fallback_view)

    label_id = int(foreground_labels[0])
    target = (segmentation == label_id).astype(np.float32)
    for view in views:
        if view >= group['img'].shape[0]:
            continue
        volume = group['img'][view].astype(np.float32) / 255.0
        assert volume.shape == target.shape, (
            f'volume/target shape mismatch for {sample_key}: '
            f'{volume.shape} != {target.shape}'
        )
        return view, label_id, volume, target
    return None


def write_nifti(path: Path, data: np.ndarray) -> None:
    """
    Write one array as a gzip-compressed NIfTI image.

    Parameters
    ----------
    path : Path
        Destination NIfTI path.
    data : np.ndarray
        Array with shape `(D, H, W)`.

    Returns
    -------
    None
        Writes the file at `path`.

    Examples
    --------
    >>> tmp_path = OUTPUT_DIR / 'example.nii.gz'
    >>> write_nifti(tmp_path, np.zeros((2, 2, 2), dtype=np.float32))

    Notes
    -----
    The FreeBrowse debug loader only needs shape-aligned voxel arrays here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), np.eye(4, dtype=np.float32))
    nib.save(image, str(path))


def build_metadata(
    engine: str,
    sample_key: str,
    view: int,
    label_id: int,
    session_dir: Path,
    arrays: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Build FreeBrowse-compatible session metadata with H5 provenance.

    Parameters
    ----------
    engine : str
        H5 source engine name.
    sample_key : str
        Selected aggregate H5 sample key.
    view : int
        Selected image view.
    label_id : int
        Selected nonzero segmentation label id.
    session_dir : Path
        FreeBrowse debug-session directory.
    arrays : Dict[str, np.ndarray]
        Exported arrays keyed by channel name.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata dictionary.

    Examples
    --------
    >>> arrays = {'volume': np.zeros((2, 2, 2), dtype=np.float32)}
    >>> metadata = build_metadata('demo', '000000', 0, 1, OUTPUT_DIR / 'demo.nii', arrays)
    >>> metadata['engine']
    'demo'

    Notes
    -----
    The `volume_name` field is kept because `predict_debug.py` reads it.
    """
    shape = list(arrays['volume'].shape)
    files = {}
    for channel_name, file_name in CHANNEL_FILES.items():
        files[channel_name] = str(Path(PROMPT_ITER_NAME) / file_name)

    return {
        'model_name': 'h5_synth_neel',
        'volume_name': session_dir.name,
        'timestamp': 'deterministic_h5_synth_neel',
        'engine': engine,
        'sample_key': sample_key,
        'view': view,
        'label_id': label_id,
        'source_h5_path': str(H5_PATH),
        'source_config_path': str(CONFIG_PATH),
        'source_index_path': str(INDEX_PATH),
        'shape': shape,
        'input_shape': [1, 5] + shape,
        'channel_names': [
            'volume',
            'previous_pred',
            'positive',
            'negative',
            'unused',
        ],
        'target_name': 'ground_truth',
        'target_voxels': int(arrays['ground_truth'].sum()),
        'positive_voxels': int(arrays['positive'].sum()),
        'negative_voxels': int(arrays['negative'].sum()),
        'files': files,
    }


def write_session(
    engine: str,
    sample_key: str,
    view: int,
    label_id: int,
    volume: np.ndarray,
    target: np.ndarray,
) -> Dict[str, Any]:
    """
    Write one selected H5 sample as a FreeBrowse debug session.

    Parameters
    ----------
    engine : str
        H5 source engine name.
    sample_key : str
        Selected aggregate H5 sample key.
    view : int
        Selected image view.
    label_id : int
        Selected nonzero segmentation label id.
    volume : np.ndarray
        Float volume with shape `(D, H, W)`.
    target : np.ndarray
        Binary target mask with shape `(D, H, W)`.

    Returns
    -------
    Dict[str, Any]
        Export record included in `manifest.json`.

    Examples
    --------
    >>> record = write_session('demo', '000000', 0, 1, np.zeros((2, 2, 2)), np.ones((2, 2, 2)))
    >>> record['engine']
    'demo'

    Notes
    -----
    Channel order matches the existing FreeBrowse prediction script.
    """
    session_dir = OUTPUT_DIR / f'{engine}_{sample_key}_view{view}.nii'
    iteration_dir = session_dir / PROMPT_ITER_NAME
    zeros = np.zeros_like(volume, dtype=np.float32)
    arrays = {
        'volume': volume.astype(np.float32),
        'previous_pred': zeros,
        'positive': create_positive_prompt(target),
        'negative': zeros,
        'unused': zeros,
        'ground_truth': target.astype(np.float32),
    }

    for channel_name, file_name in CHANNEL_FILES.items():
        write_nifti(iteration_dir / file_name, arrays[channel_name])

    metadata = build_metadata(engine, sample_key, view, label_id, session_dir, arrays)
    metadata_path = iteration_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')

    record = {
        'engine': engine,
        'sample_key': sample_key,
        'view': view,
        'label_id': label_id,
        'session_dir': str(session_dir),
        'iteration_dir': str(iteration_dir),
        'shape': metadata['shape'],
        'target_voxels': metadata['target_voxels'],
        'positive_voxels': metadata['positive_voxels'],
        'metadata_path': str(metadata_path),
    }
    return record


def validate_session(record: Dict[str, Any]) -> None:
    """
    Validate NIfTI loadability, shape alignment, and metadata presence.

    Parameters
    ----------
    record : Dict[str, Any]
        Export record from `write_session`.

    Returns
    -------
    None
        Raises an assertion error if validation fails.

    Examples
    --------
    >>> validate_session({'iteration_dir': str(OUTPUT_DIR / 'missing')})
    Traceback (most recent call last):
    ...
    AssertionError: missing metadata...

    Notes
    -----
    This is a file-format check only; it does not run model inference.
    """
    iteration_dir = Path(record['iteration_dir'])
    metadata_path = iteration_dir / 'metadata.json'
    assert metadata_path.exists(), f'missing metadata: {metadata_path}'
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)

    expected_shape = tuple(metadata['shape'])
    for channel_name, file_name in CHANNEL_FILES.items():
        file_path = iteration_dir / file_name
        assert file_path.exists(), f'missing {channel_name}: {file_path}'
        data = nib.load(str(file_path)).get_fdata(dtype=np.float32)
        assert data.shape == expected_shape, (
            f'{file_path} shape {data.shape} != {expected_shape}'
        )
        assert np.isfinite(data).all(), f'{file_path} contains non-finite values'

    positive = nib.load(str(iteration_dir / CHANNEL_FILES['positive'])).get_fdata()
    negative = nib.load(str(iteration_dir / CHANNEL_FILES['negative'])).get_fdata()
    target = nib.load(str(iteration_dir / CHANNEL_FILES['ground_truth'])).get_fdata()
    assert positive.sum() > 0, f'positive prompt is empty: {iteration_dir}'
    assert negative.sum() == 0, f'negative prompt is nonzero: {iteration_dir}'
    assert target.sum() > 0, f'target is empty: {iteration_dir}'


def export_sessions() -> List[Dict[str, Any]]:
    """
    Export one foreground-containing train sample for each requested H5 engine.

    Parameters
    ----------
    None
        Uses absolute paths declared in this script.

    Returns
    -------
    List[Dict[str, Any]]
        Export records for all generated sessions.

    Examples
    --------
    >>> records = export_sessions()
    >>> len(records)
    5

    Notes
    -----
    Selection is deterministic: sorted sample keys, first foreground sample.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_module = load_index_module(INDEX_PATH)
    assignments = index_module.probe_engine_assignments(str(H5_PATH))

    keys_by_engine: Dict[str, List[str]] = {engine: [] for engine in ENGINES}
    for sample_key, engine in assignments.items():
        if engine in keys_by_engine:
            keys_by_engine[engine].append(sample_key)
    for engine in ENGINES:
        keys_by_engine[engine].sort()
        assert keys_by_engine[engine], f'no samples found for engine {engine}'

    records: List[Dict[str, Any]] = []
    with h5py.File(H5_PATH, 'r') as h5_file:
        for engine in ENGINES:
            selected = None
            selected_key = None
            for sample_key in keys_by_engine[engine]:
                selected = load_foreground_sample(h5_file, sample_key, preferred_view=0)
                selected_key = sample_key
                if selected is not None:
                    break
            assert selected is not None, f'no foreground sample found for {engine}'
            assert selected_key is not None, f'no selected key for {engine}'
            view, label_id, volume, target = selected
            records.append(write_session(engine, selected_key, view, label_id, volume, target))

    manifest = {
        'source_h5_path': str(H5_PATH),
        'source_config_path': str(CONFIG_PATH),
        'source_index_path': str(INDEX_PATH),
        'output_dir': str(OUTPUT_DIR),
        'engines': ENGINES,
        'records': records,
    }
    manifest_path = OUTPUT_DIR / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    for record in records:
        validate_session(record)
    return records


if __name__ == '__main__':
    exported_records = export_sessions()
    print(json.dumps(exported_records, indent=2, sort_keys=True))
