# FIERY-FLAME interactive segmentation

COOL-BLOB (epoch 500) is also available in the model selector and uses the same multiclass controls. Its first prediction uses only the current annotations. Later updates pass the previous raw model-grid logits to the trained model, which randomly selects 1,024 prediction tokens using its configured sampler. No hard overlay labels are substituted for probabilities. Open it directly with `?model=cool-blob&tab=aiAnnotation`.

Previous-prediction context is private to each active session and held in memory. Clearing annotations, removing marks (including Undo), changing the prompted class set, switching models or modality, and reopening a session start a fresh context. Saved annotations and overlays still reload normally; the next prediction after reopening starts from the annotations alone. Context also expires with the session cache or a server restart. Class names, colors, and overlay visibility do not reset context.

FIERY-FLAME supports multiple named classes in one annotation session. Choose FIERY-FLAME, create a session, and click or drag on the scan to mark the selected class. Class 1 starts lime green. Add new class creates another independent foreground class; each name and color can be edited. Background is a separate correction tool.

Predictions update 250 ms after a completed gesture or Undo. One prediction runs at a time; further edits replace the pending request with the latest drawing. Drawing remains available while the model is running. Auto-update can be disabled for manual predictions. Clear removes annotations and the displayed prediction. Undo uses NiiVue's whole-gesture drawing history.

Annotations are opaque. Predictions use the same class hue with 45% of the original saturation and 50% opacity by default. Visibility, opacity, and class colors change without rerunning the model. Save & Exit preserves class names, colors, IDs, modality, and annotations. Loading the session restores those annotations and the saved prediction.

## Model and scan contract

The model configuration is `models/fiery-flame/config.yml`. It points to the external FIERY-FLAME training configuration and final `epoch-500.pt` checkpoint; weights are not copied into this repository. The backend environment needs the matching `interseg3d` source and its dependencies, including `voxel`, PyTorch, and MONAI. The complete checkpoint is loaded strictly, including the fine-tuned Anatomix backbone, with compilation disabled.

Select **CT (Hounsfield units)** for raw CT scans. CT intensities are clipped to [-1024, 1024]. **Other modality** clips to the 0th and 99.8th percentiles. Images are reoriented to RAS, resampled to isotropic spacing with the longest physical extent spanning 128 voxels, normalized after resampling, and center padded or cropped to 128³ with the same odd-padding convention as training. Already normalized CT files should not be represented as raw Hounsfield units.

Annotation NIfTI values are 0 for unmarked voxels, 1–254 for stable foreground class IDs, and 255 for explicit background. Every painted voxel is transformed into the model grid; duplicates within a class are removed. Only prompted foreground classes enter the model. Predicted logits are interpolated back to the original scan's affine and dimensions before argmax. Returned label values match the user's class IDs.

Scribbles contribute all their painted voxels. Training used 1–20 clicks per class; more extensive scribbles can increase prediction time, and their quality requires interactive evaluation. This interface does not measure segmentation accuracy.

## Storage and API

Set `ENABLE_AI=true`, `MODELS_DIR`, `AI_DIR`, `DATA_DIR`, and `NIIVUE_BUILD_DIR` using the existing server configuration. `READ_ONLY_DATA=true` blocks writes to the source scan directory while allowing annotation uploads into existing sessions in `AI_DIR`. `/data/ai-sessions` serves that independent session directory. Original scans are not rewritten by inference.

`/ai/model/list` exposes `annotation_mode` (`binary` or `multiclass`). `POST /ai/session/{id}/settings` persists `model_id`, `annotation_mode`, `classes` (`id`, `name`, `color`), and `modality` (`ct` or `other`). The existing upload, `set_annots`, and `infer` endpoints remain in use. The inference request can supply `annotation_mode` and `modality`; incompatible annotation modes are rejected. Binary SP3D and nnInteractive sessions keep their positive/negative controls.

A direct launch URL can select the scan and model with `?vol=/data/scan.nii.gz&model=fiery-flame&tab=aiAnnotation`. The scan must be present in `DATA_DIR`; create a new session in the AI panel after it loads.
