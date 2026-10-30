# Segmento V1 source provenance


The bundled architecture is the dependency closure of `AmixCrossAttentionClickCachedFixedTotalMultiLabel` used by Segmento V1 epoch 500. Sampling, attention equations, backbone module names, and state-dict keys are retained. This is an inference distribution; it does not provide the original training system or unrelated model families.

## Model and checkpoint

- Run timestamp and identifier: `20260929_213420`, `d4baca74734d99ee9f88775e57b2507a`; checkpoint: `epoch-500.pt`.
- Source checkpoint: complete `epoch-500.pt`, including the fine-tuned Anatomix backbone.
- Source checkpoint SHA-256: `782a3440f15b0539be583ec70ce3e852bdfe4b73c04892fbf140780fcef60515`.
- The tracked inference distribution serializes every tensor and original key from the source checkpoint's `model` dictionary into one payload, split into ordinary `weights-*.bin` files smaller than 45 MB. Splitting bytes also supports tensors larger than an individual file. Optimizer and other training state are omitted; model tensors are unchanged.
- `weights.index.json` uses format `segmento-v1-torch-parts-v1` and records the source checkpoint checksum, complete payload checksum, model state-key count, and each part's basename, size, and SHA-256. The loader verifies local paths, sizes, part and payload hashes, duplicate filenames, and the final key count before strict model loading. The verified payload loads once with `weights_only=True`; legacy full checkpoints remain supported.
- `model.yml` contains the original run's model options. `_class` names the bundled class; the obsolete external `anatomix_ckpt` path is empty because the complete checkpoint supplies those weights.
- The runner disables backbone compilation and initializes the architecture without reading separate Anatomix weights, matching the deployed FreeBrowse runner. It then loads every state-dict key strictly, normalizing `anatomix._orig_mod.*` to `anatomix.*`.

## Bundled source

| Bundled file | Source | Source revision | Original file SHA-256 |
| --- | --- | --- | --- |
| `segmento_v1/model.py` | [Multiverseg](https://github.com/halleewong/interseg3d), `interseg3d/models/amix_cross.py` | `84de3df12d52fd52e628a95ac2e84a154d06a26c` | `9c6a8234491bda8c3228e3ccd35e1a8dc37b0a0dff1bf31a5e9e25bbe45637e5` |
| `ChannelLayerNorm` in `segmento_v1/model.py` | Multiverseg, `interseg3d/nn/blocks.py` | same | `8fb51e04a5d367289f2703d9bcc726284e6481afd82d73dd1436aaa9da6ab05f` |
| `segmento_v1/backbone.py` | [Anatomix](https://github.com/neel-dey/prompt-amix), `anatomix/model/network.py` | `c3abd1bb04a556fd3c531e9d7f1836645e0b4490` | `a742c4096ce7eadf34150a681d1c6b10f90c7b6c4aac9d46525e99741fdfe81d` |
| `segmento_v1/sampling.py` | Anatomix, `anatomix/prompt_segmentation/inference_utils.py` | same | `917cf0040f14c356571db6246f5ecb661b15befd2fc0bf0072563386e54f8595` |
| `segmento_v1/inference.py` | FreeBrowse, `backend/src/fiery_flame.py` | integration commit `5928ac4` | `99f19b9aa04f08dca9302624e12a506c9f884da3fd5eebdff9a0686dc5be7a65` |

The model and Anatomix source files were clean at extraction. Complete selected class/function definitions are preserved. The constructor rejects unused ViT and separate pretrained/frozen-backbone loading paths rather than importing external repositories. The unsupported frozen-embedding branch is omitted. The portable runner changes the model import, runner class name, and checkpoint-file loading; preprocessing, prompt mapping, logits, context compatibility, and native-grid restoration are retained.

## Attribution and distribution

Anatomix is MIT licensed, Copyright 2024 Neel Dey; its notice is included in `licenses/ANATOMIX-MIT.txt`. Anatomix's network file credits [CUT](https://github.com/taesungp/contrastive-unpaired-translation); that attribution is retained in `segmento_v1/backbone.py`. Multiverseg's project metadata declares MIT; the checked source snapshot has no standalone LICENSE file, so this package records that declaration without inventing a copyright holder. FreeBrowse's repository LICENSE continues to apply to its integration code.

The source-code license declarations do not independently establish checkpoint redistribution terms. Transfer of the checkpoint follows the model owners' permission; no new checkpoint license or public hosting permission is asserted here.

The Voxel dependency is installed from revision `24cb8b10d698dd1dbce14426080c954a50b27858` of [dalcalab/voxel](https://github.com/dalcalab/voxel). It is not copied into this package.
