"""
Vendored Segmento V1 definitions; see PROVENANCE.md for sources and supported scope.

Sampling and attention equations retain the original implementation and parameter names.
"""
import contextlib
import math
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from segmento_v1.backbone import Unet
from segmento_v1.sampling import pad_to_multiple, crop_from_pad, sample_prompts_in_class

_UNSET = object()
_INHERIT = object()

class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each spatial position.

    Supports:
        2D: [N, C, H, W]
        3D: [N, C, D, H, W]
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, ...] -> [N, ..., C]
        x = x.movedim(1, -1)

        # Normalize across C only.
        x = self.norm(x)

        # [N, ..., C] -> [N, C, ...]
        return x.movedim(-1, 1)


class _HeadReadout(NamedTuple):
    """
    One head's hoisted setup plus its per-chunk readout closure.

    Returned by `_AttentionHeadMixin._head_readout`. `readout` slices its own
    queries, so two heads sharing one chunk loop are driven by the same `sl` —
    see `_attend_multi_fused`:

        readout(sl) -> (V, Qc) logits, or (V, Qc, c_emb) hidden

    `checkpointed_readout` has the same output contract but recomputes one
    chunk during backward. It is used for opt-in, grad-enabled chunked calls so the
    concatenated output does not retain every attention/FFN activation.
    `q` is exposed for chunk sizing: it is the PROJECTED query (post
    `embed_proj`), whose dtype and width are what `_resolve_q_chunk` must measure.
    """

    q: torch.Tensor
    readout: Callable[[Optional[slice]], torch.Tensor]
    checkpointed_readout: Callable[[slice], torch.Tensor]
    mask_active: bool
    V: int
    Q: int
    M: int



class _AmixCrossMixin(nn.Module):
    """Head-agnostic anatomix init + embedding logic.

    Concrete subclasses call `_init_amix(...)` from `__init__`. Binary
    subclasses additionally mix in `_BinarySamplingMixin` for sparse-target /
    context-prompt sampling. The multi-label class defines its own sampling.
    """

    def _init_amix(
        self,
        *,
        anatomix_ckpt: str,
        # Which architecture `anatomix_ckpt` holds. "unet" (the default, and
        # every run before 2026-08-24) is the anatomix U-Net built by
        # `load_inference_checkpoint`. "vit" is the `anatomix-dev-vit`
        # Primus-V2 ViT built by `anatomix.model.vit.load_anatomix_vit`; under
        # it `anatomix_num_downs`, `_ngf`, `_norm`, `_pooling` and `_interp`
        # are INERT (the architecture comes from the anatomix variant
        # registry, not from this config) and the backbone accepts exactly one
        # input shape — see the checks and `_pad_to_shape` below.
        anatomix_backbone: Literal["unet", "vit"] = "unet",
        anatomix_num_downs: int,
        anatomix_ngf: int,
        anatomix_output_nc: int,
        anatomix_norm: str,
        anatomix_pooling: str,
        anatomix_interp: str,
        anatomix_tile_size: Optional[Sequence[int]],
        anatomix_sw_batch_size: int,
        anatomix_sw_overlap: float,
        anatomix_sw_mode: str,
        anatomix_autocast_dtype: Optional[str],
        anatomix_compile: bool,
        anatomix_compile_mode: str,
        freeze_anatomix: bool,
        detach_context_features: bool = False,
        embedding_norm: str = "l2",
        anatomix_random_init: bool = False,
        anatomix_trainable_last_n: Optional[int] = None,
    ):
        # `freeze_anatomix=True` freezes the entire backbone (no grad, excluded
        # from the state_dict, run under inference_mode in `_embed`) so only the
        # attention head (+ channel-norm, if any) trains. It is incompatible
        # with knobs that imply a (partially) trainable backbone:
        if freeze_anatomix:
            if detach_context_features:
                raise ValueError(
                    "freeze_anatomix=True is incompatible with "
                    "detach_context_features=True: detaching the context branch "
                    "is meaningless once anatomix has no gradients at all."
                )
            if anatomix_random_init:
                raise ValueError(
                    "freeze_anatomix=True is incompatible with "
                    "anatomix_random_init: freezing a randomly-initialized "
                    "backbone is a fixed random feature extractor, not a "
                    "trained one."
                )
            if anatomix_trainable_last_n is not None:
                raise ValueError(
                    "freeze_anatomix=True is incompatible with "
                    "anatomix_trainable_last_n (which keeps the last N conv "
                    "layers trainable)."
                )
        if anatomix_backbone not in ("unet", "vit"):
            raise ValueError(
                f"anatomix_backbone must be 'unet' or 'vit'; got "
                f"{anatomix_backbone!r}"
            )
        # The ViT is a fixed-shape, fixed-width feature extractor loaded from
        # one registry entry, so three U-Net-only regimes have no counterpart
        # under it. Reject them instead of ignoring them: a config that asks
        # for one of these and gets a plain fine-tune is a run whose YAML
        # describes an experiment it did not perform.
        if anatomix_backbone == "vit":
            if freeze_anatomix:
                raise ValueError(
                    "anatomix_backbone='vit' does not support "
                    "freeze_anatomix=True: the frozen path delegates to "
                    "anatomix's `forward_embedding`, whose sliding window pads "
                    "to a multiple of 2**(num_downs+1) and would hand the ViT "
                    "an input shape its position embeddings reject. Only the "
                    "trainable path is implemented."
                )
            if anatomix_random_init:
                raise ValueError(
                    "anatomix_backbone='vit' does not support "
                    "anatomix_random_init: the architecture is defined by the "
                    "anatomix variant registry and is only built around a "
                    "checkpoint."
                )
            if anatomix_trainable_last_n is not None:
                raise ValueError(
                    "anatomix_backbone='vit' does not support "
                    "anatomix_trainable_last_n: it counts Conv3d layers in the "
                    "U-Net's flat nn.Sequential, which has no ViT counterpart."
                )
        self.anatomix_backbone = anatomix_backbone
        self.anatomix_num_downs = anatomix_num_downs
        self.anatomix_output_nc = anatomix_output_nc
        self.anatomix_tile_size = (
            tuple(anatomix_tile_size) if anatomix_tile_size is not None else None
        )
        self.anatomix_sw_batch_size = anatomix_sw_batch_size
        self.anatomix_sw_overlap = anatomix_sw_overlap
        self.anatomix_sw_mode = anatomix_sw_mode
        self.anatomix_autocast_dtype = self._parse_torch_dtype(anatomix_autocast_dtype)
        self.freeze_anatomix = freeze_anatomix
        # When True, context (support) volumes are embedded under `torch.no_grad()`
        # (see `_embed(no_grad=...)`) so the context branch contributes no gradient
        # to anatomix — only the query/target branch trains it.
        self.detach_context_features = detach_context_features

        # Non-None only for a backbone that accepts exactly ONE input shape;
        # `_embed` pads to it instead of to a multiple of 2**(num_downs+1).
        self.anatomix_input_shape = None

        assert anatomix_backbone == "unet", "Only the Segmento V1 CNN is bundled"
        assert anatomix_random_init, "Load the full Segmento V1 checkpoint"
        assert not freeze_anatomix, "The bundled checkpoint includes the backbone"
        self.anatomix = Unet(
            dimension=3,
            input_nc=1,
            output_nc=anatomix_output_nc,
            num_downs=anatomix_num_downs,
            ngf=anatomix_ngf,
            norm=anatomix_norm,
            interp=anatomix_interp,
            pooling=anatomix_pooling,
        )
        # load_inference_checkpoint always sets requires_grad=False (a fresh Unet
        # already has it on); re-enable so gradients flow back into anatomix —
        # unless freeze_anatomix=True, in which case the backbone stays frozen
        # (no grad, excluded from state_dict, run under inference_mode in `_embed`).
        if not freeze_anatomix:
            for p in self.anatomix.parameters():
                p.requires_grad_(True)

        # Optional transfer-learning regime: keep only the last N conv layers
        # trainable; the rest of the (pretrained) backbone stays frozen.
        self.anatomix_trainable_last_n = anatomix_trainable_last_n
        if anatomix_trainable_last_n is not None:
            if anatomix_random_init:
                raise ValueError(
                    "anatomix_trainable_last_n is incompatible with "
                    "anatomix_random_init (a from-scratch backbone should train "
                    "all layers)."
                )
            self._set_anatomix_trainable_last_n(anatomix_trainable_last_n)

        # NB a compiled backbone has separate no-grad and grad-enabled graphs
        # whose outputs differ at rounding level — see the "Grad mode changes
        # the bits" section of `_embed`.
        if anatomix_compile:
            self.anatomix = torch.compile(
                self.anatomix, mode=anatomix_compile_mode, dynamic=False
            )

        # Per-voxel embedding normalization applied at the end of `_embed`:
        #   "l2"      — divide each voxel's feature vector by its L2 norm
        #               (unit-length; makes score-head dot products cosines).
        #   "channel" — LayerNorm over the channel axis at each voxel
        #               (z-score per voxel + learnable affine, via
        #               `ChannelLayerNorm`). Only exposed on the attention
        #               models; score heads stay on "l2".
        if embedding_norm not in ("l2", "channel"):
            raise ValueError(
                f"embedding_norm must be 'l2' or 'channel'; got {embedding_norm!r}"
            )
        self.embedding_norm = embedding_norm
        self.channel_norm = (
            ChannelLayerNorm(anatomix_output_nc)
            if embedding_norm == "channel"
            else None
        )

    def _set_anatomix_trainable_last_n(self, n: int):
        """Freeze all anatomix params except the last `n` Conv3d layers.

        The anatomix Unet is a flat `nn.Sequential` in which only `Conv3d`
        modules carry learnable parameters (`InstanceNorm3d` uses
        `affine=False`, `ReLU` has none). `modules()` yields them in forward
        order, so the last `n` are the output conv plus the preceding decoder
        conv(s).
        """
        if n <= 0:
            raise ValueError(f"anatomix_trainable_last_n must be >= 1; got {n}")
        convs = [m for m in self.anatomix.modules() if isinstance(m, nn.Conv3d)]
        if n > len(convs):
            raise ValueError(
                f"anatomix_trainable_last_n={n} exceeds the {len(convs)} conv "
                f"layers in anatomix"
            )
        for p in self.anatomix.parameters():
            p.requires_grad_(False)
        for m in convs[-n:]:
            for p in m.parameters():
                p.requires_grad_(True)
        n_train = sum(p.numel() for p in self.anatomix.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.anatomix.parameters())
        print(
            f"[amix-cross] anatomix: training last {n}/{len(convs)} conv layers "
            f"({n_train:,}/{n_total:,} params)"
        )

    @staticmethod
    def _pad_to_shape(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """Zero-pad the trailing three dims of `x` out to exactly `shape`.

        Pads on the HIGH side only, the same convention `pad_to_multiple` uses,
        so `crop_from_pad(out, original_shape)` undoes it unchanged.

        For a backbone that accepts exactly one input shape (the ViT). It is a
        NO-OP for the amix-cross training configs, whose collate generators
        already `ResizeWithPadOrCropd` every target and context to the cube the
        model trains at — but `_embed` is also called from evaluation paths and
        from configs without those pads, and `2 ** (num_downs + 1)` cannot
        express "exactly 128" (it would leave an axis of 40 at 64, a token grid
        the position embeddings were never built for, and raise deep inside the
        attention blocks). Making the requirement explicit here turns that into
        a correct pad, or a legible error when the volume is too big to pad.
        """
        D, H, W = x.shape[-3:]
        tD, tH, tW = shape
        if D > tD or H > tH or W > tW:
            raise ValueError(
                f"volume {(D, H, W)} exceeds the backbone's fixed input shape "
                f"{tuple(shape)} on at least one axis. This backbone accepts "
                "exactly one shape and this path does NOT tile: use a data "
                "`resolution` whose longest axis fits (res128iso does; res192iso "
                "and res256iso do not)."
            )
        # F.pad order: (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi)
        return F.pad(x, (0, tW - W, 0, tH - H, 0, tD - D))

    @staticmethod
    def _parse_torch_dtype(s: Optional[str]) -> Optional[torch.dtype]:
        if s is None:
            return None
        m = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if s not in m:
            raise ValueError(f"dtype must be one of {list(m)} or null; got {s!r}")
        return m[s]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_anatomix:
            self.anatomix.eval()
        return self

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if self.freeze_anatomix:
            anatomix_prefix = prefix + "anatomix."
            for k in list(sd.keys()):
                if k.startswith(anatomix_prefix):
                    del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.freeze_anatomix:
            filtered = {
                k: v for k, v in state_dict.items() if not k.startswith("anatomix.")
            }
            result = super().load_state_dict(filtered, strict=False, assign=assign)
            real_missing = [k for k in result.missing_keys if not k.startswith("anatomix.")]
            real_unexpected = [
                k for k in result.unexpected_keys if not k.startswith("anatomix.")
            ]
            if strict and (real_missing or real_unexpected):
                errs = []
                if real_missing:
                    errs.append(f"Missing key(s) in state_dict: {real_missing}")
                if real_unexpected:
                    errs.append(f"Unexpected key(s) in state_dict: {real_unexpected}")
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(errs)
                    )
                )
            return torch.nn.modules.module._IncompatibleKeys(real_missing, real_unexpected)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _embed(
        self, vol: torch.Tensor, *, no_grad: bool = False, normalize: bool = True
    ) -> torch.Tensor:
        """`vol`: (C, H, W, D) → flat unit-norm embedding (C_emb, H*W*D).

        Anatomix is single-channel; if C>1 keep only the first channel.

        When `freeze_anatomix=True`, delegates to upstream `forward_embedding`
        wrapped in `torch.inference_mode()` (matches `amix._AmixConcatMixin._embed`).

        When `freeze_anatomix=False`, runs a grad-enabled embedding inline —
        we can't reuse `forward_embedding` because it always wraps in
        `torch.no_grad()`. Limitation: tiled (sliding-window) inference is not
        supported for the trainable path; pass the whole volume in one forward.

        `no_grad=True` runs the trainable path under `torch.no_grad()` so no
        backward graph is built (used for the context branch when
        `detach_context_features=True`). Unlike `torch.inference_mode()`, the
        returned tensor is an ordinary tensor and can still be combined with a
        grad-tracked query embedding downstream — without the `.clone()` the
        inference-mode path needs.

        `normalize=False` returns the RAW backbone features with NO per-voxel
        norm (skips both the channel-norm and the L2 branch). Used by the
        multi-scale model's `multiscale_norm="post"` mode, which embeds both
        streams un-normed, fuses, then applies a single norm on the result.

        Grad mode changes the bits when the backbone is compiled
        ---------------------------------------------------------
        With `anatomix_compile=True` on the trainable path, the SAME volume and
        the SAME weights give a different embedding under `torch.no_grad()`
        than with gradients enabled. `torch.compile` builds one graph for
        no-grad calls and another for grad-enabled calls, and the two round
        differently. Measured 2026-10-01 (torch 2.8, RTX PRO 6000) on values
        of mean magnitude 0.83:

            bf16 autocast   mean |diff| 3.6e-3   max 0.072   (63 % of elements)
            fp32            mean |diff| 9.2e-5   max 0.0021  (>99 % of elements)

        In both precisions that is the same size as compiled vs uncompiled.
        The UNCOMPILED backbone is bit-identical across grad modes, and
        repeated calls in one mode are bit-identical either way.

        Why: the grad-enabled forward must also return every intermediate the
        backward needs, so Inductor fuses it into a different set of kernels
        (this backbone: 111 kernels no-grad vs 165 grad-enabled in bf16, 69 vs
        127 in fp32). Same maths, different grouping and order of the floating
        point operations, hence different rounding. fp32 shrinks the gap ~40x
        but does not close it, so it is not bf16 rounding alone; the long
        instance-norm reductions (mean / variance over ~2M voxels, whose fp32
        result depends on summation order at about this level) are the likely
        remainder — not isolated.

        So a no-grad embedding (validation, viz, the iterative runner's
        rollout round) and a grad-enabled one (a training forward) are equal
        only to rounding, never bitwise. That is harmless for a loss, but it
        is enough to flip an argmax between near-tied classes, so anything
        DISCRETE derived from a prediction — correction clicks drawn from an
        error region, a top-k selection — can differ between the two modes.
        Do not assert bitwise parity across grad modes on a compiled model;
        compare within one mode, or uncompiled. The frozen path is unaffected:
        it always runs without gradients. Evidence and the per-batch table:
        interseg3d-notebooks, `jupyter/universeg3d/2026-09-27_amix-cross-iterative/
        MODELS.md`, arm-6 section.
        """
        x = vol[:1]
        # Trainable path: forward the whole volume in one shot, pad to a
        # multiple of 2**(num_downs+1), normalize, then crop back.
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            x = x.unsqueeze(0)
        original_shape = tuple(x.shape[-3:])
        if self.anatomix_input_shape is not None:
            # Fixed-input-shape backbone (ViT): pad out to that exact shape.
            # Usually a no-op — the collate generators already pad episodes to
            # the cube — see `_pad_to_shape`.
            x_pad = self._pad_to_shape(x.float(), self.anatomix_input_shape)
        else:
            multiple = 2 ** (self.anatomix_num_downs + 1)
            x_pad, _ = pad_to_multiple(x.float(), multiple)

        autocast_enabled = self.anatomix_autocast_dtype is not None
        grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with grad_ctx:
            with torch.autocast(
                device_type=x.device.type,
                dtype=self.anatomix_autocast_dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                emb = self.anatomix(x_pad)
            emb = crop_from_pad(emb, original_shape)
            if normalize:
                if self.embedding_norm == "channel":
                    # LayerNorm over channels; run in fp32 (params are fp32) then
                    # cast back so the autocast-bf16 path keeps its dtype.
                    emb = self.channel_norm(emb.float()).to(emb.dtype)
                else:
                    # Out-of-place L2 normalize (in-place breaks the autograd graph).
                    denom = emb.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    emb = emb / denom
        return emb[0].reshape(emb.shape[1], -1)



def _assert_border_kernel_size(
    name: str,
    border_fraction: float,
    border_kernel_size: Union[int, Sequence[int]],
) -> None:
    """Reject a border-sampling box width of 1 at model setup.

    `class_border_band` (anatomix) finds a class's boundary band by box-blurring
    the *complement* of the class with an `avg_pool3d` of width `kernel_size` and
    keeping in-class voxels where the blur is > 0. At width 1 the pool is the
    identity, so `class_mask & (~class_mask > 0)` is always empty: *no* voxel is
    ever tagged as border. Border-biased sampling then silently degenerates to
    uniform even with `border_fraction > 0`, quietly nullifying the ablation.

    Guard it here so `*_border_kernel_size=1` fails loudly at construction rather
    than silently at sampling time. Inert when `border_fraction == 0` (the kernel
    is never consulted). For a `(min, max)` range, the minimum must be >= 2, since
    any draw of 1 produces the empty-band no-op.
    """
    if border_fraction <= 0:
        return
    widths = (
        (border_kernel_size,)
        if isinstance(border_kernel_size, int)
        else tuple(int(w) for w in border_kernel_size)
    )
    assert all(w >= 2 for w in widths), (
        f"{name}={border_kernel_size} with border_fraction={border_fraction} is a "
        f"no-op: a border box kernel of width 1 selects an empty band (width-1 "
        f"avg_pool3d is the identity), so border-biased sampling silently falls "
        f"back to uniform. Use a width >= 2 (default (3, 7))."
    )



class _AttentionHeadMixin:
    """
    Cross-attention head with shared modules and optional no-gradient optimizations.

    Used by `AmixCrossAttention` (binary, via `_attend`) and
    `AmixCrossAttentionMultiLabel` (via `_attend_multi`).

    The multi-label path's readout topology is selectable via `attn_block`
    (`"plain"` / `"residual"` / `"transformer"`) and its depth via
    `attn_num_layers` — see `_build_attention_modules`. The binary `_attend`
    path deliberately has no such knob and always runs the single-layer
    `"plain"` readout.
    """

    #: Valid `attn_block` topologies — see `_build_attention_modules`.
    ATTN_BLOCKS = ("plain", "residual", "transformer")

    def _build_attention_modules(
        self,
        *,
        c_anat: int,
        attn_num_heads: int,
        attn_dropout: float,
        attn_embed_dim: Optional[int],
        attn_activation: Optional[str],
        attn_block: str = "plain",
        attn_ffn_dim: Optional[int] = None,
        attn_num_layers: int = 1,
    ) -> dict:
        """Construct one cross-attention head's modules (no `self.*` assignment).

        Returns a plain dict keyed ``embed_proj`` / ``attn`` / ``activation`` /
        ``out_proj`` / ``ln_q`` / ``ln_kv`` / ``ln_ffn`` / ``ffn`` / ``ln_out`` /
        ``extra_layers``. The last six are ``None`` unless `attn_block` /
        `attn_num_layers` ask for them. The dict is a return vehicle only —
        callers assign each module to its own ``self.*`` attribute, so
        checkpoint keys stay flat and unchanged.
        Factored out of `_init_attention_head` so a subclass can build a
        second, independent head (e.g. the separate background head) with the
        same construction logic — see `_SeparateBgHeadMixin`.

        `attn_block` selects the readout topology `_attend_multi` runs:

        - ``"plain"`` — ``logit = out_proj(activation(Attn(q, kv, kv)))``. The
          historical head. Builds NO extra modules, so a default-constructed
          model's `state_dict` is byte-identical to a pre-`attn_block` one and
          existing checkpoints load with ``strict=True``.
        - ``"residual"`` — pre-norm + query residual::

              h = q + Attn(ln_q(q), ln_kv(kv), ln_kv(kv))
              logit = out_proj(ln_out(h))

          The query's own features reach the readout additively; under
          ``"plain"`` they enter only through the attention weights, and since
          every K/V token of a class is drawn from that same class the softmax
          sits near uniform and the attention output barely moves with the
          query.
        - ``"transformer"`` — ``"residual"`` plus a position-wise FFN with its
          own residual, i.e. a standard pre-norm block::

              h = h + ffn(ln_ffn(h))

          `attn_ffn_dim` defaults to ``4 * c_emb``. The FFN's second linear is
          zero-initialized so the block starts *exactly* at the ``"residual"``
          function and the FFN is a strict addition at step 0.

        `attn_activation` is the pointwise activation before `out_proj` in
        ``"plain"``; in ``"transformer"`` it becomes the FFN's activation
        instead, so the two never stack. ``"residual"`` has no activation at
        all — it is defined as ``"transformer"`` minus the FFN, which makes
        residual→transformer a clean one-term comparison.

        `attn_num_layers` stacks L blocks of the selected topology on the same
        residual stream, decoder-style: every layer re-attends to the *same*
        K/V (the class's context tokens) and refines the stream::

            h = embed_proj(q)
            for i in range(L):
                h = h + Attn_i(ln_q_i(h), ln_kv_i(kv), ln_kv_i(kv))
                h = h + ffn_i(ln_ffn_i(h))          # "transformer" only
            logit = out_proj(ln_out(h))

        `embed_proj`, the final `ln_out` and `out_proj` are shared across the
        stack — they are the entry projection and the readout, not part of a
        layer. Layer 0 keeps the historical flat attribute names; layers
        ``1..L-1`` live in an ``nn.ModuleList`` of ``nn.ModuleDict``
        (`extra_layers`), so **L=1 builds no ModuleList at all** and a depth-1
        head's `state_dict` is byte-identical to a pre-`attn_num_layers` one.

        Every layer past the first is zero-initialized at *both* of its exits
        (the attention's `out_proj` and, under ``"transformer"``, the FFN's
        second linear), so a depth-L head starts at exactly the depth-1
        function — the same strict-addition property the FFN already has
        against ``"residual"``. A consequence worth knowing: a trained depth-1
        checkpoint loads into a depth-L model with ``strict=False`` and the
        deeper model *initially computes the identical function*, so depth can
        be grown mid-lineage rather than only from scratch.

        L>1 requires a residual topology. ``"plain"`` has no stream to refine —
        its readout consumes the attention output directly — so stacking it
        would mean re-attending to the previous layer's *output*, a different
        model than the one this knob is for.
        """
        if attn_block not in self.ATTN_BLOCKS:
            raise ValueError(
                f"attn_block must be one of {list(self.ATTN_BLOCKS)}; got {attn_block!r}"
            )
        if attn_num_layers < 1:
            raise ValueError(
                f"attn_num_layers must be >= 1; got {attn_num_layers!r}"
            )
        if attn_num_layers > 1 and attn_block == "plain":
            raise ValueError(
                "attn_num_layers > 1 requires attn_block 'residual' or "
                "'transformer'; 'plain' has no residual stream to stack (its "
                "readout consumes the attention output directly). Got "
                f"attn_num_layers={attn_num_layers} with attn_block='plain'."
            )
        c_emb = attn_embed_dim if attn_embed_dim is not None else c_anat
        # Project anatomix features (c_anat) into the attention's working dim
        # (c_emb). Identity when they match — keeps the no-extra-params default
        # so existing checkpoints load unchanged.
        embed_proj = (
            nn.Identity() if c_emb == c_anat else nn.Linear(c_anat, c_emb)
        )
        attn = nn.MultiheadAttention(
            embed_dim=c_emb,
            num_heads=attn_num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        activation = self._build_activation(attn_activation)
        out_proj = nn.Linear(c_emb, 1)
        modules = dict(
            embed_proj=embed_proj,
            attn=attn,
            activation=activation,
            out_proj=out_proj,
            ln_q=None,
            ln_kv=None,
            ln_ffn=None,
            ffn=None,
            ln_out=None,
            extra_layers=None,
        )
        if attn_block == "plain":
            return modules

        modules["ln_q"] = nn.LayerNorm(c_emb)
        modules["ln_kv"] = nn.LayerNorm(c_emb)
        # Pre-norm stacks need a final norm before the readout, otherwise the
        # residual stream's scale is unbounded and `out_proj` sees a moving
        # target. Cheap (2*c_emb params) and standard. Shared by the whole
        # stack — it belongs to the readout, not to a layer.
        modules["ln_out"] = nn.LayerNorm(c_emb)

        ffn_dim = None
        if attn_block == "transformer":
            if attn_activation is None:
                raise ValueError(
                    "attn_block='transformer' needs a non-null attn_activation: "
                    "with Identity the FFN collapses to a single linear layer, "
                    "which the residual stream already provides."
                )
            ffn_dim = 4 * c_emb if attn_ffn_dim is None else attn_ffn_dim
            ffn_out = nn.Linear(ffn_dim, c_emb)
            # Zero-init the FFN's exit so the block starts as the pure
            # "residual" function and the FFN only ever adds to it.
            nn.init.zeros_(ffn_out.weight)
            nn.init.zeros_(ffn_out.bias)
            modules["ln_ffn"] = nn.LayerNorm(c_emb)
            modules["ffn"] = nn.Sequential(
                nn.Linear(c_emb, ffn_dim),
                self._build_activation(attn_activation),
                ffn_out,
            )

        if attn_num_layers == 1:
            return modules

        # Layers 1..L-1: the same block shape as layer 0, kept in a ModuleList
        # of ModuleDicts rather than more flat attributes. Layer 0 staying flat
        # is what keeps a depth-1 state_dict byte-identical to a
        # pre-`attn_num_layers` one.
        extra = nn.ModuleList()
        for _ in range(attn_num_layers - 1):
            layer = nn.ModuleDict()
            layer["ln_q"] = nn.LayerNorm(c_emb)
            layer["ln_kv"] = nn.LayerNorm(c_emb)
            layer_attn = nn.MultiheadAttention(
                embed_dim=c_emb,
                num_heads=attn_num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            # Zero the attention's exit as well as the FFN's (below), so this
            # layer contributes exactly zero at step 0 and depth-L starts at the
            # depth-1 function. Gradients still flow out of zero: the exit's own
            # gradient is (upstream ⊗ attention output), which is nonzero.
            nn.init.zeros_(layer_attn.out_proj.weight)
            nn.init.zeros_(layer_attn.out_proj.bias)
            layer["attn"] = layer_attn
            if attn_block == "transformer":
                layer_ffn_out = nn.Linear(ffn_dim, c_emb)
                nn.init.zeros_(layer_ffn_out.weight)
                nn.init.zeros_(layer_ffn_out.bias)
                layer["ln_ffn"] = nn.LayerNorm(c_emb)
                layer["ffn"] = nn.Sequential(
                    nn.Linear(c_emb, ffn_dim),
                    self._build_activation(attn_activation),
                    layer_ffn_out,
                )
            extra.append(layer)
        modules["extra_layers"] = extra
        return modules

    def _init_attention_head(
        self,
        *,
        c_anat: int,
        attn_num_heads: int,
        attn_dropout: float,
        attn_embed_dim: Optional[int],
        attn_q_chunk_size: Optional[int],
        attn_activation: Optional[str],
        attn_block: str = "plain",
        attn_ffn_dim: Optional[int] = None,
        attn_num_layers: int = 1,
        attn_checkpoint: bool = False,
        attn_reuse_projections: bool = False,
        attn_compile: bool = False,
    ) -> None:
        """
        Initialize attention modules with opt-in checkpointing and no-gradient optimizations.

        Projection reuse and compilation affect only the multi-label no-gradient readout.
        They do not register parameters or change checkpoint keys.
        """
        modules = self._build_attention_modules(
            c_anat=c_anat,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            attn_embed_dim=attn_embed_dim,
            attn_activation=attn_activation,
            attn_block=attn_block,
            attn_ffn_dim=attn_ffn_dim,
            attn_num_layers=attn_num_layers,
        )
        self._assign_head_modules(modules)

        self.attn_num_heads = attn_num_heads
        self.attn_q_chunk_size = attn_q_chunk_size
        self.attn_checkpoint = attn_checkpoint
        self.attn_reuse_projections = attn_reuse_projections
        self.attn_compile = attn_compile
        self.attn_block = attn_block
        self.attn_num_layers = attn_num_layers
        self._fg_head = self._head_bundle(
            modules, attn_block, attn_num_heads, attn_q_chunk_size
        )
        # Stash the construction args so a subclass can rebuild an identical
        # head (the separate-bg-head model inherits these for its bg head).
        self._attn_head_cfg = dict(
            c_anat=c_anat,
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            attn_embed_dim=attn_embed_dim,
            attn_activation=attn_activation,
            attn_block=attn_block,
            attn_ffn_dim=attn_ffn_dim,
            attn_num_layers=attn_num_layers,
        )

    #: role → attribute name for a head built by `_init_attention_head`. A
    #: second head uses the same names under a prefix (`bg_`, `click_`).
    #: ORDER MATTERS: `_assign_head_modules` walks this dict, so it fixes the
    #: registration order of `state_dict` keys and of `model.parameters()` —
    #: and an optimizer `state_dict` is keyed by parameter INDEX, so reordering
    #: it would silently mispair Adam moments when resuming an existing run.
    _HEAD_ATTRS = dict(
        embed_proj="embed_proj",
        attn="attn",
        activation="attn_activation",
        out_proj="out_proj",
        ln_q="attn_ln_q",
        ln_kv="attn_ln_kv",
        ln_out="attn_ln_out",
        ln_ffn="attn_ln_ffn",
        ffn="attn_ffn",
        extra_layers="attn_extra_layers",
    )

    def _assign_head_modules(self, modules: dict, prefix: str = "") -> None:
        """Give one built head's modules their own `self.*` attributes.

        Each module is assigned individually — rather than held only in the
        `_head_bundle` — so checkpoint keys stay flat and historical
        (`embed_proj.weight`, `attn.*`, `out_proj.*`). `None` entries are not
        assigned at all: the block-only norms/FFN are None under ``"plain"``
        and `extra_layers` is None at depth 1, so a default-constructed model's
        `state_dict` is byte-identical to a pre-`attn_block` /
        pre-`attn_num_layers` one.

        `prefix` namespaces a second head (`"bg_"`, `"click_"`).
        """
        for role, attr in self._HEAD_ATTRS.items():
            if modules[role] is not None:
                setattr(self, prefix + attr, modules[role])

    def _inherit_head_cfg(self, *, attn_q_chunk_size=_INHERIT, **overrides) -> dict:
        """Resolve a second head's `_INHERIT` kwargs against the shared head.

        Each entry of `overrides` is either an explicit value or the `_INHERIT`
        sentinel, in which case the shared head's value is copied out of
        `self._attn_head_cfg` — so a second head is independently tunable but
        needs no config at all by default. Returns the kwargs an `_init_*_head`
        forwards to `_build_attention_modules`, plus `c_anat` (never
        overridable — both heads embed the same anatomix features) and
        `attn_q_chunk_size` (which lives on `self`, not in `_attn_head_cfg`).
        """
        cfg = self._attn_head_cfg
        return dict(
            {k: (cfg[k] if v is _INHERIT else v) for k, v in overrides.items()},
            c_anat=cfg["c_anat"],
            attn_q_chunk_size=(
                self.attn_q_chunk_size
                if attn_q_chunk_size is _INHERIT
                else attn_q_chunk_size
            ),
        )

    @classmethod
    def _head_bundle(
        cls,
        modules: dict,
        attn_block: str,
        attn_num_heads: int,
        attn_q_chunk_size: Optional[int],
        prefix: str = "",
    ) -> dict:
        """Pack one head's attribute NAMES + routing knobs into an `_attend_multi` arg.

        Names rather than module references, because a subclass may swap a
        submodule after construction — `AmixCrossAttentionMultiScaleMultiLabel`
        rebuilds `embed_proj` to accept the widened concat input — and a bundle
        of references would go on serving the stale module.
        `_attend_multi` resolves each name with `getattr`.

        The bundle is a plain dict, not a Module, so it adds nothing to
        `state_dict`. Passing one bundle (rather than a module per kwarg) is
        what keeps a second head from silently borrowing the shared head's
        norms/FFN/stack.

        Depth needs no entry of its own: `extra_layers` names the head's
        `nn.ModuleList` (or is ``None`` at depth 1) and its length is the depth.
        """
        return dict(
            {
                role: (prefix + attr) if modules[role] is not None else None
                for role, attr in cls._HEAD_ATTRS.items()
            },
            block=attn_block,
            num_heads=attn_num_heads,
            q_chunk_size=attn_q_chunk_size,
        )

    @staticmethod
    def _build_activation(name: Optional[str]) -> nn.Module:
        if name is None:
            return nn.Identity()
        table = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if name not in table:
            raise ValueError(
                f"attn_activation must be one of {list(table)} or null; got {name!r}"
            )
        return table[name]()

    def _resolve_q_chunk(
        self, Q: int, K: int, device: torch.device, elem_size: int,
        V: int = 1, has_mask: bool = True,
        *, num_heads: Optional[int] = None, q_chunk_size=_UNSET,
    ) -> int:
        """
        Figure out how to chunk attention over the Q query voxels to stay
        within GPU memory limits.

        `V` is the attention batch dim (number of independent K/V sets sharing
        the same Q). For binary `_attend`, V=1. For `_attend_multi`, V is the
        number of valid classes.

        `has_mask` toggles between two cost models:
          - True (default): SDPA falls back to math/mem-efficient because of
            `key_padding_mask`, materializing (V, heads, Qc, K) fp32 scores —
            per-Qc cost scales with K.
          - False: Flash Attention dispatches with no materialized scores —
            per-Qc cost is bounded by qkv/output buffers and is independent
            of K, so much larger chunks are safe.

        `num_heads` / `q_chunk_size` default to the shared head's
        `self.attn_num_heads` / `self.attn_q_chunk_size`. They are overridable
        so a model with a second head (different `num_heads` / chunk size) can
        size its chunks against that head's actual config rather than the
        shared head's.

        ⚠️ Both cost models cover the ATTENTION only, for ONE layer. Under
        `attn_block="transformer"` the FFN adds a `(V, Qc, attn_ffn_dim)`
        activation on top — 4× the residual stream by default — so the
        auto-sized chunk is optimistic for the readout's per-chunk peak. When
        `attn_checkpoint=True`, grad-enabled chunked calls checkpoint each readout
        to avoid retaining those activations across all chunks. Set
        `attn_q_chunk_size` explicitly when the per-chunk peak still matters.
        """
        # A no-grad pass (the dense val forward, the iterative runner's
        # full-volume rollout) keeps no activations past the chunk, so it can
        # afford a far larger one than a grad-enabled pass, which retains or
        # checkpoints every chunk. `attn_q_chunk_size_nograd` (click model;
        # None = no distinction) wins over everything below for such a pass.
        nograd = getattr(self, "attn_q_chunk_size_nograd", None)
        if nograd is not None and not torch.is_grad_enabled():
            return int(nograd)
        q_chunk_size = self.attn_q_chunk_size if q_chunk_size is _UNSET else q_chunk_size
        num_heads = self.attn_num_heads if num_heads is None else num_heads
        if q_chunk_size is not None:
            return q_chunk_size
        if device.type != "cuda":
            return Q
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if has_mask:
            # Math/mem-efficient SDPA: materialize (V, heads, Qc, K) fp32 scores.
            # Peak is fp32 scores + in-place softmax + output buffer ≈
            # V·heads·Qc·K·8 bytes. The ×4 multiplier on elem_size leaves ~2×
            # headroom in bf16, ~4× in fp32.
            per_q = max(1, V * K * num_heads * elem_size * 4)
        else:
            # Flash SDPA: scores are tiled, never fully materialized. Per-Qc
            # cost is dominated by qkv/output buffers (no K factor).
            per_q = max(1, V * num_heads * elem_size * 4)
        chunk = int(free_bytes * 0.8 // per_q)
        return max(4096, min(1048576, chunk))

    def _attend(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: (Q, C_anat), kv: (M, C_anat) → logits (Q,).

        `need_weights=False` lets MHA dispatch into
        `scaled_dot_product_attention`, which avoids materializing the (Q, M)
        attention matrix and picks Flash (head_dim ∈ {8,16,32,64,...}) or
        MemEfficient (any head_dim) when inputs are fp16/bf16. The autocast
        wrapper is what gets bf16 inputs in val — without it SDPA falls back
        to the math backend (fp32) and re-materializes the matrix.
        """
        autocast_dtype = self.anatomix_autocast_dtype
        autocast_enabled = autocast_dtype is not None and q.is_cuda

        with torch.autocast(
            device_type=q.device.type,
            dtype=autocast_dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            q = self.embed_proj(q)
            kv = self.embed_proj(kv)

            Q = q.size(0)
            chunk = self._resolve_q_chunk(Q, kv.size(0), q.device, q.element_size())
            if chunk >= Q:
                attn_out, _ = self.attn(
                    q.unsqueeze(0), kv.unsqueeze(0), kv.unsqueeze(0),
                    need_weights=False,
                )
                attn_out = self.attn_activation(attn_out.squeeze(0))
                return self.out_proj(attn_out).squeeze(-1)

            kv_un = kv.unsqueeze(0)
            outs = []
            for start in range(0, Q, chunk):
                q_chunk = q[start : start + chunk].unsqueeze(0)
                attn_out, _ = self.attn(q_chunk, kv_un, kv_un, need_weights=False)
                attn_out = self.attn_activation(attn_out.squeeze(0))
                outs.append(self.out_proj(attn_out).squeeze(-1))
            return torch.cat(outs, dim=0)

    def _attend_multi(
        self,
        q: torch.Tensor,                # (Q, c_emb_in) — single query set
        kv: torch.Tensor,               # (V, M, c_emb_in) — V K/V sets, fixed M
        key_padding_mask: torch.Tensor,  # (V, M) bool — True where padding
        *,
        head: Optional[dict] = None,
        prior_fn: Optional[Callable[[Optional[slice]], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Vectorized cross-attention: same Q against V K/V sets in one MHA call.

        Returns
        -------
        out : (V, Q) logits

        Steps:
          1. Broadcast q to (V, Q, c_emb).
          2. Single MHA call with key_padding_mask (need_weights=False).
          3. Run the head's readout topology → (V, Q, 1) → squeeze → (V, Q).

        Q is chunked via `_resolve_q_chunk` when large (val path with
        Q = H*W*D). With `attn_checkpoint=True`, grad-enabled chunked calls checkpoint each
        readout so the output does not retain all attention/FFN activations.
        Uses the same autocast wrapper as `_attend`.

        Caller must filter out V slots whose key_padding_mask is all-True —
        MHA produces NaN otherwise.

        `head` is a bundle from `_head_bundle` — the modules plus the `block`
        topology and the `num_heads` / `q_chunk_size` chunk-sizing knobs. It
        defaults to the shared head (`self._fg_head`) and is overridable so a
        model with a second head (the separate background head) can route a
        subset of the classes through that head — see `_SeparateBgHeadMixin`.
        Passing the whole bundle rather than a module per kwarg is deliberate:
        a partial override would let the second head silently reuse the shared
        head's norms and FFN.

        `prior_fn` makes the query PER CLASS: given a q-slice (`None` = all Q)
        it returns the `(V, Qc, c_emb_in)` pre-projection query rows for the V
        classes of this call, and the readout projects and normalises those in
        place of the shared `q` — see `_head_readout`. `None` (the default) is
        the shared-query path, byte for byte.
        """
        head = self._fg_head if head is None else head

        autocast_dtype = self.anatomix_autocast_dtype
        autocast_enabled = autocast_dtype is not None and q.is_cuda

        with torch.autocast(
            device_type=q.device.type,
            dtype=autocast_dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            r = self._head_readout(
                head, q, kv, key_padding_mask, prior_fn=prior_fn
            )

            chunk = self._resolve_q_chunk(
                r.Q, r.M, r.q.device, r.q.element_size(), V=r.V,
                has_mask=r.mask_active,
                num_heads=head["num_heads"], q_chunk_size=head["q_chunk_size"],
            )
            if chunk >= r.Q:
                return r.readout(None)                                       # (V, Q)
            use_checkpoint = self.attn_checkpoint and torch.is_grad_enabled()
            readout = r.checkpointed_readout if use_checkpoint else r.readout
            return torch.cat(                                                # (V, Q)
                [readout(slice(s, s + chunk)) for s in range(0, r.Q, chunk)],
                dim=1,
            )

    @staticmethod
    def _project_attention_bank(
        attention: nn.MultiheadAttention, bank: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project a `(classes, tokens, channels)` bank once for all voxel chunks.

        Returns keys and values shaped `(classes, heads, tokens, channels_per_head)`.
        The caller owns the no-gradient scope and retains these only for one prediction.
        """
        channels = attention.embed_dim
        bias = attention.in_proj_bias
        bank_bias = None if bias is None else bias[channels:]
        projected = F.linear(bank, attention.in_proj_weight[channels:], bank_bias)
        keys, values = projected.chunk(2, dim=-1)
        shape = (*bank.shape[:2], attention.num_heads, channels // attention.num_heads)
        return keys.reshape(shape).transpose(1, 2), values.reshape(shape).transpose(1, 2)

    @staticmethod
    def _attend_projected(
        attention: nn.MultiheadAttention, query: torch.Tensor,
        projected_bank: Tuple[torch.Tensor, torch.Tensor], mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Attend with preprojected queries and keys/values, preserving masking and dropout.

        Query and output have shape `(classes, voxels, channels)`. The Boolean padding
        mask has shape `(classes, tokens)` with true entries excluded from attention.
        """
        classes, voxels, channels = query.shape
        shape = (classes, voxels, attention.num_heads, channels // attention.num_heads)
        query_heads = query.reshape(shape).transpose(1, 2)
        keys, values = projected_bank
        # SDPA's Boolean mask includes positions, the opposite of MHA's padding mask.
        attention_mask = None if mask is None else ~mask[:, None, None, :]
        dropout = attention.dropout if attention.training else 0.0
        attended = F.scaled_dot_product_attention(
            query_heads, keys, values, attn_mask=attention_mask, dropout_p=dropout
        )
        attended = attended.transpose(1, 2).reshape(classes, voxels, channels)
        return attention.out_proj(attended)

    def _head_readout(
        self,
        head: dict,
        q: torch.Tensor,                 # (Q, c_emb_in)
        kv: torch.Tensor,                # (V, M, c_emb_in)
        key_padding_mask: Optional[torch.Tensor],  # (V, M) bool, True where padding
        *,
        hidden: bool = False,
        prior_fn: Optional[Callable[[Optional[slice]], torch.Tensor]] = None,
    ) -> _HeadReadout:
        """
        Hoist one head's loop-invariant setup and return its readout closure.

        Factored out of `_attend_multi` so a caller can drive the SAME readout
        for two heads inside ONE q-chunk loop — see `_attend_multi_fused`. The
        caller owns the `torch.autocast` context and the chunking; this method
        only prepares the head.

        `hidden=True` stops one step short of the final projection, returning
        the readout's `(V, Qc, c_emb)` hidden state instead of `(V, Qc)` logits:
        `activation(attn_out)` under `"plain"`, `ln_out(h)` otherwise. That is
        the representation a joint readout over two heads concatenates. A
        bundle whose `out_proj` is None (a head built for hidden-only use) is
        legal only with `hidden=True`.

        `prior_fn` (default None) switches the QUERY from shared to per class.
        It maps a q-slice (`None` = all Q) to `(V, Qc, c_emb_in)` pre-projection
        rows — one query set per K/V set of this call — which the chunk then
        runs through `embed_proj` (and `ln_q`) itself, in place of the shared,
        hoisted `q`. The per-class rows are built lazily per chunk because a
        materialised `(V, Q, c_emb_in)` is V× the query at val (Q = D·H·W).
        `q` is still required: it sizes the chunks (`_resolve_q_chunk` reads
        the projected shared query) and it is what runs when `prior_fn` is
        None. Used by the click model's `query_prior` path, where the rows are
        the target features with the query's type slot holding that class's
        previous probability (`AmixCrossAttentionClickFixedTotalMultiLabel`).

        With gradients disabled, `attn_reuse_projections` shares first-layer query
        projections across classes and every layer's bank projections across chunks.
        Class-specific queries from `prior_fn` retain their separate projections.
        `attn_compile` compiles the complete chunk readout, not the image encoder or
        the gradient-enabled training pass. Compiler graphs are cached by PyTorch,
        while features and projected banks remain local to this prediction.
        """

        def _mod(role):
            name = head[role]
            return None if name is None else getattr(self, name)

        embed_proj = _mod("embed_proj")
        attn = cast(nn.MultiheadAttention, _mod("attn"))
        activation = _mod("activation")
        out_proj = _mod("out_proj")
        block = head["block"]
        # None at depth 1; otherwise the ModuleList of layers 1..L-1, whose
        # length IS the extra depth (nothing else needs to track it).
        extra_layers = _mod("extra_layers")
        reuse_projections = self.attn_reuse_projections and not torch.is_grad_enabled()

        if out_proj is None and not hidden:
            raise ValueError(
                "_head_readout was asked for logits from a head bundle with no "
                "`out_proj`. Such a head is built for hidden-state fusion only "
                "(see AmixCrossAttentionSeparateClickHeadFixedTotalMultiLabel "
                "under click_fusion="
                "'hidden_concat'); call it with hidden=True."
            )

        q = embed_proj(q)                 # (Q, c_emb)
        kv = embed_proj(kv)               # (V, M, c_emb)

        V = kv.size(0)
        Q = q.size(0)
        M = kv.size(1)

        # Drop the key_padding_mask when no position is actually padded;
        # without a mask, SDPA can dispatch into Flash Attention (10-50×
        # faster than the math/mem-efficient path on H100 / Ampere).
        mask_active = key_padding_mask is not None and bool(key_padding_mask.any())
        if not mask_active:
            key_padding_mask = None

        # Pre-norm the query and K/V ONCE, outside the chunk loop: both are
        # position-wise, so normalizing before the (V, Qc, c_emb) broadcast
        # costs V× less memory than normalizing the expanded view, and the
        # K/V norm is loop-invariant. Note `ln_kv` maps padded (all-zero)
        # positions to its bias rather than zero — harmless, because those
        # positions are exactly the ones `key_padding_mask` excludes.
        if block == "plain":
            q_attn, kv_attn = q, kv
        else:
            ln_ffn, ffn, ln_out = _mod("ln_ffn"), _mod("ffn"), _mod("ln_out")
            q_attn, kv_attn = _mod("ln_q")(q), _mod("ln_kv")(kv)
        extra = []
        # Bank features are shared across chunks; later-layer queries are class-dependent.
        if extra_layers is not None:
            for layer in extra_layers:
                layer_bank = layer["ln_kv"](kv)
                projected = None
                if reuse_projections:
                    projected = self._project_attention_bank(layer["attn"], layer_bank)
                extra.append((layer, layer_bank, projected))

        projected_bank = None
        if reuse_projections:
            if prior_fn is None:
                channels = attn.embed_dim
                bias = None if attn.in_proj_bias is None else attn.in_proj_bias[:channels]
                q_attn = F.linear(q_attn, attn.in_proj_weight[:channels], bias)
            projected_bank = self._project_attention_bank(attn, kv_attn)

        def _readout_chunk(
            q_raw: torch.Tensor,
            q_norm: torch.Tensor,
            q_cls: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Compute one already-sliced query chunk for this attention head.

            `q_cls` — `(V, Qc, c_emb_in)` per-class pre-projection rows from
            `prior_fn`, or None for the shared query. With it, the projection
            and the query norm run here, per class, and the residual stream
            is the per-class projected query.
            """
            Qc = q_raw.size(0)
            if q_cls is None:
                q_b = q_norm.unsqueeze(0).expand(V, Qc, q_norm.size(-1))
                q_res = q_raw.unsqueeze(0).expand(V, Qc, q_raw.size(-1))
            else:
                q_res = embed_proj(q_cls)
                q_b = q_res if block == "plain" else _mod("ln_q")(q_res)
                if projected_bank is not None:
                    # Previous-prediction features differ by class and cannot share queries.
                    channels = attn.embed_dim
                    bias = None if attn.in_proj_bias is None else attn.in_proj_bias[:channels]
                    q_b = F.linear(q_b, attn.in_proj_weight[:channels], bias)
            if projected_bank is not None:
                attn_out = self._attend_projected(attn, q_b, projected_bank, key_padding_mask)
            else:
                attn_out, _ = attn(
                    q_b, kv_attn, kv_attn, key_padding_mask=key_padding_mask, need_weights=False
                )
            if block == "plain":
                h = activation(attn_out)
                return h if hidden else out_proj(h).squeeze(-1)
            # Residual against the RAW (un-normed) query slice — pre-norm
            # convention: the norm feeds the branch, not the stream.
            h = q_res + attn_out
            if block == "transformer":
                h = h + ffn(ln_ffn(h))
            # Layers 1..L-1 refine the same stream, re-attending to the same
            # (already normed) K/V.
            for lyr, lyr_kv, layer_bank in extra:
                layer_query = lyr["ln_q"](h)
                if layer_bank is not None:
                    attention = lyr["attn"]
                    channels = attention.embed_dim
                    bias = attention.in_proj_bias
                    query_bias = None if bias is None else bias[:channels]
                    # Later-layer queries depend on the class-specific residual stream.
                    layer_query = F.linear(
                        layer_query, attention.in_proj_weight[:channels], query_bias
                    )
                    lyr_out = self._attend_projected(
                        attention, layer_query, layer_bank, key_padding_mask
                    )
                else:
                    lyr_out, _ = lyr["attn"](
                        layer_query, lyr_kv, lyr_kv,
                        key_padding_mask=key_padding_mask, need_weights=False
                    )
                h = h + lyr_out
                if block == "transformer":
                    h = h + lyr["ffn"](lyr["ln_ffn"](h))
            h = ln_out(h)
            return h if hidden else out_proj(h).squeeze(-1)

        if self.attn_compile and not torch.is_grad_enabled():
            _readout_chunk = torch.compile(_readout_chunk, fullgraph=True, dynamic=True)

        def _readout(sl: Optional[slice]) -> torch.Tensor:
            """
            One q-slice (`None` = all Q) → logits or hidden state.
            """
            q_raw, q_norm = (q, q_attn) if sl is None else (q[sl], q_attn[sl])
            q_cls = None if prior_fn is None else prior_fn(sl)
            return _readout_chunk(q_raw, q_norm, q_cls)

        def _checkpointed_readout(sl: slice) -> torch.Tensor:
            """
            Recompute one q-chunk during backward to release its activations.
            """
            q_cls = None if prior_fn is None else prior_fn(sl)
            return checkpoint(
                _readout_chunk, q[sl], q_attn[sl], q_cls, use_reentrant=False
            )

        return _HeadReadout(
            q=q, readout=_readout, checkpointed_readout=_checkpointed_readout,
            mask_active=mask_active, V=V, Q=Q, M=M,
        )

    def _attend_multi_fused(
        self,
        q: torch.Tensor,                        # (Q, c_emb_in) — single query set
        *,
        kv_a: Optional[torch.Tensor],           # (V, M_a, c_emb_in) or None
        mask_a: Optional[torch.Tensor],         # (V, M_a) bool
        head_a: dict,
        kv_b: Optional[torch.Tensor],           # (V, M_b, c_emb_in) or None
        mask_b: Optional[torch.Tensor],         # (V, M_b) bool
        head_b: dict,
        fuse_proj: nn.Linear,                   # Linear(c_emb_a + c_emb_b, 1)
    ) -> torch.Tensor:
        """Two heads over two K/V banks, hidden states concatenated, one readout.

        Returns (V, Q) logits, where V is the shared class-batch dim of the two
        banks. Both heads see the SAME queries; they differ only in what they
        attend to (`kv_a` = context prompt tokens, `kv_b` = click tokens).

        Either bank may be `None` — that source is absent for these classes —
        in which case its half of the concatenation is **zeros**, not a skip, so
        `fuse_proj`'s bias and the other head's half still apply. Both `None` is
        a caller bug: such a class has no evidence at all and must be left at
        its 0 logit, as the parent does.

        The concatenation happens INSIDE the q-chunk loop, by necessity: a full
        `(V, Q, c_emb)` hidden is ~9.6 GB at V=36, Q=128³, c_emb=64, bf16, so
        the hidden states can never be materialized across all of Q.

        Chunk size is the smaller of the two heads' `_resolve_q_chunk` results,
        halved — two `c_emb`-wide hiddens are alive at once where `_attend_multi`
        holds one `(V, Qc)` scalar map. The halving applies only when BOTH heads
        auto-size (`q_chunk_size is None`); an explicitly configured chunk is
        honored verbatim, as it is everywhere else. `_resolve_q_chunk`'s cost
        model covers attention only, so configs should set `attn_q_chunk_size`
        explicitly here — see its docstring.
        """
        if kv_a is None and kv_b is None:
            raise ValueError(
                "_attend_multi_fused needs at least one K/V bank; a class with "
                "neither context nor click tokens carries no evidence and must "
                "be left at its 0 logit by the caller."
            )

        autocast_dtype = self.anatomix_autocast_dtype
        autocast_enabled = autocast_dtype is not None and q.is_cuda

        with torch.autocast(
            device_type=q.device.type,
            dtype=autocast_dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            r_a = (
                None if kv_a is None
                else self._head_readout(head_a, q, kv_a, mask_a, hidden=True)
            )
            r_b = (
                None if kv_b is None
                else self._head_readout(head_b, q, kv_b, mask_b, hidden=True)
            )

            present = [(r, h) for r, h in ((r_a, head_a), (r_b, head_b)) if r is not None]
            Q = present[0][0].Q

            chunk = min(
                self._resolve_q_chunk(
                    r.Q, r.M, r.q.device, r.q.element_size(), V=r.V,
                    has_mask=r.mask_active,
                    num_heads=h["num_heads"], q_chunk_size=h["q_chunk_size"],
                )
                for r, h in present
            )
            if all(h["q_chunk_size"] is None for _, h in present):
                chunk = max(1, chunk // 2)

            def _fused(sl: Optional[slice]) -> torch.Tensor:
                """(V, Qc) logits for one q-slice (or all of Q when `sl` is None)."""
                halves = [None if r is None else r.readout(sl) for r in (r_a, r_b)]
                # An absent source contributes a zero vector of ITS OWN width.
                # The two heads may carry different `attn_embed_dim`s, so that
                # width is not the present half's — it is whatever `fuse_proj`
                # expects minus the present half, which also pins the two to
                # agree with the readout they feed.
                ref = next(h for h in halves if h is not None)
                halves = [
                    h if h is not None
                    else ref.new_zeros(
                        ref.shape[:-1] + (fuse_proj.in_features - ref.size(-1),)
                    )
                    for h in halves
                ]
                return fuse_proj(torch.cat(halves, dim=-1)).squeeze(-1)

            if chunk >= Q:
                return _fused(None)                                          # (V, Q)
            return torch.cat(                                                # (V, Q)
                [_fused(slice(s, s + chunk)) for s in range(0, Q, chunk)],
                dim=1,
            )



class _MultiLabelSamplingMixin:
    """Per-class multi-label target & context prompt sampling.

    Used by `AmixCrossAttentionMultiLabel` and `AmixCrossScoreMultiLabel`.
    Owns the runtime-set `_batch_labels` state (the experiment runner sets
    this before each forward) and provides:

      * `_validate_labels` — strict contiguous-from-0 check.
      * `_sample_target_indices` — per-class sparse-target sampling for the
        binary or score loss; falls back to class 0 when a class is absent
        from the target crop.
      * `_sample_context_prompts_multilabel` — fixed-shape per-(class, context)
        K/V embedding bank with a padding mask.

    Depends on `_embed` from `_AmixCrossMixin`.
    """

    # When True, the context K/V bank uses a FIXED TOTAL budget per class
    # (`num_context_voxels_total_per_class`) spread evenly over only the
    # contexts that contain the class, so the bank is fully populated (no
    # wasted padding) — see `AmixCrossScoreFixedTotalMultiLabel`. When False
    # (default), the legacy per-(class, context) sampling is used and the bank
    # is `(C, N*b)` with padding for absent (class, context) pairs.
    fixed_total_context_voxels: bool = False
    # Total context voxels per class when `fixed_total_context_voxels=True`.
    # None in the legacy mode (the per-(class, context) budget
    # `num_context_voxels_per_class` is used instead).
    num_context_voxels_total_per_class: Optional[int] = None

    def _init_multilabel_sampling(
        self,
        *,
        num_target_voxels_per_class: int,
        num_context_voxels_per_class: int,
        context_border_fraction: float = 0.0,
        context_border_kernel_size: Union[int, Sequence[int]] = (3, 7),
        target_border_fraction: float = 0.0,
        target_border_kernel_size: Union[int, Sequence[int]] = (3, 7),
        target_border_bg_only: bool = False,
        target_sampling: Literal[
            "stratified", "uniform", "sqrt", "proportional"
        ] = "stratified",
        target_sampling_min_voxels_per_class: int = 1,
    ):
        self.num_target_voxels_per_class = num_target_voxels_per_class
        self.num_context_voxels_per_class = num_context_voxels_per_class
        # Border-biased context sampling: draw this fraction of each class's
        # context prompts from its boundary band, computed with a
        # RandBorderClick-style box kernel of width `context_border_kernel_size`
        # (int, or a (min, max) range sampled per call). 0.0 = uniform (legacy).
        # Applies to context prompts only.
        self.context_border_fraction = context_border_fraction
        self.context_border_kernel_size = context_border_kernel_size
        # Target (sparse-query) sampling controls — the analog of the context
        # border knobs above, plus a stratify/uniform mode switch:
        #   target_sampling="stratified" (default) — the legacy behavior:
        #     `_sample_target_indices` draws exactly `num_target_voxels_per_class`
        #     voxels per class (K = a * num_classes). `target_border_fraction`
        #     (>0) then draws that fraction of each class's quota from its border
        #     band (same box-kernel semantics as `context_border_*`); 0.0 =
        #     uniform-within-class (legacy).
        #   target_sampling="uniform" — draw all K = a * num_classes target
        #     voxels uniformly over the whole volume, ignoring class (no
        #     per-class quota). The border knobs are inert in this mode.
        #   target_sampling="sqrt" — keep the same total budget
        #     K = a * num_classes but split it across only the classes PRESENT
        #     in the crop, proportional to sqrt(voxel count), with a floor of
        #     `target_sampling_min_voxels_per_class` query voxels per present
        #     class. Absent classes get 0 (no bg fallback). The border knobs
        #     apply within each class exactly as in stratified mode (0.0 =
        #     uniform-within-class).
        #   target_sampling="proportional" — identical to "sqrt" except the
        #     per-present-class split is proportional to the raw voxel count
        #     (linear weight, not its square root), so large classes take a
        #     larger share than under sqrt but still bounded by the same total
        #     budget K and the same per-present-class floor.
        # `target_sampling_min_voxels_per_class` (default 1) is the per-present-
        # class floor for the "sqrt" and "proportional" modes: every class
        # present in the crop is guaranteed at least this many query voxels
        # before the remaining budget is apportioned by weight. Inert for
        # "stratified" (fixed per-class quota `a`) and "uniform" (no quota).
        # Requires a >= this floor so the floors fit within K = a * num_classes.
        # `target_border_bg_only` (default False) restricts border-biased
        # within-class sampling to the background class (channel 0): when True,
        # only class 0 draws `target_border_fraction` of its quota from its
        # boundary band, while every foreground class is sampled uniformly
        # within-class. Inert when `target_border_fraction == 0` or under
        # "uniform" sampling. Applies to "stratified", "sqrt" and "proportional".
        # Defaults reproduce the legacy uniform-stratified target sampling.
        if target_sampling not in (
            "stratified", "uniform", "sqrt", "proportional"
        ):
            raise ValueError(
                "target_sampling must be 'stratified', 'uniform', 'sqrt' or "
                f"'proportional'; got {target_sampling!r}"
            )
        if target_sampling_min_voxels_per_class < 1:
            raise ValueError(
                "target_sampling_min_voxels_per_class must be >= 1; got "
                f"{target_sampling_min_voxels_per_class}"
            )
        if (
            target_sampling in ("sqrt", "proportional")
            and target_sampling_min_voxels_per_class > num_target_voxels_per_class
        ):
            raise ValueError(
                "target_sampling_min_voxels_per_class "
                f"({target_sampling_min_voxels_per_class}) must be <= "
                f"num_target_voxels_per_class ({num_target_voxels_per_class}) so "
                "the per-present-class floor fits within the K = a * num_classes "
                "budget."
            )
        self.target_border_fraction = target_border_fraction
        self.target_border_kernel_size = target_border_kernel_size
        self.target_border_bg_only = target_border_bg_only
        self.target_sampling = target_sampling
        self.target_sampling_min_voxels_per_class = (
            target_sampling_min_voxels_per_class
        )
        _assert_border_kernel_size(
            "context_border_kernel_size",
            context_border_fraction,
            context_border_kernel_size,
        )
        # Target border knobs are inert under uniform sampling (no per-class
        # band); they apply within each class for stratified/sqrt/proportional
        # (or within background only when target_border_bg_only=True).
        if target_sampling in ("stratified", "sqrt", "proportional"):
            _assert_border_kernel_size(
                "target_border_kernel_size",
                target_border_fraction,
                target_border_kernel_size,
            )
        # Runtime state: set by the experiment runner before each batch.
        # 1-D long tensor of class IDs for the current batch.
        self._batch_labels: Optional[torch.Tensor] = None

    @staticmethod
    def _validate_labels(labels: torch.Tensor) -> None:
        """Asserts contiguous-from-0 (v1 requirement) and that bg (0) is included."""
        if labels.ndim != 1:
            raise ValueError(
                f"labels must be 1-D, got shape {tuple(labels.shape)}"
            )
        if labels.numel() == 0:
            raise ValueError("labels must be non-empty")
        if int(labels[0].item()) != 0:
            raise ValueError(
                f"labels[0] must be 0 (background); got {int(labels[0].item())}"
            )
        expected = torch.arange(
            labels.numel(), device=labels.device, dtype=labels.dtype
        )
        if not torch.equal(labels, expected):
            raise ValueError(
                f"labels must be contiguous-from-0 (v1 requirement); got {labels.tolist()}"
            )

    @torch.no_grad()
    def _sample_target_indices(self, y_target_b: torch.Tensor) -> torch.Tensor:
        """Choose which target voxels this element will predict at (multi-label).

        Returns one row of the `sparse_indices` tensor handed back to `forward`
        — the subset of target voxels that become attention/score queries and
        the only positions the loss sees. See the module docstring's "Sparse
        target prediction" section; the only multi-label difference is that the
        budget is spread over classes rather than fg/bg.

        The `target_sampling` mode (set at init) selects one of four schemes,
        all of which return exactly K = a * len(self._batch_labels) flat indices
        so per-element results stack cleanly into the runner's (B, K) tensor:

          * "stratified" (default) — per-class quota of `a`. For each class `c`:
              - If present, sample `a` indices from class c.
              - Elif class 0 (background) is present, sample `a` from class 0
                (background fallback for absent classes).
              - Else sample `a` uniformly over all voxels.
          * "uniform" — draw all K indices uniformly over the whole volume,
            ignoring class.
          * "sqrt" — split the fixed budget K across the classes PRESENT in the
            crop, proportional to sqrt(voxel count), floor
            `target_sampling_min_voxels_per_class` per present class; absent
            classes get 0 (see `_distribute_weighted_budget`).
          * "proportional" — as "sqrt" but the split is proportional to the raw
            voxel count (linear weight), so large classes take a bigger share.

        Within-class draws honour `target_border_fraction` /
        `target_border_kernel_size` (biasing toward the class boundary band);
        `target_border_bg_only=True` restricts that bias to the background class
        (channel 0), sampling every foreground class uniformly within-class.

        True per-voxel labels are recovered downstream by the runner's
        `_gather_sparse_y` (which gathers `y_target` at these flat indices), so
        only the total K — not the per-class position — matters for the loss.
        """
        if self._batch_labels is None:
            raise RuntimeError(
                f"{type(self).__name__}._batch_labels is None — the "
                "experiment runner must set it before calling "
                "_sample_target_indices."
            )

        seg = y_target_b.to(torch.long)
        a = self.num_target_voxels_per_class
        labels = self._batch_labels

        # Non-stratified mode: draw all K = a * len(labels) target voxels
        # uniformly over the whole volume, ignoring class. True per-voxel labels
        # are recovered downstream by the runner's `_gather_sparse_y` (it gathers
        # `y_target` at these flat indices), so no per-class quota is needed and
        # the K = a * len(labels) contract is preserved for a fair comparison
        # against the stratified baseline. The border knobs are inert here.
        if self.target_sampling == "uniform":
            k_total = a * int(labels.numel())
            return torch.randint(
                seg.numel(), (k_total,), device=seg.device, dtype=torch.long
            )

        # Weighted-proportional stratified modes ("sqrt" / "proportional"): keep
        # the same total budget K = a * len(labels) but split it across only the
        # classes PRESENT in this crop, proportional to voxel count raised to a
        # power (0.5 for "sqrt", 1.0 for "proportional"), with a floor of
        # `target_sampling_min_voxels_per_class` per present class. Absent
        # classes get 0 (no bg fallback, unlike the per-class quota above).
        # Within each class, samples are drawn with the same border knobs as
        # stratified (0.0 = uniform-within-class; bg-only when configured). The K
        # contract is preserved and true per-voxel labels are recovered
        # downstream, so the variable per-class split is safe.
        if self.target_sampling in ("sqrt", "proportional"):
            k_total = a * int(labels.numel())
            present: List[int] = []
            counts: List[int] = []
            for c in labels.tolist():
                n_c = int((seg == c).sum().item())
                if n_c > 0:
                    present.append(c)
                    counts.append(n_c)
            if not present:
                # Pathological crop with no labeled class present — sample
                # uniformly over all voxels rather than crash.
                return torch.randint(
                    seg.numel(), (k_total,), device=seg.device, dtype=torch.long
                )
            power = 0.5 if self.target_sampling == "sqrt" else 1.0
            budgets = self._distribute_weighted_budget(
                counts,
                k_total,
                power=power,
                min_per_class=self.target_sampling_min_voxels_per_class,
            )
            chunks: List[torch.Tensor] = []
            for c, k_c in zip(present, budgets):
                if k_c <= 0:
                    continue
                bf, bk = self._target_border_params(c)
                chunks.append(
                    sample_prompts_in_class(
                        seg, c, k_c,
                        border_fraction=bf,
                        border_kernel_size=bk,
                    )
                )
            return torch.cat(chunks, dim=0)

        # Whether class 0 (background) exists in this crop. When it does, it is
        # the fallback source for any class absent from the crop. When it does
        # NOT (an all-foreground crop, e.g. a fixed 128**3 cube centered inside
        # a large organ), there is no background to fall back to — every voxel
        # belongs to some present class, so we sample uniformly over all voxels
        # instead of raising, keeping the K = a * len(labels) contract intact.
        bg_present = bool((seg == 0).any().item())

        chunks: List[torch.Tensor] = []
        for c in labels.tolist():
            present = (seg == c).any()
            if bool(present.item()):
                bf, bk = self._target_border_params(c)
                chunks.append(
                    sample_prompts_in_class(
                        seg, c, a,
                        border_fraction=bf,
                        border_kernel_size=bk,
                    )
                )
            elif bg_present:
                # bg fallback for absent classes: drawn from class 0, so use the
                # background border params.
                bf, bk = self._target_border_params(0)
                chunks.append(
                    sample_prompts_in_class(
                        seg, 0, a,
                        border_fraction=bf,
                        border_kernel_size=bk,
                    )
                )
            else:
                # Pathological all-foreground crop: class 0 is itself absent.
                # Sample `a` flat indices uniformly over every voxel (all of
                # which are foreground here) rather than crash.
                chunks.append(
                    torch.randint(
                        seg.numel(), (a,), device=seg.device, dtype=torch.long
                    )
                )
        return torch.cat(chunks, dim=0)

    def _target_border_params(self, c: int):
        """Border (fraction, kernel_size) to use when drawing class `c`.

        Normally every class uses the configured `target_border_fraction` /
        `target_border_kernel_size`. When `target_border_bg_only=True`, only the
        background class (channel 0) gets the border bias; every foreground
        class is sampled uniformly within-class (fraction 0.0). The kernel is
        returned unchanged (it is inert when the fraction is 0.0).
        """
        if self.target_border_bg_only and c != 0:
            return 0.0, self.target_border_kernel_size
        return self.target_border_fraction, self.target_border_kernel_size

    @staticmethod
    def _distribute_weighted_budget(
        counts: List[int],
        total: int,
        power: float = 0.5,
        min_per_class: int = 1,
    ) -> List[int]:
        """Split `total` target voxels over present classes ∝ count**`power`.

        `counts` are the per-present-class voxel counts (all > 0), in the order
        the caller emits chunks. Returns a same-length list of non-negative
        ints summing exactly to `total`, with a floor of `min_per_class` per
        class whenever the budget allows it. `power` sets how the remaining
        budget is weighted: 0.5 = sqrt (dampens large classes, e.g.
        background), 1.0 = raw count-proportional. The floor guarantees every
        present class contributes at least `min_per_class` query voxels; the
        pool left after the floors is apportioned by the largest-remainder
        method, so the result is deterministic (no RNG).
        """
        n = len(counts)
        if n == 0:
            return []
        m = max(1, int(min_per_class))
        # Not enough budget to floor every class at `m`: hand `m` voxels at a
        # time to the largest classes until the budget runs out. Cannot happen
        # when num_target_voxels_per_class >= min_per_class (guaranteed at init,
        # since total = a * num_classes >= m * n), but handled so the exact-sum
        # contract holds regardless.
        if total < n * m:
            order = sorted(range(n), key=lambda i: counts[i], reverse=True)
            alloc = [0] * n
            rem = total
            for i in order:
                take = min(m, rem)
                alloc[i] = take
                rem -= take
                if rem <= 0:
                    break
            return alloc

        weights = [c ** power for c in counts]
        wsum = sum(weights)
        pool = total - n * m                   # remainder after the per-class floors
        raw = [pool * w / wsum for w in weights]
        alloc = [m + int(r) for r in raw]      # floor(raw) + the per-class minimum
        leftover = total - sum(alloc)          # 0 <= leftover < n
        # Hand the leftover to the largest fractional remainders.
        frac_order = sorted(
            range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True
        )
        for j in range(leftover):
            alloc[frac_order[j]] += 1
        return alloc

    @classmethod
    def _distribute_sqrt_budget(cls, counts: List[int], total: int) -> List[int]:
        """Back-compat alias: sqrt weighting with a floor of 1 per class.

        Retained for callers/tests that predate `_distribute_weighted_budget`.
        """
        return cls._distribute_weighted_budget(
            counts, total, power=0.5, min_per_class=1
        )

    @staticmethod
    def _distribute_context_budget(num_present: int, total: int) -> List[int]:
        """Split `total` context voxels evenly over `num_present` contexts.

        Returns a list of per-(present-context) counts; the first
        `total % num_present` contexts get one extra so the counts sum exactly
        to `total`. Empty list when `num_present == 0`.
        """
        if num_present <= 0:
            return []
        base, rem = divmod(total, num_present)
        return [base + (1 if j < rem else 0) for j in range(num_present)]

    def _sample_context_prompts_multilabel(
        self,
        support_vols: torch.Tensor,    # (N, H, W, D)
        support_segs: torch.Tensor,    # (N, H, W, D), uint8/long
        labels: torch.Tensor,          # (C,) long
    ):
        """
        Sample per-class context voxel embeddings with fixed shape.

        An explicitly installed `evaluation_context_sampler` supplies evaluation-only
        banks before the ordinary training/default sampling path.

        Legacy mode (`fixed_total_context_voxels=False`)
        ------------------------------------------------
        cls_kvs : (C, N*b, c_emb)
            For class c, context n: positions [n*b : (n+1)*b] contain either
            real embeddings (if c is present in context n) or zero
            embeddings (if c is absent from n).
        cls_mask : (C, N*b) bool
            True where padding (class c absent from context n), False where
            real K/V.

        Fixed-total mode (`fixed_total_context_voxels=True`)
        ----------------------------------------------------
        cls_kvs : (C, B, c_emb), cls_mask : (C, B) bool
            Each class gets a fixed TOTAL budget `B =
            num_context_voxels_total_per_class`, distributed evenly over only
            the contexts that contain it (`_distribute_context_budget`). A
            class present in any context is fully populated (mask all False, no
            padding); a class absent from every context is left fully padded
            (mask all True), exactly like the legacy absent-everywhere case.
        """
        evaluation_sampler = getattr(self, "evaluation_context_sampler", None)
        if evaluation_sampler is not None:
            return evaluation_sampler(self, support_vols, support_segs, labels)

        N = support_vols.shape[0]
        C = int(labels.numel())
        device = support_vols.device

        # Pre-embed each context once. Each is (c_emb, ctx_HWD).
        ctx_embs = [
            self._embed(
                support_vols[n].unsqueeze(0),
                no_grad=self.detach_context_features,
            )
            for n in range(N)
        ]
        c_emb = ctx_embs[0].shape[0] if ctx_embs else self.anatomix_output_nc

        if self.fixed_total_context_voxels:
            return self._sample_context_prompts_fixed_total(
                support_segs, labels, ctx_embs, c_emb, device
            )

        b = self.num_context_voxels_per_class
        cls_kvs = torch.zeros(
            (C, N * b, c_emb), device=device, dtype=ctx_embs[0].dtype
        )
        cls_mask = torch.ones(
            (C, N * b), device=device, dtype=torch.bool
        )

        for n in range(N):
            seg_n = support_segs[n].to(torch.long)
            emb_n = ctx_embs[n]              # (c_emb, ctx_HWD)
            for ci, c in enumerate(labels.tolist()):
                present = (seg_n == c).any()
                if not bool(present.item()):
                    continue
                idx = sample_prompts_in_class(
                    seg_n, c, b,
                    border_fraction=self.context_border_fraction,
                    border_kernel_size=self.context_border_kernel_size,
                )  # (b,)
                cls_kvs[ci, n * b : (n + 1) * b] = emb_n[:, idx].T
                cls_mask[ci, n * b : (n + 1) * b] = False
        return cls_kvs, cls_mask

    def _sample_context_prompts_fixed_total(
        self,
        support_segs: torch.Tensor,    # (N, H, W, D), uint8/long
        labels: torch.Tensor,          # (C,) long
        ctx_embs: List[torch.Tensor],  # N tensors of (c_emb, ctx_HWD)
        c_emb: int,
        device: torch.device,
    ):
        """Fixed-total context bank — see `_sample_context_prompts_multilabel`."""
        B = self.num_context_voxels_total_per_class
        if B is None:
            raise RuntimeError(
                f"{type(self).__name__}: fixed_total_context_voxels=True but "
                "num_context_voxels_total_per_class is None."
            )
        N = len(ctx_embs)
        C = int(labels.numel())

        # Pre-cast each context seg to long once (reused across classes).
        segs_long = [support_segs[n].to(torch.long) for n in range(N)]

        cls_kvs = torch.zeros(
            (C, B, c_emb), device=device, dtype=ctx_embs[0].dtype
        )
        cls_mask = torch.ones((C, B), device=device, dtype=torch.bool)

        for ci, c in enumerate(labels.tolist()):
            present = [
                n for n in range(N) if bool((segs_long[n] == c).any().item())
            ]
            if not present:
                continue  # class absent everywhere → fully padded (mask True)
            counts = self._distribute_context_budget(len(present), B)
            w = 0
            for n, k_n in zip(present, counts):
                if k_n == 0:
                    continue
                idx = sample_prompts_in_class(
                    segs_long[n], c, k_n,
                    border_fraction=self.context_border_fraction,
                    border_kernel_size=self.context_border_kernel_size,
                )  # (k_n,)
                cls_kvs[ci, w : w + k_n] = ctx_embs[n][:, idx].T
                cls_mask[ci, w : w + k_n] = False
                w += k_n
        return cls_kvs, cls_mask



class AmixCrossAttentionMultiLabel(_AmixCrossMixin, _MultiLabelSamplingMixin, _AttentionHeadMixin):
    """
    Multi-label cross-attention: one shared head, C output channels.

    Per batch element, runs a single vectorized MHA call over all `C` classes
    in the current label set. Each class `c` has its own (zero-padded) K/V set
    of length `N*b`; classes absent from a particular context get
    zero-embedding K/V positions flagged via `key_padding_mask`. Classes
    absent from ALL contexts are excluded from the MHA call and emit a 0
    logit (scattered post-hoc).

    No per-class parameters. `self.attn` and `self.out_proj` are shared
    across all classes and all label sets — the same weights handle seg35
    (C=36 passes) and seg4 (C=5 passes) natively.

    Label set is supplied per-forward via the `labels` kwarg (1-D long tensor)
    or via `self._batch_labels` (set by the experiment runner before each
    forward + per-element sparse-target sampling). Strict — fails loudly if
    neither is set.

    Set `model.attn_checkpoint: true` in the training config to recompute chunked
    `_attend_multi` readouts during backward instead of retaining their activations.
    The default is false. This does not change query chunk sizes, unchunked calls,
    no-gradient evaluation, or the separate fused-head readout path.

    Set `model.attn_reuse_projections` to share first-layer voxel projections across
    classes and each layer's bank projections across chunks. Set `model.attn_compile`
    to compile the complete chunk readout, including its feed-forward network. Both
    default to false and apply only when gradients are disabled, including correction
    passes while the model is in training mode. They are independent and checkpoint
    compatible. Compilation has a first-use cost and may change floating-point results.
    """

    def __init__(
        self,
        num_target_voxels_per_class: int = 32,    # `a`
        num_context_voxels_per_class: int = 16,   # `b`
        context_border_fraction: float = 0.0,
        context_border_kernel_size: Union[int, Sequence[int]] = (3, 7),
        target_border_fraction: float = 0.0,
        target_border_kernel_size: Union[int, Sequence[int]] = (3, 7),
        target_border_bg_only: bool = False,
        target_sampling: Literal[
            "stratified", "uniform", "sqrt", "proportional"
        ] = "stratified",
        target_sampling_min_voxels_per_class: int = 1,
        #
        anatomix_ckpt: str = "anatomix-dev.pth",
        # "unet" (default) or "vit" — see `_AmixCrossMixin._init_amix`.
        anatomix_backbone: Literal["unet", "vit"] = "unet",
        anatomix_num_downs: int = 5,
        anatomix_ngf: int = 20,
        anatomix_output_nc: int = 16,
        anatomix_norm: str = "instance",
        anatomix_pooling: str = "Avg",
        anatomix_interp: str = "trilinear",
        anatomix_tile_size: Optional[Sequence[int]] = (128, 128, 128),
        anatomix_sw_batch_size: int = 1,
        anatomix_sw_overlap: float = 0.25,
        anatomix_sw_mode: str = "gaussian",
        anatomix_autocast_dtype: Optional[Literal["bf16", "fp16", "fp32"]] = "bf16",
        anatomix_compile: bool = False,
        anatomix_compile_mode: Literal[
            "default", "reduce-overhead", "max-autotune"
        ] = "reduce-overhead",
        freeze_anatomix: bool = False,
        detach_context_features: bool = False,
        # Per-voxel embedding normalization: "l2" (unit-norm, default) or
        # "channel" (LayerNorm over channels via ChannelLayerNorm).
        embedding_norm: str = "l2",
        # Backbone training regime:
        #   anatomix_random_init      — build a fresh, randomly-initialized
        #                               backbone (train from scratch) instead of
        #                               loading `anatomix_ckpt`.
        #   anatomix_trainable_last_n — freeze all but the last N conv layers of
        #                               the pretrained backbone (transfer learning).
        #                               Mutually exclusive with random_init.
        anatomix_random_init: bool = False,
        anatomix_trainable_last_n: Optional[int] = None,
        # Attention head config:
        attn_num_heads: int = 4,
        attn_dropout: float = 0.0,
        attn_embed_dim: Optional[int] = None,
        attn_q_chunk_size: Optional[int] = None,
        attn_activation: Optional[
            Literal["relu", "gelu", "silu", "tanh", "sigmoid"]
        ] = None,
        # Readout topology of the attention head — see
        # `_AttentionHeadMixin._build_attention_modules`. "plain" (the default)
        # is the historical head and builds no extra modules, so existing
        # checkpoints load unchanged.
        attn_block: Literal["plain", "residual", "transformer"] = "plain",
        attn_ffn_dim: Optional[int] = None,
        # Depth of the readout: L stacked blocks on one residual stream, all
        # re-attending to the same K/V. Requires a residual `attn_block`; L=1
        # (the default) builds no extra modules. See
        # `_AttentionHeadMixin._build_attention_modules`.
        attn_num_layers: int = 1,
        *,
        attn_checkpoint: bool = False,
        attn_reuse_projections: bool = False,
        attn_compile: bool = False,
    ) -> None:
        """
        Build the multilabel model with optional chunk and no-gradient optimizations.

        `attn_checkpoint`, `attn_reuse_projections`, and `attn_compile` default to false
        and do not add parameters to saved weights. The latter two options never apply
        to gradient-enabled predictions.
        """
        # NOTE: No **kwargs swallow — unrecognized config keys raise loudly
        # so misconfigured multi-label runs fail fast instead of silently
        # ignoring binary-only params copied from old configs.
        nn.Module.__init__(self)
        self._init_amix(
            anatomix_ckpt=anatomix_ckpt,
            anatomix_backbone=anatomix_backbone,
            anatomix_num_downs=anatomix_num_downs,
            anatomix_ngf=anatomix_ngf,
            anatomix_output_nc=anatomix_output_nc,
            anatomix_norm=anatomix_norm,
            anatomix_pooling=anatomix_pooling,
            anatomix_interp=anatomix_interp,
            anatomix_tile_size=anatomix_tile_size,
            anatomix_sw_batch_size=anatomix_sw_batch_size,
            anatomix_sw_overlap=anatomix_sw_overlap,
            anatomix_sw_mode=anatomix_sw_mode,
            anatomix_autocast_dtype=anatomix_autocast_dtype,
            anatomix_compile=anatomix_compile,
            anatomix_compile_mode=anatomix_compile_mode,
            freeze_anatomix=freeze_anatomix,
            detach_context_features=detach_context_features,
            embedding_norm=embedding_norm,
            anatomix_random_init=anatomix_random_init,
            anatomix_trainable_last_n=anatomix_trainable_last_n,
        )
        self._init_attention_head(
            c_anat=anatomix_output_nc + self._extra_head_in_channels(),
            attn_num_heads=attn_num_heads,
            attn_dropout=attn_dropout,
            attn_embed_dim=attn_embed_dim,
            attn_q_chunk_size=attn_q_chunk_size,
            attn_checkpoint=attn_checkpoint,
            attn_reuse_projections=attn_reuse_projections,
            attn_compile=attn_compile,
            attn_activation=attn_activation,
            attn_block=attn_block,
            attn_ffn_dim=attn_ffn_dim,
            attn_num_layers=attn_num_layers,
        )
        self._init_multilabel_sampling(
            num_target_voxels_per_class=num_target_voxels_per_class,
            num_context_voxels_per_class=num_context_voxels_per_class,
            context_border_fraction=context_border_fraction,
            context_border_kernel_size=context_border_kernel_size,
            target_border_fraction=target_border_fraction,
            target_border_kernel_size=target_border_kernel_size,
            target_border_bg_only=target_border_bg_only,
            target_sampling=target_sampling,
            target_sampling_min_voxels_per_class=target_sampling_min_voxels_per_class,
        )

        print(
            f"[AmixCrossAttentionMultiLabel] num_target_voxels_per_class="
            f"{self.num_target_voxels_per_class}, "
            f"num_context_voxels_per_class="
            f"{self.num_context_voxels_per_class}, "
            f"attn_block={self.attn_block}"
            + (
                f", attn_ffn_dim={self.attn_ffn[0].out_features}"
                if self.attn_block == "transformer"
                else ""
            )
            + f", attn_num_layers={self.attn_num_layers}"
            + (
                f", token_type_dim={d}"
                if (d := getattr(self, "token_type_dim", 0))
                else ""
            )
        )

    def _extra_head_in_channels(self) -> int:
        """Hook: extra input channels the attention head must be built to accept.

        0 here. `_build_attention_modules` derives
        `embed_proj = Linear(c_anat, c_emb)` from the sum, and `_attn_head_cfg`
        carries it to any second head built via `_inherit_head_cfg`, so a
        subclass that widens the tokens it feeds the head need only override
        this. Overridden by
        `AmixCrossAttentionClickFixedTotalMultiLabel` under `token_type_dim > 0`,
        which appends type channels to the query and to both halves of the K/V
        bank. `attn_embed_dim` (= c_emb) is deliberately NOT affected, so MHA's
        embed_dim and `attn_ffn_dim`'s 4*c_emb default stay put.
        """
        return 0

    def _query_tokens(self, q: torch.Tensor) -> torch.Tensor:
        """Hook: last chance to transform the query rows before attention.

        Identity here. Overridden by
        `AmixCrossAttentionClickFixedTotalMultiLabel` under `token_type_dim > 0`
        to append the QUERY type channels, which is mandatory once
        `embed_proj`'s input is widened — `_head_readout` runs the SAME
        `embed_proj` over the query and the K/V bank.
        """
        return q

    def _attend_valid_classes(
        self,
        q: torch.Tensor,                 # (Q, c_emb_in)
        kv: torch.Tensor,                # (V, M, c_emb_in) — valid classes only
        key_padding_mask: torch.Tensor,  # (V, M) bool
        valid_idx: torch.Tensor,         # (V,) channel indices of these rows
        prior_fn: Optional[Callable[[Optional[slice]], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Attend the valid classes and return their (V, Q) logits, in
        `valid_idx` order.

        Default: a single shared head over all valid classes (ignores
        `valid_idx`). Overridden by `_SeparateBgHeadMixin` to route the
        background channel (channel 0) through a separate head.

        `prior_fn` (per-class query rows for THESE V classes, see
        `_attend_multi`) is forwarded when given; the default call is left
        exactly as it was so the shared-query path is unchanged.
        """
        if prior_fn is None:
            return self._attend_multi(q, kv, key_padding_mask)
        return self._attend_multi(q, kv, key_padding_mask, prior_fn=prior_fn)

    @torch._dynamo.disable(recursive=True)
    def _predict_one_multilabel(
        self,
        target_vol: torch.Tensor,
        support_vols: torch.Tensor,
        support_segs: torch.Tensor,
        sparse_indices_b: Optional[torch.Tensor],  # (K,) flat target voxels to predict at; None = all
        spatial_shape,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (C, K) if sparse else (C, H, W, D). C = len(labels).

        Vectorized: ONE MHA call covering all classes with at least one valid
        K/V token. Classes whose K/V are all padding (absent from every
        context) get 0 logit scattered into their channel.
        """
        device = target_vol.device
        C = int(labels.numel())

        emb_t_flat = self._embed(target_vol)  # (c_emb, target_HWD)
        # Queries = the target voxels to predict at: the K sparse ones,
        # else every voxel. Only the query set differs; the rest is shared.
        if sparse_indices_b is not None:
            q = emb_t_flat[:, sparse_indices_b].T  # (K, c_emb)
        else:
            q = emb_t_flat.T                        # (HWD, c_emb)
        q = self._query_tokens(q)                   # identity unless token-typed
        Q = q.shape[0]

        cls_kvs, cls_mask = self._sample_context_prompts_multilabel(
            support_vols, support_segs, labels
        )

        # "Fully empty" classes: ALL N*b positions are padding.
        valid_per_class = (~cls_mask).any(dim=1)            # (C,)
        valid_idx = valid_per_class.nonzero(as_tuple=True)[0]

        out = torch.zeros((C, Q), device=device, dtype=q.dtype)
        if valid_idx.numel() == 0:
            if sparse_indices_b is not None:
                return out
            return out.view(C, *spatial_shape)

        valid_out = self._attend_valid_classes(
            q,
            cls_kvs[valid_idx],            # (V, N*b, c_emb)
            cls_mask[valid_idx],           # (V, N*b)
            valid_idx,                      # (V,) channel indices, ascending
        )                                  # (V, Q)
        out[valid_idx] = valid_out.to(out.dtype)

        if sparse_indices_b is not None:
            return out
        return out.view(C, *spatial_shape)

    def forward(
        self,
        target_image: torch.Tensor,                     # (B, C, H, W, D) or (B, 1, 1, H, W, D)
        support_images: Optional[torch.Tensor] = None,   # (B, N, 1, H_c, W_c, D_c)
        support_labels: Optional[torch.Tensor] = None,   # (B, N, 1, H_c, W_c, D_c) uint8/long
        sparse_indices: Optional[torch.Tensor] = None,   # (B, K) target voxels to predict at
        labels: Optional[torch.Tensor] = None,           # (C,) long
    ) -> torch.Tensor:
        """Returns (B, C, K) train / (B, C, H, W, D) val. C = len(labels).

        Strict label-source contract:
          - If `labels` kwarg is provided: use it (after validation).
          - Else if `self._batch_labels` was set by the runner: use that.
          - Else: raise. NO silent torch.unique fallback.
        Stashes the effective labels on `self._batch_labels` so subsequent
        calls to `_sample_target_indices` see the same set.
        """
        if target_image.ndim == 6:
            target_image = target_image.squeeze(1)
        B, _, H, W, D = target_image.shape
        spatial_shape = (H, W, D)

        # Resolve effective labels.
        if labels is not None:
            effective_labels = labels.to(device=target_image.device, dtype=torch.long)
        elif self._batch_labels is not None:
            effective_labels = self._batch_labels.to(
                device=target_image.device, dtype=torch.long
            )
        else:
            raise RuntimeError(
                "AmixCrossAttentionMultiLabel.forward requires either "
                "`labels` kwarg or `self._batch_labels` set by the runner. "
                "Got neither — refusing to silently guess via torch.unique."
            )
        self._validate_labels(effective_labels)
        # Stash so _sample_target_indices etc. see the same set.
        self._batch_labels = effective_labels

        C = int(effective_labels.numel())

        no_context = (
            support_images is None
            or support_labels is None
            or support_images.size(1) == 0
        )

        outs: List[torch.Tensor] = []
        for b in range(B):
            if no_context:
                Q = sparse_indices.shape[1] if sparse_indices is not None else (H * W * D)
                zeros = torch.zeros(
                    (C, Q), device=target_image.device, dtype=target_image.dtype,
                )
                outs.append(
                    zeros if sparse_indices is not None
                    else zeros.view(C, H, W, D)
                )
            else:
                out_b = self._predict_one_multilabel(
                    target_vol=target_image[b],
                    support_vols=support_images[b, :, 0],
                    support_segs=support_labels[b, :, 0],
                    sparse_indices_b=(
                        None if sparse_indices is None else sparse_indices[b]
                    ),
                    spatial_shape=spatial_shape,
                    labels=effective_labels,
                )
                outs.append(out_b.to(target_image.dtype))
        return torch.stack(outs, dim=0)  # (B, C, K) or (B, C, H, W, D)



class AmixCrossAttentionFixedTotalMultiLabel(AmixCrossAttentionMultiLabel):
    """Fixed-total context-budget variant of `AmixCrossAttentionMultiLabel`.

    Same as the parent cross-attention model except the per-class context K/V
    bank uses a FIXED TOTAL budget spread evenly over only the present contexts
    — see `AmixCrossScoreFixedTotalMultiLabel` for the full description. Because
    present classes are now fully populated (zero padding), the MHA call sees
    no padded keys for them, which is strictly safer for the attention.
    """

    fixed_total_context_voxels: bool = True

    def __init__(self, *, num_context_voxels_total_per_class: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.num_context_voxels_total_per_class = num_context_voxels_total_per_class



class _ClickContextMixin:
    """Adds interactive clicks on the TARGET volume to a multi-label model.

    Mix in *before* an `AmixCrossAttentionMultiLabel` subclass. The model gains
    one optional `forward` kwarg, `click_indices`, holding flat voxel indices
    into the target volume per label, so interactive and in-context evidence
    reach the same prediction.

    This mixin owns only the click INPUT side: the capability flag, the
    flat-index contract and its validation (`_resolve_click_indices`), and the
    `forward` loop that hands one element's clicks to `_predict_one_multilabel`.
    HOW those clicks are consumed belongs to the subclass, and the two in this
    file answer it differently — one concatenates click tokens onto each label's
    context K/V bank and runs the shared head over the union
    (`AmixCrossAttentionClickFixedTotalMultiLabel`), the other gives each source
    its own bank and its own head
    (`AmixCrossAttentionSeparateClickHeadFixedTotalMultiLabel`). So a concrete
    subclass MUST override `_predict_one_multilabel` with the extra
    `click_indices_b` keyword; this mixin alone does not make a context-only
    model click-aware.

    Adds NO parameters of its own, so where the subclass adds none either a
    trained `AmixCrossAttentionFixedTotalMultiLabel` checkpoint loads into a
    click model with `strict=True`, and `click_indices=None` reproduces the
    parent's output bit-for-bit (that path delegates to `super().forward`).

    Two behaviors differ from the parent, both deliberate and expected of every
    subclass:

      * A label absent from EVERY context example emits an exact 0 logit in the
        parent (its K/V row is all padding, so it is dropped from the MHA call).
        With clicks its bank is populated and it becomes predictable — the main
        point of combining interactive with in-context.
      * The parent's `no_context` early-out returns all-zero logits. With clicks
        present, the predict path runs anyway on click evidence alone, so a
        prediction can be driven with no context examples at all (S=0).

    Negative clicks are deliberately NOT part of this API. Background is always
    channel 0 (`_validate_labels` guarantees a contiguous-from-0 label set) and
    the per-label logits compete across channels, so "voxel v is not label c"
    is expressed as a positive click for label 0. That mapping belongs to the
    click sampler upstream, not here — do not add a `negative_click_indices`.
    """

    # Capability flag for `getattr`-negotiation by callers that may hold a
    # non-click model (see `evaluation.forward.multilabel_incontext_forward` and
    # the same pattern for class-aware clicks in `experiment/multiverseg_v3.py`).
    supports_click_indices = True

    @staticmethod
    def _has_clicks(click_indices_b: Optional[Dict[int, torch.Tensor]]) -> bool:
        """Whether this batch element carries at least one click index.

        Cheap pre-check used by `forward` to decide whether the `no_context`
        early-out still applies. `None`, `{}` and a dict of empty tensors all
        count as no clicks, so those cases keep the parent's behavior exactly.
        """
        if not click_indices_b:
            return False
        return any(
            torch.as_tensor(idx).numel() > 0 for idx in click_indices_b.values()
        )

    def _resolve_click_indices(
        self,
        click_indices_b: Optional[Dict[int, torch.Tensor]],
        spatial_shape,
        labels: torch.Tensor,
        device: torch.device,
    ) -> Optional[List[torch.Tensor]]:
        """Validate one element's click dict into a per-label index list.

        Parameters
        ----------
        click_indices_b : Optional[Dict[int, torch.Tensor]]
            `{label: flat indices}` for ONE target volume, or None/{}.
        spatial_shape : Sequence[int]
            The target's `(D, H, W)` — the indices are bounds-checked against
            `D*H*W`, so an index built against a differently-shaped (resampled,
            padded, cropped) volume fails loudly instead of silently pointing at
            unrelated anatomy.
        labels : torch.Tensor
            The `(C,)` effective label set; the returned list is in this order.

        Returns
        -------
        Optional[List[torch.Tensor]]
            One 1-D long tensor per label (empty for unclicked labels), or
            `None` when the element has no clicks at all — the caller then runs
            the parent's code path verbatim.

        Strict, matching `_validate_labels`' refuse-to-guess stance: a label
        outside `labels`, a non-integer dtype, or an out-of-range index raises.
        Duplicate indices are allowed (they contribute duplicate tokens, exactly
        as `sample_prompts_in_class` sampling with replacement already does).
        """
        if not click_indices_b:
            return None

        n_vox = math.prod(int(s) for s in spatial_shape)
        label_list = [int(c) for c in labels.tolist()]
        label_pos = {c: i for i, c in enumerate(label_list)}

        resolved = [
            torch.empty(0, dtype=torch.long, device=device) for _ in label_list
        ]
        for raw_label, raw_idx in click_indices_b.items():
            c = int(raw_label)
            if c not in label_pos:
                raise ValueError(
                    f"click_indices contains label {c}, which is not in the "
                    f"effective label set {label_list}. Clicks must reference a "
                    "label the model is predicting a channel for."
                )
            idx = torch.as_tensor(raw_idx)
            if idx.dtype == torch.bool:
                raise ValueError(
                    f"click_indices[{c}] has dtype bool — it looks like a voxel MASK, "
                    "but click_indices takes flat voxel INDICES. Convert with "
                    "`mask.flatten().nonzero(as_tuple=True)[0]`."
                )
            if idx.dtype.is_floating_point or idx.dtype.is_complex:
                raise ValueError(
                    f"click_indices[{c}] has dtype {idx.dtype}; flat voxel indices "
                    "must be an integer dtype."
                )
            idx = idx.reshape(-1).to(device=device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            lo = int(idx.min().item())
            hi = int(idx.max().item())
            if lo < 0 or hi >= n_vox:
                raise ValueError(
                    f"click_indices[{c}] has indices outside [0, {n_vox}) "
                    f"(min={lo}, max={hi}) for a target of spatial shape "
                    f"{tuple(int(s) for s in spatial_shape)}. Flat indices must "
                    "be built against the POST-COLLATE target shape this "
                    "forward received — the collate stack resamples, pads and "
                    "crops, so indices from another shape are meaningless."
                )
            resolved[label_pos[c]] = idx

        # All-empty (e.g. `{3: empty_tensor}`) counts as no clicks, so the
        # caller falls through to the parent's path.
        if not any(t.numel() for t in resolved):
            return None
        return resolved

    def forward(
        self,
        target_image: torch.Tensor,                      # (B, C, D, H, W) or (B, 1, 1, D, H, W)
        support_images: Optional[torch.Tensor] = None,    # (B, N, 1, D, H, W)
        support_labels: Optional[torch.Tensor] = None,    # (B, N, 1, D, H, W) uint8/long
        sparse_indices: Optional[torch.Tensor] = None,    # (B, K) target voxels to predict at
        labels: Optional[torch.Tensor] = None,            # (C,) long
        click_indices: Optional[
            List[Optional[Dict[int, torch.Tensor]]]
        ] = None,                                         # len B, {label: flat idx}
        prev_logits: Optional[torch.Tensor] = None,       # (B, C, K) / (B, C, D, H, W)
        prev_indices: Optional[torch.Tensor] = None,      # (B, K) flat voxels of a sparse prev_logits
    ) -> torch.Tensor:
        """Returns (B, C, K) train / (B, C, D, H, W) val. C = len(labels).

        `prev_logits` is the PREVIOUS ROUND's prediction for this target:
        `(B, C, K)` sparse or `(B, C, D, H, W)` dense, detached by the caller.
        Accepted only by a model advertising `supports_prev_prediction`
        (`AmixCrossAttentionClickFixedTotalMultiLabel` with `query_prior` or
        `prediction_tokens`); passing it to any other model raises rather than
        silently dropping it. `None` = no previous prediction (round 0), the
        historical call. Which query set a sparse `prev_logits` must cover
        depends on the knob:

          * `query_prior` — the SAME query set as this call (same
            `sparse_indices`, or dense); `prev_indices` is not taken.
          * `prediction_tokens` (`supports_prev_indices`) — any K target
            voxels, named by `prev_indices` `(B, K)` (flat indices into the
            target's `D*H*W`, the `sparse_indices` of the round that produced
            `prev_logits`); a dense `prev_logits` must be at the target's
            spatial shape and `prev_indices` must then be None.

        `click_indices` is a list of `B` dicts (one per target volume, `None` or
        `{}` for an element with no clicks) mapping a label to a 1-D integer
        tensor of FLAT voxel indices into the target's `D*H*W`. A row of the
        repo's click samplers slices straight in — `RandomClick`
        (`prompts3d/clicks3.py`) returns `(b_idx, flat_idx, polarity)`, so e.g.
        `[{3: flat_idx[b][polarity[b] == 1]} for b in range(B)]`.

        The index space is the same one `sparse_indices`, `sample_prompts_in_class`
        and `_gather_sparse_y` already use: C-order over `(D, H, W)`, i.e.
        `flat = (a0 * H + a1) * W + a2` in tensor-spatial axis order
        (`axis0=D, axis1=H, axis2=W`). This is exact because `_embed` returns
        features at the INPUT resolution — anatomix is a full U-Net and the only
        spatial op is pad -> forward -> `crop_from_pad` back — so a voxel index
        is a feature-column index with no rescaling.

        Indices must be built against the post-collate target shape passed to
        this call; they are bounds-checked against it (see `_resolve_click_indices`).

        `click_indices=None` (and no `prev_logits`) delegates to the parent
        forward unchanged.
        """
        if prev_logits is not None and not getattr(
            self, "supports_prev_prediction", False
        ):
            raise TypeError(
                f"prev_logits was passed but {type(self).__name__} does not "
                "take a previous prediction (no `supports_prev_prediction`). "
                "Use AmixCrossAttentionClickFixedTotalMultiLabel with "
                "`query_prior: true` or `prediction_tokens: true`."
            )
        takes_indices = bool(getattr(self, "supports_prev_indices", False))
        if prev_indices is not None:
            if prev_logits is None:
                raise ValueError(
                    "prev_indices was passed without prev_logits — the indices "
                    "name the voxels of a previous prediction, so pass both."
                )
            if not takes_indices:
                raise TypeError(
                    f"prev_indices was passed but {type(self).__name__} does not "
                    "take them (no `supports_prev_indices`): a `query_prior` "
                    "model reads the previous prediction at THIS call's query "
                    "set and needs no indices. Only `prediction_tokens: true` "
                    "takes them."
                )
        if click_indices is None and prev_logits is None:
            return super().forward(
                target_image,
                support_images,
                support_labels,
                sparse_indices=sparse_indices,
                labels=labels,
            )

        if target_image.ndim == 6:
            target_image = target_image.squeeze(1)
        B = target_image.shape[0]
        spatial_shape = tuple(target_image.shape[-3:])

        if click_indices is None:
            # A prior without clicks (a correction round that drew none):
            # run the click path on context + prior, no click tokens.
            click_indices = [None] * B
        if len(click_indices) != B:
            raise ValueError(
                f"click_indices has {len(click_indices)} entries but the batch has "
                f"{B} target volumes — pass one dict (or None) per element."
            )

        # Same strict label-source contract as the parent: explicit `labels`
        # kwarg, else the runner-set `_batch_labels`, else raise.
        effective_labels = labels if labels is not None else self._batch_labels
        if effective_labels is None:
            raise RuntimeError(
                f"{type(self).__name__}.forward requires either `labels` kwarg "
                "or `self._batch_labels` set by the runner. Got neither — "
                "refusing to silently guess via torch.unique."
            )
        effective_labels = effective_labels.to(
            device=target_image.device, dtype=torch.long
        )
        self._validate_labels(effective_labels)
        self._batch_labels = effective_labels

        C = int(effective_labels.numel())

        if prev_logits is not None:
            if prev_logits.ndim not in (3, 5) or tuple(prev_logits.shape[:2]) != (B, C):
                raise ValueError(
                    f"prev_logits must be (B, C, K) or (B, C, D, H, W) with "
                    f"B={B}, C={C}; got {tuple(prev_logits.shape)}. Pass the "
                    "previous round's output of THIS call shape (same "
                    "`sparse_indices`, same label set)."
                )
            if takes_indices:
                self._check_prev_indices(prev_logits, prev_indices, spatial_shape)
            elif (prev_logits.ndim == 3) != (sparse_indices is not None):
                raise ValueError(
                    f"prev_logits is {prev_logits.ndim - 2}-D spatial but the call "
                    f"is {'sparse' if sparse_indices is not None else 'dense'}: "
                    "the previous prediction must be at the SAME query set as "
                    "this forward (round 1 predicts at round 0's target points)."
                )

        no_context = (
            support_images is None
            or support_labels is None
            or support_images.size(1) == 0
        )

        outs: List[torch.Tensor] = []
        for b in range(B):
            click_indices_b = click_indices[b]
            # Only the parent's zero-logit early-out when there is neither
            # context nor a click to predict from. A prior alone is no
            # evidence: every bank would be empty.
            if no_context and not self._has_clicks(click_indices_b):
                Q = (
                    sparse_indices.shape[1]
                    if sparse_indices is not None
                    else math.prod(spatial_shape)
                )
                zeros = torch.zeros(
                    (C, Q), device=target_image.device, dtype=target_image.dtype,
                )
                outs.append(
                    zeros if sparse_indices is not None
                    else zeros.view(C, *spatial_shape)
                )
            else:
                # The prior kwarg is added ONLY when there is one, so a click
                # subclass whose `_predict_one_multilabel` predates it is
                # never handed an unknown keyword on the historical path.
                prior_kw = {}
                if prev_logits is not None:
                    prev_b = self._maybe_drop_prior(prev_logits[b])
                    if prev_b is not None:
                        prior_kw["prev_logits_b"] = prev_b
                        if prev_indices is not None:
                            prior_kw["prev_indices_b"] = prev_indices[b]
                out_b = self._predict_one_multilabel(
                    target_vol=target_image[b],
                    support_vols=None if no_context else support_images[b, :, 0],
                    support_segs=None if no_context else support_labels[b, :, 0],
                    sparse_indices_b=(
                        None if sparse_indices is None else sparse_indices[b]
                    ),
                    spatial_shape=spatial_shape,
                    labels=effective_labels,
                    click_indices_b=click_indices_b,
                    **prior_kw,
                )
                outs.append(out_b.to(target_image.dtype))
        return torch.stack(outs, dim=0)  # (B, C, K) or (B, C, D, H, W)

    def _maybe_drop_prior(
        self, prev_logits_b: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Hook: one element's previous prediction, or None to withhold it.

        Identity here. `AmixCrossAttentionClickFixedTotalMultiLabel` overrides
        it for `query_prior_dropout` / `prediction_tokens_dropout` (train-only
        Bernoulli).
        """
        return prev_logits_b



class AmixCrossAttentionClickFixedTotalMultiLabel(
    _ClickContextMixin, AmixCrossAttentionFixedTotalMultiLabel
):
    """Fixed-total in-context model that also accepts clicks on the target.

    `AmixCrossAttentionFixedTotalMultiLabel` plus the optional `click_indices`
    forward kwarg — see `_ClickContextMixin` for the flat-index contract and why
    negative clicks are not part of the API. Clicks are consumed by SHARING the
    context head: each label's cross-attention K/V bank becomes

        [ that label's context prompt tokens ] + [ its click tokens ]

    so interactive and in-context evidence go through the same projections.
    (`AmixCrossAttentionSeparateClickHeadFixedTotalMultiLabel` is the same click
    input feeding two separate heads instead.)

    Parameters
    ----------
    clicks_reduce_context_budget : bool, default=False
        How clicks share the budget `B = num_context_voxels_total_per_class`:

          * `False` — ADDITIVE. The context budget is sampled exactly as it is
            without clicks and click tokens are appended, so a label with `n`
            clicks holds `B + n` tokens.
          * `True` — RESERVED. Each label drops `n` of its context tokens, so
            its total stays exactly `B`. The drops are spread evenly over the
            contexts that hold the label (`_trim_context_evenly`), so no context
            loses more than one token more than any other. Every label present
            in >=1 context then holds the same `B` tokens no matter how many
            clicks it has, which collapses them into a SINGLE attention call
            (see `_predict_one_multilabel` on bucketing). Clicks are never
            dropped: a label with `n > B` clicks keeps no context tokens and
            holds `n`.

    token_type_dim : int, default=0
        Width `d` of a learned TOKEN-TYPE embedding that tells the shared head
        which tokens are clicks and which are context — the capacity this class
        otherwise lacks, since a single bank with an all-False
        `key_padding_mask` makes MHA permutation-invariant over its keys and
        nothing else marks a token's source.

        `0` disables it and the class adds no parameters. With `d > 0` an
        `nn.Embedding(3, d)` (rows `0=context`, `1=click`, `2=query`) is
        CONCATENATED onto the feature axis of every tensor entering
        `embed_proj`, whose input therefore widens from `c_anat` to
        `c_anat + d`::

            q    (Q, c_anat)  ->  cat(q,   w[2])  ->  (Q, c_anat + d)
            ctx  (R, c_anat)  ->  cat(ctx, w[0])  ->  (R, c_anat + d)
            clk  (n, c_anat)  ->  cat(clk, w[1])  ->  (n, c_anat + d)
            bank = cat([ctx, clk], dim=0)         ->  (R + n, c_anat + d)

        The query needs a row of its own because `_head_readout` runs the SAME
        `embed_proj` over the query and the bank. In the logits a click key then
        carries `q^T Wq^T Wk W_extra w[1]` where a context key carries
        `... w[0]`: a learned, QUERY-DEPENDENT bias toward or away from clicks.
        Requires an explicit `attn_embed_dim` (see `__init__`). Costs
        `c_emb * d + 3 * d` parameters — 1048 at `d=8, c_emb=128` — against a
        whole second head for
        `AmixCrossAttentionSeparateClickHeadFixedTotalMultiLabel`, which is the
        same distinction taken to its extreme.

        Note the context marker is applied in the `_sample_context_prompts_-
        multilabel` wrapper, so context tokens are marked identically whether or
        not the episode has clicks — there is no train/val skew between the
        click path and the no-click delegation.

    token_type_click_only : bool, default=False
        Learn a token-type embedding for CLICK tokens only: the CONTEXT and
        QUERY rows are zeroed at init and held at exactly zero for the whole
        training run (their gradient is masked; see `__init__` for why AdamW
        then keeps them bitwise zero). Requires `token_type_dim > 0`.

        This is the warm-start mode: with rows 0 and 2 exactly zero, the
        appended type channels contribute nothing on context or query tokens,
        so a no-click parent checkpoint whose `embed_proj.weight` has been
        placed into the first `c_anat` columns of the widened tensor makes the
        SAME in-context predictions as the parent — at init and at every later
        step, since the frozen rows can never drift. The extra `embed_proj`
        columns and the click row still learn freely (they receive gradient
        exclusively from click tokens: `dL/dW_extra = g @ t^T` vanishes when
        `t = 0`). Everything the model learns about token types is therefore
        attributable to clicks alone.

    query_prior : bool, default=False
        Feed the PREVIOUS round's prediction back in through the query. Adds
        **no parameters**: with the knob on, a forward that receives
        `prev_logits` (the previous round's output for the same query set,
        `(B, C, K)` sparse or `(B, C, D, H, W)` dense) makes the query PER
        CLASS — for class `c` at voxel `v` the `token_type_dim` type slot of
        the query row, where the learned QUERY marker sits, instead holds

            [ prior_c(v), 1, 0, ..., 0 ]

        i.e. the previous probability (`query_prior_input="prob"`, softmax
        over the C channels in fp32) or raw logit (`"logit"`) of THIS class at
        THIS voxel, plus a has-prior bit in the second channel (absent at
        `token_type_dim == 1`), so `prior_c = 0` is never mistaken for "no
        prior". The context and click tokens keep their learned markers. A
        forward WITHOUT `prev_logits` (round 0, the first prompted eval
        forward) is the historical one, bit for bit: the learned QUERY marker
        is used and nothing else changes — so iteration-0 metrics stay
        comparable to a model without the knob and its checkpoint loads with
        `strict=True`. Requires `token_type_dim > 0` (the slot must exist).
        The class then advertises `supports_prev_prediction`, which the runner
        (`experiment/amix_cross_v3.py`) and `evaluation/forward.py` negotiate
        on before passing `prev_logits`. Round 1 must predict at round 0's
        query points: the runner refuses
        `train.resample_target_points_per_iter: true` with this knob.

        Per-class query rows are built per q-chunk inside `_head_readout`
        (`prior_fn`), so nothing of size `(C, Q, ·)` is materialised at val.
        The cost is `embed_proj` + `ln_q` per class per chunk instead of once
        (they are hoisted on the shared-query path).

    query_prior_input : {"prob", "logit"}, default="prob"
        What fills the slot's first channel — see `query_prior`.

    query_prior_dropout : float, default=0.0
        TRAIN-only Bernoulli per batch element: with this probability the
        element's `prev_logits` is withheld and its forward runs the
        no-prior (learned marker) path. Consumes global torch RNG only when
        > 0 (one draw per element that has a prior), so `0.0` leaves the RNG
        stream of a non-prior run untouched.

    prediction_tokens : bool, default=False
        Feed the PREVIOUS round's prediction back in through the K/V BANK, as
        soft PREDICTION TOKENS. Adds **no parameters**. With the knob on, a
        forward that receives `prev_logits` (and, for a sparse one,
        `prev_indices`) turns `P = prediction_tokens_budget` of the previously
        predicted target voxels into tokens of EVERY class's bank: the target's
        own backbone feature at the voxel, with the `token_type_dim` type slot
        holding

            [ p_c(v), 0, 1, 0, ..., 0 ]

        — the previous softmax probability (over the call's C channels, fp32)
        of the BANK's class `c` at that voxel in channel 0, and a fixed
        prediction-token flag in channel 2 (channel 1 is `query_prior`'s
        has-prior bit; kept disjoint so a later composed arm keeps one meaning
        per channel). A token with `p_c ~ 1` is a target-domain positive
        exemplar of `c`; one with `p_c ~ 0` says "this looks like NOT-c on this
        image" — soft negative evidence the independent per-class head never
        otherwise sees. Each class's bank becomes

            [ context tokens ] + [ P prediction tokens ] + [ click tokens ]

        so the attention costs `(B + P + n) / (B + n)` of the no-prior pass
        (+50 % at B = 2048, P = 1024), at train and at the dense val pass. A
        class with neither context tokens nor clicks gets NO prediction tokens
        and keeps its exact-0 logit: an empty bank is no evidence and a prior
        alone must not invent some (moot at train, where the collate drops
        labels absent from every context and cs-0 steps click every label; it
        matters at eval). Clicked voxels are not excluded from the pool: a
        voxel may carry its click token in the clicked label's bank and also
        appear as a prediction token, with its raw `p_c`, in any bank.

        Where the P voxels come from (the POOL). With a sparse `prev_logits`
        `(C, K)` the pool is `prev_indices` — round 0's `sparse_indices`, the
        model's own target draw (proportional to GT class volume with a floor
        per class under `target_sampling: proportional`). With a dense one
        (val) the model draws the pool itself with the SAME sampler
        (`_sample_target_indices`) applied to `argmax(prev_logits)`, so it is
        proportional to PREDICTED class volume. This call's queries are
        unrelated to the pool: the runner keeps
        `resample_target_points_per_iter: true`, so round 1 queries a fresh
        draw that almost never coincides with a token (off is legal — a query
        may then attend to its own labelled token, a known small leak — and
        the runner warns).

        A forward WITHOUT `prev_logits` (round 0, the first prompted eval
        forward) is the historical one, bit for bit, and draws no extra RNG,
        so iteration-0 metrics stay comparable to a model without the knob and
        its checkpoint loads with `strict=True`. Requires `token_type_dim >= 3`
        (slot channels 0 and 2 must exist). Exclusive with `query_prior` in
        this version: both consume `prev_logits` under different contracts
        (the prior at THIS call's queries vs. any K voxels named by
        `prev_indices`), and composing them is a separate arm, not a silent
        union. The class then advertises `supports_prev_prediction` and
        `supports_prev_indices`; the runner stashes round 0's `sparse_indices`
        as `prev_indices` for it, the evaluator passes the dense previous
        round and no indices.

    prediction_tokens_budget : int, default=1024
        `P`, the prediction tokens per bank. Drawn with replacement when the
        pool is smaller than P (never at train: the pool is `1024 * C >= 2048`).

    prediction_tokens_selection : {"random", "topk"}, default="random"
        Which P pool voxels become tokens. `"random"` — ONE `randperm` draw of
        P voxels on the element-level torch RNG, shared by every class's bank;
        the banks then differ only in slot channel 0, and the low-`p_c` tokens
        supply the soft negatives. `"topk"` — per class `c`, the P pool voxels
        with the HIGHEST `p_c`: every token is a confident positive exemplar of
        its bank's class (a "self-prototype" bank), different voxels per bank,
        no RNG. Ties in `topk` break in an unspecified order, which cannot
        change the token SET the attention (a set operation) sees.

    prediction_tokens_dropout : float, default=0.0
        TRAIN-only Bernoulli per batch element: with this probability the
        element's `prev_logits` is withheld and its forward runs the no-prior
        path (the confirmation-bias lever). `1.0` withholds on every train
        step — the knob-on model trained as the knob-off one, for A/B checks.
        Consumes global torch RNG only when > 0, like `query_prior_dropout`.

    At `token_type_dim=0` this class adds no parameters, so a trained fixed-total
    checkpoint loads here with `strict=True` and `click_indices=None` reproduces
    its output exactly under either budget mode. `token_type_dim > 0` changes
    `embed_proj.weight`'s SHAPE, so such a checkpoint no longer loads directly
    (a shape mismatch raises regardless of `strict`). `query_prior` and
    `prediction_tokens` add no parameters on top of `token_type_dim`.
    """

    #: `query_prior_input` values.
    QUERY_PRIOR_INPUTS = ("prob", "logit")
    #: `prediction_tokens_selection` values.
    PREDICTION_TOKEN_SELECTIONS = ("random", "topk")
    #: Slot channel of a prediction token that holds its fixed flag (channel 0
    #: holds `p_c`; channel 1 is `query_prior`'s has-prior bit, kept disjoint).
    _PTOK_FLAG_CHANNEL = 2

    #: `token_type_embed` row per token source. Order is API: it fixes what a
    #: trained embedding's rows mean, so never reorder these.
    _TYPE_CONTEXT = 0
    _TYPE_CLICK = 1
    _TYPE_QUERY = 2

    def __init__(
        self,
        *,
        clicks_reduce_context_budget: bool = False,
        token_type_dim: int = 0,
        token_type_click_only: bool = False,
        query_prior: bool = False,
        query_prior_input: Literal["prob", "logit"] = "prob",
        query_prior_dropout: float = 0.0,
        prediction_tokens: bool = False,
        prediction_tokens_budget: int = 1024,
        prediction_tokens_selection: Literal["random", "topk"] = "random",
        prediction_tokens_dropout: float = 0.0,
        attn_q_chunk_size_nograd: Optional[int] = None,
        **kwargs,
    ):
        # Only meaningful against a fixed TOTAL budget: in legacy mode the
        # budget is per-(class, context), so there is no total to subtract from.
        # Reachable only from a subclass that flips the inherited class flag
        # back off — this class itself is always fixed-total. Checked BEFORE
        # `super().__init__`, which builds the anatomix backbone: a rejected
        # config should not cost a backbone construction. Reading the flag is
        # safe pre-`nn.Module.__init__` because it is a plain CLASS attribute.
        if clicks_reduce_context_budget and not self.fixed_total_context_voxels:
            raise ValueError(
                "clicks_reduce_context_budget=True requires a fixed-total "
                "context budget (fixed_total_context_voxels=True), since it "
                "subtracts the click count from "
                "`num_context_voxels_total_per_class`. The legacy sampler's "
                "budget is per-(class, context) and has no total to reduce."
            )
        if token_type_dim < 0:
            raise ValueError(
                f"token_type_dim must be >= 0 (0 disables); got {token_type_dim!r}"
            )
        if token_type_click_only and token_type_dim <= 0:
            raise ValueError(
                "token_type_click_only=True requires token_type_dim > 0: with "
                "no token-type embedding there is no click row to learn. Got "
                f"token_type_dim={token_type_dim!r}."
            )
        if query_prior and token_type_dim <= 0:
            raise ValueError(
                "query_prior=True requires token_type_dim > 0: the previous "
                "prediction is written into the query's type slot, which does "
                f"not exist at token_type_dim={token_type_dim!r}."
            )
        if query_prior_input not in self.QUERY_PRIOR_INPUTS:
            raise ValueError(
                f"query_prior_input must be one of {list(self.QUERY_PRIOR_INPUTS)}; "
                f"got {query_prior_input!r}"
            )
        if not (0.0 <= float(query_prior_dropout) < 1.0):
            raise ValueError(
                f"query_prior_dropout must be in [0, 1); got {query_prior_dropout!r}"
            )
        if query_prior_dropout and not query_prior:
            raise ValueError(
                "query_prior_dropout > 0 without query_prior=True has nothing to "
                "drop."
            )
        if prediction_tokens and query_prior:
            raise ValueError(
                "prediction_tokens=True and query_prior=True are exclusive in "
                "this version: both consume `prev_logits`, under different "
                "contracts (the prior at THIS call's queries vs. any K voxels "
                "named by `prev_indices`). Composing them is a separate arm, "
                "not a silent union."
            )
        if prediction_tokens and token_type_dim < 3:
            raise ValueError(
                "prediction_tokens=True requires token_type_dim >= 3: the "
                "previous probability is written into slot channel 0 and the "
                "prediction-token flag into channel 2, which do not exist at "
                f"token_type_dim={token_type_dim!r}."
            )
        if (
            isinstance(prediction_tokens_budget, bool)
            or not isinstance(prediction_tokens_budget, int)
            or prediction_tokens_budget < 1
        ):
            raise ValueError(
                "prediction_tokens_budget must be an int >= 1; got "
                f"{prediction_tokens_budget!r}"
            )
        if prediction_tokens_selection not in self.PREDICTION_TOKEN_SELECTIONS:
            raise ValueError(
                "prediction_tokens_selection must be one of "
                f"{list(self.PREDICTION_TOKEN_SELECTIONS)}; got "
                f"{prediction_tokens_selection!r}"
            )
        if not (0.0 <= float(prediction_tokens_dropout) <= 1.0):
            raise ValueError(
                "prediction_tokens_dropout must be in [0, 1]; got "
                f"{prediction_tokens_dropout!r}"
            )
        if prediction_tokens_dropout and not prediction_tokens:
            raise ValueError(
                "prediction_tokens_dropout > 0 without prediction_tokens=True "
                "has nothing to drop."
            )
        if token_type_dim > 0 and kwargs.get("attn_embed_dim") is None:
            # `_build_attention_modules` sets c_emb = c_anat when attn_embed_dim
            # is None — and c_anat is what `_extra_head_in_channels` widens, so
            # `embed_proj` would collapse to nn.Identity and the raw type
            # channels would enter MultiheadAttention instead of being projected
            # down, silently changing its embed_dim.
            raise ValueError(
                "token_type_dim > 0 requires an explicit attn_embed_dim. With "
                "attn_embed_dim=None the attention works at c_anat, which "
                "token_type_dim widens, so `embed_proj` would collapse to "
                "nn.Identity and the type channels would enter "
                "MultiheadAttention directly instead of being projected down. "
                f"Got token_type_dim={token_type_dim} with attn_embed_dim=None."
            )

        # Set BEFORE `super().__init__`, which builds the attention head and
        # therefore calls `_extra_head_in_channels`. A plain int is safe to
        # assign pre-`nn.Module.__init__`: `nn.Module.__setattr__` only guards
        # Parameters, Modules and buffers, and falls through to
        # `object.__setattr__` for anything else. The public name is set here
        # too so `_append_token_type` can read it unconditionally.
        self.token_type_dim = token_type_dim

        super().__init__(**kwargs)
        self.clicks_reduce_context_budget = clicks_reduce_context_budget
        self.token_type_click_only = token_type_click_only
        self.query_prior = bool(query_prior)
        self.query_prior_input = query_prior_input
        self.query_prior_dropout = float(query_prior_dropout)
        self.prediction_tokens = bool(prediction_tokens)
        self.prediction_tokens_budget = int(prediction_tokens_budget)
        self.prediction_tokens_selection = prediction_tokens_selection
        self.prediction_tokens_dropout = float(prediction_tokens_dropout)
        # Query chunk for NO-GRAD readouts only (dense val forward, the
        # iterative runner's full-volume rollout) — see `_resolve_q_chunk`.
        # None = one chunk size for every pass, as before. Compute only: the
        # chunking never changes the logits, so it is not a config knob of the
        # trained function and a checkpoint loads either way.
        if attn_q_chunk_size_nograd is not None and int(attn_q_chunk_size_nograd) < 1:
            raise ValueError(
                f"attn_q_chunk_size_nograd must be >= 1 or None; got {attn_q_chunk_size_nograd!r}"
            )
        self.attn_q_chunk_size_nograd = (
            None if attn_q_chunk_size_nograd is None else int(attn_q_chunk_size_nograd)
        )

        if token_type_dim:
            # Registered AFTER `super().__init__` so every head parameter keeps
            # its index: an optimizer `state_dict` is keyed by parameter INDEX
            # (see `_HEAD_ATTRS`), so prepending would mispair Adam moments on
            # resume. Built only when the knob is on, which is what keeps a
            # default model's `state_dict` byte-identical to a pre-knob one.
            self.token_type_embed = nn.Embedding(3, token_type_dim)
            # ORDINARY init — deliberately NOT zeroed. `embed_proj` contributes
            # `W @ x + W_extra @ t`, which is BILINEAR in (W_extra, t): zeroing
            # either factor zeroes the OTHER factor's gradient
            # (dL/dW_extra = g t^T, dL/dt = W_extra^T g), stalling half the
            # mechanism for a step and making its first update rank-1 along
            # whatever survived. With both at ordinary init both learn from
            # step 0. Same policy as the separate click head's readouts.
            #
            # std=0.02, NOT nn.Embedding's N(0,1) default: the features beside
            # these channels are channel-LayerNorm'd to ~unit rms, so the type
            # term's size relative to the feature term is
            # sqrt(d)*std / sqrt(c_anat) ~= 0.5*std for d=8, c_anat=32. At
            # std=1 the type channels would supply ~50% of `embed_proj`'s output
            # and swamp the anatomical features; at 0.02 they are the intended
            # ~1% nudge that training can grow. Raise this if the marker
            # underfits. (`token_type_click_only` below is the deliberate
            # exception, for warm starts from a no-click parent.)
            nn.init.normal_(self.token_type_embed.weight, std=0.02)

            if token_type_click_only:
                # Zero the CONTEXT and QUERY rows and hold them there, so the
                # type channels are inert everywhere except on click tokens —
                # the warm-start parity contract in the class docstring. The
                # bilinear stall the comment above warns about does not apply:
                # the CLICK row keeps its ordinary init, so `W_extra` still
                # gets gradient from click tokens at step 0.
                with torch.no_grad():
                    self.token_type_embed.weight[self._TYPE_CONTEXT].zero_()
                    self.token_type_embed.weight[self._TYPE_QUERY].zero_()
                # Freeze via a gradient hook, NOT `padding_idx`:
                # `_append_token_type` indexes `.weight[row]` directly and
                # never goes through `F.embedding`, so `padding_idx` would
                # zero-init the row but autograd would still hand it an
                # ordinary gradient. A buffer/Parameter split would freeze
                # too, but it changes the `state_dict` key set and breaks
                # `strict=True` loads; the hook adds no state. AdamW then
                # keeps the rows BITWISE zero: masked grad means both moment
                # buffers stay 0 so the step term is 0, and the decoupled
                # weight decay multiplies an exact 0.0 by (1 - lr*wd), which
                # is still exactly 0.0. Re-registered on every construction
                # (the flag lives in the run config), so resumes stay frozen.
                self.token_type_embed.weight.register_hook(
                    self._mask_non_click_token_type_grad
                )

    def _extra_head_in_channels(self) -> int:
        """The `d` type channels appended to every tensor entering `embed_proj`.

        See `AmixCrossAttentionMultiLabel._extra_head_in_channels`. Called from
        `super().__init__`, hence the pre-init assignment of `token_type_dim`.
        """
        return self.token_type_dim

    # ------------------------------------------------------------------
    # Previous-prediction query prior (`query_prior`)
    # ------------------------------------------------------------------

    @property
    def supports_prev_prediction(self) -> bool:
        """Runner / evaluator negotiation flag: this instance takes `prev_logits`."""
        return bool(getattr(self, "query_prior", False)) or bool(
            getattr(self, "prediction_tokens", False)
        )

    @property
    def supports_prev_indices(self) -> bool:
        """Negotiation flag: a sparse `prev_logits` must come with the flat
        voxels it was predicted at (`prev_indices`) — `prediction_tokens`
        builds its pool from them. See `_ClickContextMixin.forward`."""
        return bool(getattr(self, "prediction_tokens", False))

    def _maybe_drop_prior(
        self, prev_logits_b: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """`query_prior_dropout` / `prediction_tokens_dropout`: withhold one
        element's prior at train. (The two knobs are exclusive, so exactly one
        rate applies.)"""
        p = (
            self.prediction_tokens_dropout if self.prediction_tokens
            else self.query_prior_dropout
        )
        if prev_logits_b is None or not self.training or p <= 0.0:
            return prev_logits_b
        if float(torch.rand(())) < p:
            return None
        return prev_logits_b

    def _prior_values(
        self, prev_logits_b: torch.Tensor, C: int, Q: int
    ) -> torch.Tensor:
        """One element's `(C, Q)` prior values aligned with this call's queries.

        `prev_logits_b` is `(C, K)` sparse or `(C, D, H, W)` dense; its
        flattened query axis must have exactly `Q` entries — the previous
        round predicted at the SAME points this round queries. Returns fp32
        softmax probabilities (`query_prior_input="prob"`) or the raw logits
        as fp32.
        """
        if int(prev_logits_b.shape[0]) != C:
            raise ValueError(
                f"prev_logits has {int(prev_logits_b.shape[0])} channels for a "
                f"{C}-class label set"
            )
        prev = prev_logits_b.reshape(C, -1)
        if int(prev.shape[1]) != Q:
            raise ValueError(
                f"prev_logits covers {int(prev.shape[1])} query voxels but this "
                f"forward queries {Q}: the previous prediction must be at the "
                "same query set (same `sparse_indices` at train; the dense "
                "volume at val)."
            )
        prev = prev.float()
        if self.query_prior_input == "prob":
            return torch.softmax(prev, dim=0)
        return prev

    def _make_prior_fn(
        self,
        q_feat: torch.Tensor,      # (Q, c_anat) UNMARKED query features
        prior_vals: torch.Tensor,  # (C, Q) fp32
        valid_idx: torch.Tensor,   # (V,) the classes of this attention call
    ) -> Callable[[Optional[slice]], torch.Tensor]:
        """Per-class query rows for `_head_readout(prior_fn=...)`.

        For q-slice `sl` returns `(V, Qc, c_anat + d)`: the shared target
        features with the type slot `[prior_c(v), 1, 0, ...]` per class (see
        the class docstring). Built lazily per chunk — never for all Q.
        """
        d = self.token_type_dim
        vals = prior_vals[valid_idx]                       # (V, Q)

        def prior_fn(sl: Optional[slice]) -> torch.Tensor:
            qf = q_feat if sl is None else q_feat[sl]      # (Qc, c_anat)
            v = vals if sl is None else vals[:, sl]         # (V, Qc)
            V, Qc = v.shape
            slot = torch.zeros((V, Qc, d), device=qf.device, dtype=qf.dtype)
            slot[..., 0] = v.to(qf.dtype)
            if d > 1:
                slot[..., 1] = 1
            return torch.cat(
                [qf.unsqueeze(0).expand(V, Qc, qf.size(-1)), slot], dim=-1
            )

        return prior_fn

    # ------------------------------------------------------------------
    # Previous-prediction tokens in the bank (`prediction_tokens`)
    # ------------------------------------------------------------------

    def _check_prev_indices(
        self,
        prev_logits: torch.Tensor,              # (B, C, K) or (B, C, D, H, W)
        prev_indices: Optional[torch.Tensor],   # (B, K) or None
        spatial_shape,
    ) -> None:
        """The `prediction_tokens` contract for `prev_logits` / `prev_indices`,
        checked once per batch by `_ClickContextMixin.forward` (after its
        `(B, C)` check). Sparse needs the indices — `(B, K)`, integer, in
        `[0, D*H*W)`; dense needs none and the target's spatial shape.
        Refuse-to-guess, as `_resolve_click_indices`.
        """
        shape = tuple(int(s) for s in spatial_shape)
        n_vox = math.prod(shape)
        if prev_logits.ndim == 3:
            B, _, K = (int(s) for s in prev_logits.shape)
            if prev_indices is None:
                raise ValueError(
                    f"prev_logits is sparse {tuple(prev_logits.shape)} but "
                    "prev_indices is None: prediction_tokens needs the flat "
                    "target voxels the previous round predicted at (its "
                    "`sparse_indices`) — this round's queries are a different "
                    "draw, so they cannot stand in for them."
                )
            if prev_indices.dtype.is_floating_point or prev_indices.dtype in (
                torch.bool, torch.complex64, torch.complex128
            ):
                raise ValueError(
                    f"prev_indices has dtype {prev_indices.dtype}; flat voxel "
                    "indices must be an integer dtype."
                )
            if tuple(int(s) for s in prev_indices.shape) != (B, K):
                raise ValueError(
                    f"prev_indices must be (B, K) = {(B, K)}, one flat index per "
                    f"column of prev_logits; got {tuple(prev_indices.shape)}."
                )
            lo = int(prev_indices.min().item())
            hi = int(prev_indices.max().item())
            if lo < 0 or hi >= n_vox:
                raise ValueError(
                    f"prev_indices has indices outside [0, {n_vox}) (min={lo}, "
                    f"max={hi}) for a target of spatial shape {shape}. Flat "
                    "indices must be built against the POST-COLLATE target shape "
                    "this forward received."
                )
        else:
            if prev_indices is not None:
                raise ValueError(
                    f"prev_logits is dense {tuple(prev_logits.shape)} but "
                    "prev_indices was passed: a dense previous prediction "
                    "covers every voxel and the pool is drawn from its argmax "
                    "here — pass prev_indices=None."
                )
            got = tuple(int(s) for s in prev_logits.shape[2:])
            if got != shape:
                raise ValueError(
                    f"dense prev_logits has spatial shape {got} but the target "
                    f"is {shape}: the previous prediction must be at this "
                    "call's target resolution."
                )

    def _prediction_token_indices(
        self,
        prev_logits_b: torch.Tensor,             # (C, K) sparse or (C, D, H, W) dense
        prev_indices_b: Optional[torch.Tensor],  # (K,) flat voxels of the sparse one; None if dense
        spatial_shape,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Which target voxels become this element's prediction tokens, and
        their per-class probabilities.

        Returns `(idx, probs)`: `idx` is `(P,)` long flat target voxels
        (`prediction_tokens_selection="random"`, shared by every bank) or
        `(C, P)` (`"topk"`, one row per bank); `probs` is `(C, P)` fp32 with
        `probs[c, j] = softmax(prev_logits_b[:, idx[j]])[c]` (resp.
        `idx[c, j]`).

        The pool (see the class docstring): sparse = `prev_indices_b` itself;
        dense = `_sample_target_indices` on `argmax(prev_logits_b)`, the
        training target draw applied to the PREDICTED segmentation (it reads
        the label set from `_batch_labels`, which `forward` set). A pool
        smaller than P is first padded with draws with replacement — never at
        train, where the pool is `1024 * C`; a guard for an odd eval label
        set. Pool duplicates (the sampler draws with replacement within a
        class) pass through as duplicate tokens, the same harmless duplication
        `sample_prompts_in_class` already produces for context tokens.
        """
        C = int(prev_logits_b.shape[0])
        P = self.prediction_tokens_budget
        device = prev_logits_b.device
        pool_logits = prev_logits_b.reshape(C, -1)
        if prev_indices_b is not None:
            pool = prev_indices_b.reshape(-1).to(device=device, dtype=torch.long)
        else:
            pred_seg = pool_logits.argmax(dim=0).view(
                *(int(s) for s in spatial_shape)
            )
            pool = self._sample_target_indices(pred_seg)
            pool_logits = pool_logits[:, pool]
        n = int(pool.numel())
        if n < P:
            extra = torch.randint(n, (P - n,), device=device)
            pool = torch.cat([pool, pool[extra]])
            pool_logits = torch.cat([pool_logits, pool_logits[:, extra]], dim=1)
        probs_pool = torch.softmax(pool_logits.float(), dim=0)      # (C, n) fp32
        if self.prediction_tokens_selection == "random":
            pos = torch.randperm(int(pool.numel()), device=device)[:P]
            return pool[pos], probs_pool[:, pos]
        vals, pos = probs_pool.topk(P, dim=1)                       # (C, P), descending
        return pool[pos], vals

    def _prediction_token_rows(
        self,
        emb_t_flat: torch.Tensor,   # (c_anat, D*H*W) target embedding
        idx: torch.Tensor,          # (P,) or (C, P), from `_prediction_token_indices`
        probs: torch.Tensor,        # (C, P) fp32
        bank_dtype: torch.dtype,
    ) -> torch.Tensor:
        """`(C, P, c_anat + d)` bank rows `[feat(v) ; p_c(v), 0, 1, 0, ...]`,
        one per class, in the bank dtype.

        One gather for every bank (`C * P * (c_anat + d)` bf16 is 7.8 MB at
        C = 95, P = 1024); `random`'s shared features are expanded, not
        copied, until the cat. The slot is cast to the bank dtype at the cat,
        the same policy as `_append_token_type`.
        """
        C, P = (int(s) for s in probs.shape)
        feat = emb_t_flat[:, idx.reshape(-1)].T.to(bank_dtype)
        feat = (
            feat.reshape(C, P, feat.size(-1)) if idx.ndim == 2
            else feat.unsqueeze(0).expand(C, P, feat.size(-1))
        )
        slot = torch.zeros(
            (C, P, self.token_type_dim), device=feat.device, dtype=bank_dtype
        )
        slot[..., 0] = probs.to(bank_dtype)
        slot[..., self._PTOK_FLAG_CHANNEL] = 1
        return torch.cat([feat, slot], dim=-1)

    def _mask_non_click_token_type_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero the CONTEXT and QUERY rows of `token_type_embed.weight`'s grad.

        Registered as a tensor hook on the weight when `token_type_click_only`
        is on — the freeze half of the contract (`__init__` zeroes the rows;
        this keeps any optimizer from moving them). Clones rather than mutating:
        a hook must not modify its input in place.
        """
        grad = grad.clone()
        grad[self._TYPE_CONTEXT].zero_()
        grad[self._TYPE_QUERY].zero_()
        return grad

    # ------------------------------------------------------------------
    # Token-type embedding (no-ops at token_type_dim=0)
    # ------------------------------------------------------------------

    def _append_token_type(self, x: torch.Tensor, row: int) -> torch.Tensor:
        """Concat `token_type_embed`'s `row` onto `x`'s last axis.

        `x` is `(..., c_anat)` with any number of leading dims — `(Q, c_anat)`
        queries, `(n, c_anat)` click tokens, `(C, B, c_anat)` context banks.
        Returns `(..., c_anat + d)`.

        The marker is cast to `x.dtype` rather than the reverse: the banks are
        assembled in the embedding's dtype (bf16 under
        `anatomix_autocast_dtype: bf16`) and `torch.cat` needs one dtype, so
        promoting `x` to the parameter's fp32 here would silently double the
        bank's memory at val, where the context bank is the largest tensor.
        """
        if not self.token_type_dim:
            return x
        marker = self.token_type_embed.weight[row].to(x.dtype)
        return torch.cat(
            [x, marker.expand(*x.shape[:-1], self.token_type_dim)], dim=-1
        )

    def _query_tokens(self, q: torch.Tensor) -> torch.Tensor:
        """Mark the query rows. See `AmixCrossAttentionMultiLabel._query_tokens`."""
        return self._append_token_type(q, self._TYPE_QUERY)

    def _sample_context_prompts_multilabel(self, *args, **kwargs):
        """Sample the context bank, then mark every token as CONTEXT.

        Wrapping the sampler — rather than marking inside the bank loop of
        `_predict_one_multilabel` — is what makes the marking consistent across
        BOTH paths: this method is `self.`-dispatched, so it also applies to the
        no-click delegation to `super()._predict_one_multilabel`. Marking in the
        bank loop instead would leave context unmarked in no-click episodes and
        marked in click episodes, i.e. a train/val skew inside one model.

        Padded positions get the marker too. Harmless, and for the same reason
        `_head_readout`'s `ln_kv` may map them to its bias: those positions are
        exactly the ones `key_padding_mask` excludes.
        """
        cls_kvs, cls_mask = super()._sample_context_prompts_multilabel(
            *args, **kwargs
        )
        return self._append_token_type(cls_kvs, self._TYPE_CONTEXT), cls_mask

    def _trim_context_evenly(
        self, kv_ctx: torch.Tensor, num_present: int, n_drop: int
    ) -> torch.Tensor:
        """Drop `n_drop` context tokens, split as evenly as possible per context.

        `kv_ctx` is one label's compacted `(R, c_emb)` context bank. The
        fixed-total sampler lays it out as one contiguous block per present
        context, in ascending context order, each block sized by
        `_distribute_context_budget(num_present, R)` — so re-deriving that split
        recovers the block boundaries without duplicating the sampler's logic.

        The kept-per-context counts are `_distribute_context_budget(num_present,
        R - n_drop)`, i.e. the same even-split-with-front-loaded-remainder rule
        the sampler used to build the bank. Because that split is monotone in
        its total and `R - n_drop <= R`, the kept count can never exceed a
        context's available block, so no clamping is needed. Removal is
        deterministic and balanced: no context loses more than one token more
        than any other.

        Within a context the tokens are i.i.d. draws (`sample_prompts_in_class`
        samples with replacement), so dropping that block's tail is arbitrary in
        the right way — it introduces no spatial bias.
        """
        R = kv_ctx.shape[0]
        if n_drop >= R:
            return kv_ctx[:0]
        if num_present <= 1:
            # One block — nothing to balance across.
            return kv_ctx[: R - n_drop]

        counts = self._distribute_context_budget(num_present, R)
        keep = self._distribute_context_budget(num_present, R - n_drop)
        idx: List[torch.Tensor] = []
        w = 0
        for block, k_keep in zip(counts, keep):
            idx.append(torch.arange(w, w + k_keep, device=kv_ctx.device))
            w += block
        return kv_ctx[torch.cat(idx)]

    @staticmethod
    def _num_present_contexts(
        support_segs: torch.Tensor, labels: torch.Tensor
    ) -> List[int]:
        """Per label, how many context examples contain it (the sampler's `M`).

        One `bincount` per context plus a single host sync, rather than the
        `N x C` grid of `.any().item()` calls the shared sampler does — the
        per-element `.item()` sync is the pattern `prompts3d/clicks3.py` avoids.

        The int64 cast `bincount` needs is done one context at a time and freed
        each iteration: casting all `N` at once peaks at `N * V * 8` bytes, which
        is 537 MB for four 256**3 contexts.
        """
        n_bins = int(labels.max()) + 1
        present = torch.stack([
            torch.bincount(
                seg.reshape(-1).to(torch.long), minlength=n_bins
            )[:n_bins] > 0
            for seg in support_segs
        ])                                              # (N, n_bins)
        return present[:, labels].sum(dim=0).tolist()

    @torch._dynamo.disable(recursive=True)
    def _predict_one_multilabel(
        self,
        target_vol: torch.Tensor,
        support_vols: Optional[torch.Tensor],
        support_segs: Optional[torch.Tensor],
        sparse_indices_b: Optional[torch.Tensor],  # (K,) flat target voxels to predict at; None = all
        spatial_shape,
        labels: torch.Tensor,
        click_indices_b: Optional[Dict[int, torch.Tensor]] = None,
        prev_logits_b: Optional[torch.Tensor] = None,
        prev_indices_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (C, K) if sparse else (C, H, W, D), as the parent does.

        Delegates straight to `super()` when this element has no clicks (and
        no prior), so the no-click path is the parent's code and stays
        bit-identical.

        `prev_logits_b` is this element's previous prediction. Under
        `query_prior` it is at the same query set and each bucket's attention
        runs with per-class query rows (`_make_prior_fn`) instead of the shared
        marked query; the bank assembly is unchanged. Under
        `prediction_tokens` the query is untouched and every non-empty bank
        gains P prediction tokens drawn from the pool `prev_indices_b` (sparse)
        or from `argmax(prev_logits_b)` (dense) — `_prediction_token_indices`
        / `_prediction_token_rows`. The token draw happens AFTER the context
        draw, so a knob-on round differs from a knob-off one only by the
        appended tokens, not by a shifted context sample.

        With clicks, each label's bank is built as [compacted real context
        tokens] (+ [prediction tokens]) + [click tokens gathered from the
        target embedding], then labels are BUCKETED BY TOKEN COUNT and one
        attention call is issued per bucket.

        Why bucket instead of one padded call: `_attend_multi` drops the
        `key_padding_mask` entirely when nothing is padded, which is what lets
        SDPA dispatch Flash Attention (10-50x, see its comment). `mask_active`
        is a single bool over the whole `(V, M)` tensor, so one ragged bank
        padded to the max click count would flip the mask on for EVERY label and
        lose Flash across the board — punishing at val, where Q = D*H*W. Bucket
        counts are few in practice (`{ctx_budget + n_clicks}` values, plus
        `{n_clicks}` for click-only labels; each `+ P` with prediction
        tokens), every bucket's bank is fully
        populated, and because the head has no per-class parameters and `q` is
        shared, the grouped result is identical to the single padded call.

        Each bucket goes through `_attend_valid_classes` (not `_attend_multi`
        directly) so `_SeparateBgHeadMixin` still routes channel 0 to the bg
        head within whatever subset it is handed.
        """
        click_list = self._resolve_click_indices(
            click_indices_b, spatial_shape, labels, target_vol.device
        )
        if click_list is None and prev_logits_b is None:
            return super()._predict_one_multilabel(
                target_vol=target_vol,
                support_vols=support_vols,
                support_segs=support_segs,
                sparse_indices_b=sparse_indices_b,
                spatial_shape=spatial_shape,
                labels=labels,
            )

        device = target_vol.device
        C = int(labels.numel())
        if click_list is None:
            # Prior but no clicks: context-only banks, per-class queries.
            click_list = [
                torch.empty(0, dtype=torch.long, device=device) for _ in range(C)
            ]

        emb_t_flat = self._embed(target_vol)  # (c_emb, target_HWD)
        # Queries = the target voxels to predict at: the K sparse ones,
        # else every voxel. Only the query set differs; the rest is shared.
        q_feat = (
            emb_t_flat if sparse_indices_b is None
            else emb_t_flat[:, sparse_indices_b]
        ).T                                                 # (Q, c_anat)
        q = self._query_tokens(q_feat)
        prior_vals = (
            self._prior_values(prev_logits_b, C, int(q.shape[0]))
            if prev_logits_b is not None and self.query_prior else None
        )

        # Context bank, unless there are no context examples at all (S=0 or the
        # caller passed none) — clicks alone can drive the prediction then.
        cls_kvs = cls_mask = None
        if support_vols is not None and support_vols.shape[0] > 0:
            cls_kvs, cls_mask = self._sample_context_prompts_multilabel(
                support_vols, support_segs, labels
            )
        bank_dtype = cls_kvs.dtype if cls_kvs is not None else emb_t_flat.dtype

        # Prediction tokens, `(C, P, c_emb)`: drawn after the context bank so
        # the context draw is the knob-off model's under the same seed.
        ptok_rows = None
        if prev_logits_b is not None and self.prediction_tokens:
            idx, probs = self._prediction_token_indices(
                prev_logits_b, prev_indices_b, spatial_shape
            )
            ptok_rows = self._prediction_token_rows(
                emb_t_flat, idx, probs, bank_dtype
            )

        # Per-label bank: real (non-padding) context tokens, then click tokens.
        # Under the fixed-total budget the compaction is exact — a label present
        # in >=1 context has zero padding, a label absent everywhere has zero
        # real tokens — so this neither drops nor invents context evidence.
        # RESERVED mode needs each label's present-context count to spread the
        # dropped tokens evenly over the contexts. Computed once, and only when
        # the mode is on (it re-scans the context segs).
        n_present = (
            self._num_present_contexts(support_segs, labels)
            if self.clicks_reduce_context_budget and cls_kvs is not None
            else None
        )

        # Ascending `ci` iteration keeps every bucket's channel indices
        # ascending, which `_attend_valid_classes` documents and
        # `_SeparateBgHeadMixin` relies on to route channel 0.
        banks: Dict[int, torch.Tensor] = {}
        buckets: Dict[int, List[int]] = {}
        for ci in range(C):
            parts: List[torch.Tensor] = []
            n_c = int(click_list[ci].numel())
            kv_ctx = None
            if cls_kvs is not None:
                kv_ctx = cls_kvs[ci][~cls_mask[ci]]
                if n_present is not None and n_c:
                    # Give up one context token per click so the label's TOTAL
                    # stays at the configured budget, taken evenly across the
                    # contexts that hold it.
                    kv_ctx = self._trim_context_evenly(
                        kv_ctx, n_present[ci], n_c
                    )
                parts.append(kv_ctx)
            if ptok_rows is not None and (
                n_c or (kv_ctx is not None and kv_ctx.shape[0])
            ):
                # Only into a bank that already holds evidence (context or
                # clicks): a label with neither keeps its exact-0 logit below,
                # prior or not.
                parts.append(ptok_rows[ci])
            if n_c:
                # Marked CLICK, against the CONTEXT rows the sampler wrapper
                # already appended — so the two halves of this bank differ by a
                # learned vector in the type channels, not just by provenance.
                parts.append(
                    self._append_token_type(
                        emb_t_flat[:, click_list[ci]].T.to(bank_dtype),
                        self._TYPE_CLICK,
                    )
                )
            if not parts:
                continue  # no context and no clicks → 0 logit, as the parent
            bank = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if bank.shape[0] == 0:
                continue  # context present but empty for this label → 0 logit
            banks[ci] = bank
            buckets.setdefault(int(bank.shape[0]), []).append(ci)

        out = torch.zeros((C, q.shape[0]), device=device, dtype=q.dtype)
        for m, cis in buckets.items():
            valid_idx = torch.tensor(cis, device=device, dtype=torch.long)
            kv = torch.stack([banks[ci] for ci in cis], dim=0)  # (V, m, c_emb)
            # Fully populated by construction, so the mask is all-False and the
            # SDPA fast path survives.
            key_padding_mask = torch.zeros(
                (len(cis), m), device=device, dtype=torch.bool
            )
            if prior_vals is None:
                valid_out = self._attend_valid_classes(
                    q, kv, key_padding_mask, valid_idx
                )                                               # (V, Q)
            else:
                valid_out = self._attend_valid_classes(
                    q, kv, key_padding_mask, valid_idx,
                    prior_fn=self._make_prior_fn(q_feat, prior_vals, valid_idx),
                )                                               # (V, Q)
            out[valid_idx] = valid_out.to(out.dtype)

        if sparse_indices_b is not None:
            return out
        return out.view(C, *spatial_shape)



class AmixCrossAttentionClickCachedFixedTotalMultiLabel(
    AmixCrossAttentionClickFixedTotalMultiLabel
):
    """The click model, with Anatomix embeddings cached across the rounds of
    one iterative-training step.

    A compute-only variant of `AmixCrossAttentionClickFixedTotalMultiLabel`:
    **no new parameters**, so its `state_dict` is identical to the parent's
    and a parent checkpoint loads with `strict=True` (and vice versa). The
    class name ends in `MultiLabel`, so `experiment/amix_cross_v3.py`'s
    `_select_experiment_class` picks the multi-label runner, and eval
    wrappers that instantiate the run-dir `config.yml`'s `_class` work
    unchanged.

    Why it exists. Iterative interactive training (`train.prompt_iter > 1`)
    runs several forwards per step over the SAME target and context volumes;
    only the click indices (and the freshly sampled context-token positions)
    change between rounds. The expensive part of each forward is the Anatomix
    U-Net over the 128³ target and each context volume (`_embed`), and its
    inputs do not change between rounds. So, inside an explicit EPISODE,
    `_embed` memoizes its output per input volume.

    What is cached — and what is deliberately not
    ---------------------------------------------
    ONLY `_embed` outputs: the target embedding (`_predict_one_multilabel`,
    both the click path and the no-click delegation to the parent) and each
    context volume's embedding (`_sample_context_prompts_multilabel`, called
    with `support_vols[n].unsqueeze(0)`). The context BANK — the sampled
    per-class token positions gathered from those embeddings — is NOT cached:
    every forward reruns `_sample_context_prompts_multilabel`, consumes the
    same RNG stream as the uncached model, and builds a fresh `(cls_kvs,
    cls_mask)`. Round-to-round context draws therefore advance exactly as they
    would on the uncached class, and a reseeded val forward reproduces its
    banks and logits bit for bit. (A bank-cached / frozen-draw variant would
    change the training semantics and belongs to a separate, documented arm.)
    This relies on the configured Anatomix embedding path being deterministic
    and consuming no RNG (no dropout, no random sampling) — true of the
    U-Net backbone; asserted by `tests/test_amix_click_cached_embeddings.py`.

    Scope: explicit, off by default
    -------------------------------
    Caching happens only between `begin_cached_episode()` and
    `end_cached_episode()` (or inside the `cached_episode()` context manager).
    **Outside an episode every call is the parent's code path, bit for bit** —
    no lookups, nothing stored — so the viz callback, the offline evaluator and
    a `prompt_iter: 1` run see no change. The runner opens one episode per
    batch (`_run_step`) when it iterates. Nested episodes are refused.

    Cache identity
    --------------
    The key is `(vol.data_ptr(), shape, stride, device, dtype, no_grad,
    normalize, grad_enabled)`. `data_ptr` is stable across rounds because
    `_ClickContextMixin.forward` re-slices the SAME `target_image` /
    `support_images` tensors each call (views share storage); shape + stride +
    device + dtype keep distinct layouts from aliasing, and distinct batch
    elements have distinct pointers. It is a deliberately narrow IDENTITY
    cache: a caller must not mutate a volume inside an episode (an in-place
    write would be served the stale embedding — unsupported, not detected).
    The cache is cleared at `end_cached_episode()`, including on an exception
    inside `cached_episode()`.

    Autograd contract (train)
    -------------------------
    With `freeze_anatomix: false`, grad-enabled embeddings retain the Anatomix graph.
    Accumulating callers must retain that graph until the final backward pass. No-grad
    rollout features use a separate cache entry: a final grad-enabled prediction recomputes
    features so its loss can train Anatomix. Only the earlier no-grad rounds share those
    detached features. The cache is discarded when the episode ends.
    """

    # Runner negotiation flag, same `getattr` pattern as `supports_click_indices`.
    caches_embeddings = True

    # None = outside an episode (the parent's path, nothing cached);
    # a dict = inside one. A plain attribute, not a buffer: never serialized.
    _embed_cache: Optional[Dict[tuple, torch.Tensor]] = None
    # True inside a `shared_grad` episode — see `begin_cached_episode`.
    _episode_shared_grad: bool = False

    @property
    def in_cached_episode(self) -> bool:
        return self._embed_cache is not None

    def begin_cached_episode(self, shared_grad: bool = False) -> None:
        """Start memoizing `_embed` per input volume. Refuses to nest.

        `shared_grad=True`: ONE grad-enabled Anatomix pass per volume for the
        whole episode, whatever grad mode each caller is in. A caller under
        `torch.no_grad()` is handed a detached VIEW of that embedding (same
        storage, no graph); a grad-enabled caller gets the embedding itself,
        graph attached. This is the shortcut for a final-only iterative step
        (`train.backprop_on_last_prompt_only` + `train.shared_encoder_graph`
        in `experiment/amix_cross_v3.py`): no weight changes between the
        no-grad rollout round the correction clicks are drawn from and the
        final round that learns, so the final round need not run Anatomix
        again.

        What it changes, and what it does not. The FINAL round's embeddings
        are bit-identical to the recompute's: both come from the grad-enabled
        graph. The ROLLOUT round's are not, on a compiled backbone: the
        recompute embeds the rollout under no-grad, this mode hands it the
        grad-enabled embedding, and those two differ at rounding level (see
        "Grad mode changes the bits" in `_AmixCrossMixin._embed`). The rollout
        logits therefore move slightly (<= 0.07 measured), which can flip an
        argmax between near-tied classes and so change WHICH correction clicks
        are drawn. Whenever the clicks come out the same, the loss and the
        gradients equal the recompute's exactly; when they differ, the
        gradient cosine was 0.9996-0.99999 (12 batches, untrained head, the
        worst case for ties). Uncompiled, the two modes are bitwise equal —
        which is what the CPU tests assert. The graph is built on
        the FIRST call and lives as long as the episode holds the embedding,
        which is the memory cost; the caller must backward exactly once, with
        no `retain_graph`. Only `no_grad=False` calls take this path — a
        `no_grad=True` call (`detach_context_features`) never carries a graph
        and keeps the plain cache — and grad mode is dropped from the key for
        them, which is what lets the two rounds share one entry. Train only by
        design: opened at eval it would build graphs nobody frees.
        """
        if self._embed_cache is not None:
            raise RuntimeError(
                f"{type(self).__name__}: begin_cached_episode() called inside "
                "an open episode — episodes do not nest; call "
                "end_cached_episode() first."
            )
        self._embed_cache = {}
        self._episode_shared_grad = bool(shared_grad)

    def end_cached_episode(self) -> None:
        """Drop every cached embedding and return to the parent's path."""
        self._embed_cache = None
        self._episode_shared_grad = False

    @contextlib.contextmanager
    def cached_episode(self, shared_grad: bool = False):
        """`with model.cached_episode(): ...` — cleared on exit, exception or not."""
        self.begin_cached_episode(shared_grad=shared_grad)
        try:
            yield self
        finally:
            self.end_cached_episode()

    @staticmethod
    def _embed_cache_key(vol: torch.Tensor, no_grad: bool, normalize: bool) -> Tuple[object, ...]:
        """
        Separate no-grad rollout features from features needed for a later backward pass.
        """
        return (
            int(vol.data_ptr()),
            tuple(int(s) for s in vol.shape),
            tuple(int(s) for s in vol.stride()),
            str(vol.device),
            vol.dtype,
            bool(no_grad),
            bool(normalize),
            torch.is_grad_enabled(),
        )

    def _embed(
        self, vol: torch.Tensor, *, no_grad: bool = False, normalize: bool = True
    ) -> torch.Tensor:
        if self._embed_cache is None:
            return super()._embed(vol, no_grad=no_grad, normalize=normalize)
        if self._episode_shared_grad and not no_grad:
            # One grad-enabled pass, shared across grad modes: the last key
            # field (`torch.is_grad_enabled()`) is replaced so a no-grad
            # caller and a grad-enabled caller land on the same entry.
            key = self._embed_cache_key(vol, no_grad, normalize)[:-1] + ("shared",)
            emb = self._embed_cache.get(key)
            if emb is None:
                with torch.enable_grad():
                    emb = super()._embed(vol, no_grad=False, normalize=normalize)
                self._embed_cache[key] = emb
            return emb if torch.is_grad_enabled() else emb.detach()
        key = self._embed_cache_key(vol, no_grad, normalize)
        emb = self._embed_cache.get(key)
        if emb is None:
            emb = super()._embed(vol, no_grad=no_grad, normalize=normalize)
            self._embed_cache[key] = emb
        return emb
