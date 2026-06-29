"""Model wrapper for the KIND-RIVER gradient-accumulation checkpoint."""

from collections.abc import Callable, Sequence
from typing import Literal

import neurite as ne
import torch
from torch import nn


class SegModel(ne.nn.models.BasicUNet):
    """Neurite BasicUNet architecture used by the KIND-RIVER checkpoint.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    in_channels : int
        Number of input channels. FreeBrowse supplies image, previous logits, positive prompts,
        negative prompts, and one unused channel.
    out_channels : int
        Number of output channels.
    nb_features : Sequence[int] or Sequence[Sequence[int]]
        Feature counts for each U-Net level. The first level is `5` for this checkpoint.
    padding_mode : {'zeros', 'replicate', 'reflect'}
        Padding mode used by the convolution blocks.
    upsample_mode : {'linear', 'transposed', 'nearest'}
        Upsampling mode used by the decoder.
    normalizations : list[Callable or str] or Callable or str or None
        Normalization configuration passed through to `BasicUNet`.
    activations : list[Callable or str] or Callable or str or None
        Activation configuration passed through to `BasicUNet`.
    order : str
        Layer ordering passed through to `BasicUNet`.
    final_activation : str or torch.nn.Module or None
        Optional final activation. Checkpoints store logits, so this defaults to `None`.
    skip_connections : bool
        Whether to use U-Net skip connections.
    """

    def __init__(
        self,
        ndim: int = 3,
        in_channels: int = 5,
        out_channels: int = 1,
        nb_features: Sequence[int] | Sequence[Sequence[int]] = (5, 16, 64, 256, 256),
        padding_mode: Literal["zeros", "replicate", "reflect"] = "zeros",
        upsample_mode: Literal["linear", "transposed", "nearest"] = "linear",
        normalizations: list[Callable | str] | Callable | str | None = None,
        activations: list[Callable | str] | Callable | str | None = nn.ReLU,
        order: str = "ca",
        final_activation: str | nn.Module | None = None,
        skip_connections: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            nb_features=nb_features,
            padding_mode=padding_mode,
            upsample_mode=upsample_mode,
            normalizations=normalizations,
            activations=activations,
            order=order,
            final_activation=final_activation,
            skip_connections=skip_connections,
            *args,
            **kwargs,
        )

    @property
    def device(self) -> torch.device:
        """torch.device: Device that owns the model parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        target_image: torch.Tensor,
        support_images: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run segmentation inference.

        Parameters
        ----------
        target_image : torch.Tensor
            Input tensor with shape `(B, 5, *spatial)`.
        support_images : torch.Tensor or None
            Unused compatibility argument.
        support_labels : torch.Tensor or None
            Unused compatibility argument.

        Returns
        -------
        torch.Tensor
            Logit tensor with shape `(B, 1, *spatial)`.
        """
        return super().forward(target_image)
