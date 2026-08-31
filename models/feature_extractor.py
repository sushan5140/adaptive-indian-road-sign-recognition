"""Feature extraction using a timm image backbone."""

from typing import Final

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureExtractor(nn.Module):
    """Create embeddings from images using a classifier-free timm backbone.

    Args:
        backbone_name: A model name accepted by :func:`timm.create_model`.
        pretrained: Whether timm should load pretrained weights. This can trigger a
            network download and therefore defaults to ``False``.
        normalize_embeddings: Whether to L2-normalize each output embedding.
    """

    DEFAULT_BACKBONE: Final[str] = "mobilenetv3_small_100"

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        *,
        pretrained: bool = False,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if not backbone_name.strip():
            raise ValueError("backbone_name must be a non-empty string")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        num_features = getattr(self.backbone, "num_features", None)
        if not isinstance(num_features, int) or num_features <= 0:
            raise RuntimeError(
                f"Backbone {backbone_name!r} does not expose a valid num_features"
            )
        # MobileNetV3 applies its optional convolutional head even when timm's
        # classifier is disabled. Recent timm versions therefore return
        # ``head_hidden_size`` features while ``num_features`` still describes
        # the tensor entering that head.
        head_hidden_size = getattr(self.backbone, "head_hidden_size", None)
        self.embedding_dim = (
            head_hidden_size
            if isinstance(head_hidden_size, int) and head_hidden_size > 0
            else num_features
        )
        self.normalize_embeddings = normalize_embeddings

    def forward(self, images: Tensor) -> Tensor:
        """Return one embedding per image in an ``NCHW`` batch."""
        if images.ndim != 4:
            raise ValueError(
                f"Expected an NCHW image batch with 4 dimensions, got {images.ndim}"
            )
        embeddings = self.backbone(images)
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise RuntimeError("Backbone did not return a two-dimensional tensor")
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
