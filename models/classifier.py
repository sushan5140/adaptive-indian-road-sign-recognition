"""Supervised base classifier built on the shared feature extractor."""

from typing import cast

from torch import Tensor, nn

from models.feature_extractor import FeatureExtractor


class RoadSignClassifier(nn.Module):
    """Classify base road-sign classes from reusable image embeddings.

    Incremental classes are intentionally not added to this linear head; they are
    represented by a separate :class:`models.prototype_registry.PrototypeRegistry`.

    Args:
        num_classes: Number of classes in the immutable base training dataset.
        feature_extractor: Optional configured extractor to reuse.
        backbone_name: timm backbone used when an extractor is not supplied.
        pretrained: Whether timm may load pretrained weights.
        dropout: Dropout probability before the linear classification head.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        feature_extractor: FeatureExtractor | None = None,
        backbone_name: str = FeatureExtractor.DEFAULT_BACKBONE,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.feature_extractor = feature_extractor or FeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            normalize_embeddings=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(
            self.feature_extractor.embedding_dim,
            num_classes,
        )
        self.num_classes = num_classes

    def extract_embeddings(self, images: Tensor) -> Tensor:
        """Extract normalized embeddings without applying the classifier head."""
        return cast(Tensor, self.feature_extractor(images))

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalized base-class logits for an image batch."""
        embeddings = self.extract_embeddings(images)
        return cast(Tensor, self.classifier(self.dropout(embeddings)))
