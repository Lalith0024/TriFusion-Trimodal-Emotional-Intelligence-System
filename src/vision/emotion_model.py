"""
src/vision/emotion_model.py
───────────────────────────
EfficientNet-B0 fine-tuned head for facial emotion recognition.

Architecture rationale:
  • EfficientNet-B0 is the smallest/fastest member of the family,
    giving a good trade-off between accuracy and inference speed for
    real-time webcam use (~30 ms on CPU per frame).
  • We replace the stock single-Linear classifier with a two-layer MLP
    (Linear → ReLU → Dropout → Linear) which gives the model more
    capacity to discriminate subtle expression differences.
  • Label smoothing (0.1) during training mitigates FER2013's ~20 %
    label noise, reducing overconfidence.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from config.emotions import FER2013_NUM_CLASSES


class FacialEmotionNet(nn.Module):
    """
    EfficientNet-B0 backbone with a custom two-layer classification head.

    Input tensor shape:  (B, 3, 224, 224)  — ImageNet-normalized RGB
    Output tensor shape: (B, num_classes)   — raw logits (use softmax externally)
    """

    def __init__(self, num_classes: int = FER2013_NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Fetch the feature dimension from the original classifier head
        in_features = self.backbone.classifier[1].in_features

        # Two-stage head: coarse → fine discrimination
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),   # regularise high-level features
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)         # final logit layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits. Apply softmax for probabilities."""
        return self.backbone(x)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns a probability distribution."""
        return torch.softmax(self.forward(x), dim=-1)


# ---------------------------------------------------------------------------
# Data augmentation / normalisation transforms
# ---------------------------------------------------------------------------

def get_transforms(train: bool = True) -> T.Compose:
    """
    Returns ImageNet-normalised transforms.

    Training pipeline includes:
      • Random horizontal flip  — faces are horizontally symmetric
      • Colour jitter           — handles lighting variation across webcams
      • Random rotation (±10°) — slight head tilt is common in FER2013
    """
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
