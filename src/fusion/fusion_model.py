"""
src/fusion/fusion_model.py
──────────────────────────
Late-fusion MLP that combines the three modality probability distributions.

Input dimensionality:
  • Vision probs:  7  (FER2013 classes — no "calm")
  • Audio probs:   8  (RAVDESS classes)
  • Text probs:    8  (unified GoEmotions classes)
  Total input:    23 dims

Why late fusion (probabilities) vs early/intermediate fusion (features)?
  • Late fusion is modality-agnostic — the fusion layer only sees
    normalised distributions, not raw embeddings from different spaces.
  • It naturally handles missing modalities: uniform distribution = "no info".
  • Simpler to train: can be supervised with synthetic fusion labels without
    full end-to-end back-prop through all three backbone models.

Architecture: two-layer MLP with BatchNorm after the first layer.
BatchNorm is important here because the three probability vectors are on
different scales (vision 7-class vs audio/text 8-class) before concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionMLP(nn.Module):
    """
    Late-fusion MLP for trimodal emotion integration.

    Input:  concatenation of [vision(7), audio(8), text(8)] = 23-dim vector
    Output: 8-class unified probability distribution
    """

    def __init__(self, input_dim: int = 23, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),          # normalise across modalities
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)    # raw logits — softmax applied in forward
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init — good default for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        vision_probs: torch.Tensor,   # (B, 7)
        audio_probs:  torch.Tensor,   # (B, 8)
        text_probs:   torch.Tensor    # (B, 8)
    ) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, 8) — softmax probability distribution.
        """
        x      = torch.cat([vision_probs, audio_probs, text_probs], dim=-1)
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
