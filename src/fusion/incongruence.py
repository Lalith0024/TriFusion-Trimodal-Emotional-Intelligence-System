"""
src/fusion/incongruence.py
──────────────────────────
KL-divergence based incongruence scoring.

What is incongruence?
  When a person's facial expression, voice tone, and words tell three
  different emotional stories, it often indicates:
    • Emotional masking / suppression
    • Social desirability bias ("I'm fine" when not)
    • Genuine emotional ambiguity / complexity

How we measure it:
  We compute the average pairwise *symmetric KL divergence* between
  the three modality distributions:
    KL_sym(P, Q) = [KL(P||Q) + KL(Q||P)] / 2

  This is symmetric (KL itself is not) and more numerically stable.
  We average three pairs: (vision,audio), (vision,text), (audio,text).

Normalisation:
  Raw KL can technically be unbounded, but for 8-class distributions the
  practical maximum is log(8) ≈ 2.08 nats (when one distribution is
  perfectly concentrated and the other is uniform).
  We divide by 2.08 and clamp to [0, 1] for display.

Interpretation:
  0.0 – 0.3 → Aligned      (signals agree)
  0.3 – 0.7 → Moderate     (some disagreement — normal variation)
  0.7 – 1.0 → High         (likely masking — WellnessAgent escalates)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from config.emotions import UNIFIED_EMOTIONS

# Maximum possible average symmetric KL for 8-class distributions (nats)
_MAX_KL_8CLASS = 2.08


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Symmetric KL divergence: [KL(P||Q) + KL(Q||P)] / 2.

    Args:
        p, q: Probability tensors of same shape. Must sum to 1 and be > 0.

    Returns:
        Scalar float.
    """
    # F.kl_div expects log-probabilities as first arg and probabilities as second
    kl_pq = F.kl_div(q.log(), p, reduction="sum")  # KL(P || Q)
    kl_qp = F.kl_div(p.log(), q, reduction="sum")  # KL(Q || P)
    return ((kl_pq + kl_qp) / 2.0).item()


def compute_incongruence(
    vision_probs: Dict[str, float],
    audio_probs:  Dict[str, float],
    text_probs:   Dict[str, float]
) -> float:
    """
    Compute the normalised incongruence score across three modality distributions.

    Args:
        vision_probs: Dict mapping unified emotion → probability (vision).
        audio_probs:  Dict mapping unified emotion → probability (audio).
        text_probs:   Dict mapping unified emotion → probability (text).

    Returns:
        Float in [0.0, 1.0].
        0.0 = perfectly aligned; 1.0 = maximally incongruent.
    """
    eps = 1e-8  # small floor to avoid log(0)

    def to_tensor(d: Dict[str, float]) -> torch.Tensor:
        """Convert prob dict → float32 tensor ordered by UNIFIED_EMOTIONS."""
        t = torch.tensor(
            [d.get(e, eps) for e in UNIFIED_EMOTIONS], dtype=torch.float32
        )
        t = t + eps                  # ensure strictly positive
        return t / t.sum()           # normalise to sum-1

    p_v = to_tensor(vision_probs)
    p_a = to_tensor(audio_probs)
    p_t = to_tensor(text_probs)

    kl_va = _symmetric_kl(p_v, p_a)
    kl_vt = _symmetric_kl(p_v, p_t)
    kl_at = _symmetric_kl(p_a, p_t)

    avg_kl = (kl_va + kl_vt + kl_at) / 3.0

    # Clamp to [0, 1] after normalisation
    return float(min(avg_kl / _MAX_KL_8CLASS, 1.0))


def get_incongruence_label(score: float) -> Tuple[str, str]:
    """
    Convert a numeric incongruence score to a human-readable label and hex colour.

    Returns:
        (label_string, hex_colour)
    """
    if score < 0.3:
        return "ALIGNED", "#22c55e"        # green
    elif score < 0.7:
        return "MODERATE", "#f59e0b"       # amber
    else:
        return "HIGH — MASKING DETECTED", "#ef4444"   # red
