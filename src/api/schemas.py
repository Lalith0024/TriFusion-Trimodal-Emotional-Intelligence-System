"""
src/api/schemas.py
──────────────────
Pydantic request/response schemas for the TriFusion REST API.

Keeping schemas in a dedicated file (not in routes.py) ensures:
  • Single source of truth for API contract
  • Easy OpenAPI documentation generation
  • Re-usable types across multiple route files if the API grows
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """
    Minimal request body — text is the transcribed speech for this frame.
    session_id groups multiple frames into one user session.
    """
    text:       str = Field(..., description="Transcribed speech text for this analysis frame.")
    session_id: str = Field(default="default", description="Session identifier for history tracking.")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ModalityOutput(BaseModel):
    """Per-modality inference result returned by the API."""
    dominant:      str
    confidence:    float
    probabilities: Dict[str, float]


class FusionOutput(BaseModel):
    """Complete trimodal fusion result including incongruence metadata."""
    dominant_emotion:   str
    confidence:         float
    fused_probabilities:Dict[str, float]
    incongruence_score: float
    incongruence_label: str
    vision:             ModalityOutput
    audio:              ModalityOutput
    text:               ModalityOutput


class AgentResponse(BaseModel):
    """Full API response including emotion analysis and WellnessAgent output."""
    emotion_analysis:   FusionOutput
    agent_response:     str
    intervention_type:  str
    escalated:          bool
    session_id:         str


class HealthResponse(BaseModel):
    status:  str
    service: str
    version: str
