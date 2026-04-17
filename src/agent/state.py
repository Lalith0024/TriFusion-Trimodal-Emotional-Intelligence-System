"""
src/agent/state.py
──────────────────
Pydantic state schema for the WellnessAgent LangGraph.

LangGraph requires a typed state object that is passed between nodes.
Every node receives the full state, mutates specific fields, and returns
the updated state. Pydantic provides automatic validation and JSON serialisation.

Key design decisions:
  • EmotionFrame captures one complete trimodal analysis snapshot.
  • AgentState tracks the full session context across multiple frames.
  • session_history stores all frames so the dashboard can render timelines.
  • consecutive_high_incongruence is used by emotion_router to decide
    when to escalate (threshold: 3 consecutive high-incongruence frames).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModalityResult(BaseModel):
    """Per-modality emotion inference output."""
    dominant:      str
    confidence:    float
    probabilities: Dict[str, float]


class EmotionFrame(BaseModel):
    """
    A single snapshot of trimodal emotion analysis.
    One frame is produced per analysis cycle (~3 seconds in real-time).
    """
    timestamp:          str = Field(default_factory=lambda: datetime.now().isoformat())
    dominant_emotion:   str
    confidence:         float
    incongruence_score: float
    vision:             ModalityResult
    audio:              ModalityResult
    text:               ModalityResult
    # The verbatim text transcribed in this frame (from Whisper)
    user_text:          str = ""


class AgentState(BaseModel):
    """
    Full session state threaded through every LangGraph node.

    Fields:
        current_frame:                 Latest EmotionFrame from FusionInference.
        session_history:               All past EmotionFrames (for timeline chart).
        consecutive_high_incongruence: Counter reset each time score drops below 0.7.
        intervention_count:            Total interventions delivered this session.
        last_intervention:             Tool name of the last intervention delivered.
        escalated:                     True once crisis escalation has been triggered.
        agent_response:                Text output produced by the chosen tool node.
        routing_decision:              Internal routing signal between nodes.
        messages:                      LangChain-style message list for LLM context.
    """
    current_frame:                  Optional[EmotionFrame]   = None
    session_history:                List[EmotionFrame]       = Field(default_factory=list)
    consecutive_high_incongruence:  int                      = 0
    intervention_count:             int                      = 0
    last_intervention:              Optional[str]            = None
    escalated:                      bool                     = False
    agent_response:                 str                      = ""
    routing_decision:               Optional[str]            = None
    messages:                       List[dict]               = Field(default_factory=list)
