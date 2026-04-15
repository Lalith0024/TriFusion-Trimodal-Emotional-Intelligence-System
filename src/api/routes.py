"""
src/api/routes.py
─────────────────
FastAPI route handlers for TriFusion API.

Endpoints:
  POST /api/v1/analyze   — Accept text input, run text inference + agent, return result.
  GET  /api/v1/history   — Return session emotion history from Redis.
  GET  /api/v1/incongruence — Return current incongruence score.

The vision and audio modalities are NOT called here (they run in the
Streamlit frontend process which has direct webcam/mic access).
The API handles text-based inference and agent orchestration for external
clients (e.g., mobile apps, REST integrations).

For full trimodal analysis, use the Streamlit dashboard directly.
"""

import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from src.api.schemas import AnalyzeRequest, AgentResponse, FusionOutput, ModalityOutput
from src.text.inference import TextInference
from src.agent.state import AgentState, EmotionFrame, ModalityResult
from src.agent.graph import wellness_agent
from config.emotions import UNIFIED_EMOTIONS
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-initialise text inference (avoids loading model at import time)
_text_inference: TextInference = None


def get_text_inference() -> TextInference:
    global _text_inference
    if _text_inference is None:
        model_path = os.getenv("TEXT_MODEL_PATH")
        _text_inference = TextInference(model_path=model_path)
    return _text_inference


@router.post("/analyze", response_model=AgentResponse)
def analyze(request: AnalyzeRequest):
    """
    Run text emotion inference + WellnessAgent on a text input.
    Returns the fused emotion analysis and wellness intervention.

    Note: Vision and audio modalities default to uniform distributions
    when called via REST (no webcam/mic access in API context).
    """
    try:
        text_result = get_text_inference().predict(request.text)
    except Exception as e:
        logger.error(f"Text inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text inference error: {str(e)}")

    # Use uniform distribution for vision/audio when called via REST
    uniform = {e: 1.0 / 8 for e in UNIFIED_EMOTIONS}
    vision_result = {"probabilities": uniform, "dominant": "neutral", "confidence": 0.0}
    audio_result  = {"probabilities": uniform, "dominant": "neutral", "confidence": 0.0}

    # Build EmotionFrame for agent
    frame = EmotionFrame(
        dominant_emotion=text_result["dominant"],
        confidence=text_result["confidence"],
        incongruence_score=0.0,   # no multi-modal data via REST
        vision=ModalityResult(dominant="neutral", confidence=0.0, probabilities=uniform),
        audio=ModalityResult(dominant="neutral",  confidence=0.0, probabilities=uniform),
        text=ModalityResult(
            dominant=text_result["dominant"],
            confidence=text_result["confidence"],
            probabilities=text_result["probabilities"]
        ),
        user_text=request.text
    )

    # Run WellnessAgent
    try:
        initial_state = AgentState(current_frame=frame)
        final_state   = wellness_agent.invoke(initial_state)
    except Exception as e:
        logger.error(f"WellnessAgent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Build response
    modality_out = ModalityOutput(
        dominant=text_result["dominant"],
        confidence=text_result["confidence"],
        probabilities=text_result["probabilities"]
    )
    uniform_out = ModalityOutput(dominant="neutral", confidence=0.0, probabilities=uniform)

    fusion_out = FusionOutput(
        dominant_emotion=text_result["dominant"],
        confidence=text_result["confidence"],
        fused_probabilities=text_result["probabilities"],
        incongruence_score=0.0,
        incongruence_label="N/A (text-only)",
        vision=uniform_out,
        audio=uniform_out,
        text=modality_out
    )

    return AgentResponse(
        emotion_analysis=fusion_out,
        agent_response=final_state.agent_response,
        intervention_type=final_state.last_intervention or "none",
        escalated=final_state.escalated,
        session_id=request.session_id
    )


@router.get("/health")
def api_health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
