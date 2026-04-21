import pytest
from src.agent.state import AgentState, EmotionFrame, ModalityResult
from src.agent.nodes import emotion_router
from config.emotions import UNIFIED_EMOTIONS

def create_dummy_frame(dominant: str, incongruence: float):
    probs = {e: 0.0 for e in UNIFIED_EMOTIONS}
    probs[dominant] = 1.0
    return EmotionFrame(
        dominant_emotion=dominant,
        confidence=1.0,
        incongruence_score=incongruence,
        vision=ModalityResult(dominant=dominant, confidence=1.0, probabilities=probs),
        audio=ModalityResult(dominant=dominant, confidence=1.0, probabilities=probs),
        text=ModalityResult(dominant=dominant, confidence=1.0, probabilities=probs)
    )

def test_emotion_router_escalate():
    state = AgentState(consecutive_high_incongruence=2)
    state.current_frame = create_dummy_frame("fearful", 0.8) # Severity 3, Incongruence high
    new_state = emotion_router(state)
    assert new_state.routing_decision == "escalate"
    assert new_state.consecutive_high_incongruence == 3

def test_emotion_router_intervene_high_incongruence():
    state = AgentState(consecutive_high_incongruence=0)
    state.current_frame = create_dummy_frame("happy", 0.8) # Severity 0, Incongruence high
    new_state = emotion_router(state)
    assert new_state.routing_decision == "intervene"
    assert new_state.consecutive_high_incongruence == 1

def test_emotion_router_reinforce():
    state = AgentState(consecutive_high_incongruence=0)
    state.current_frame = create_dummy_frame("happy", 0.1) # Severity 0, Incongruence low
    new_state = emotion_router(state)
    assert new_state.routing_decision == "reinforce"

def test_emotion_router_idle_max_interventions():
    state = AgentState(consecutive_high_incongruence=0, intervention_count=5)
    state.current_frame = create_dummy_frame("fearful", 0.8)
    new_state = emotion_router(state)
    assert new_state.routing_decision == "idle"
