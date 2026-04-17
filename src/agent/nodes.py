"""
src/agent/nodes.py
──────────────────
LangGraph node functions for WellnessAgent.

Each function:
  • Accepts the full AgentState.
  • Performs its specific task (route decision, tool call, memory write).
  • Returns the mutated AgentState.

Node responsibilities:
  emotion_router        → decides routing path based on severity + incongruence
  intervention_planner  → selects which wellness tool to use
  breathing_node        → calls breathing_exercise tool
  grounding_node        → calls grounding_technique tool
  affirmation_node      → calls affirmation_generator tool
  cognitive_reframe_node→ calls cognitive_reframe tool
  music_node            → calls music_recommendation tool
  reinforce_node        → calls positive_reinforcement tool
  escalation_node       → delivers crisis resources
  memory_writer         → appends current_frame to session_history
  idle_node             → handles the no-face / no-audio startup state
"""

import logging
from src.agent.state import AgentState
from src.agent.tools import (
    breathing_exercise,
    grounding_technique,
    affirmation_generator,
    cognitive_reframe,
    music_recommendation,
    positive_reinforcement,
    escalation_response
)
from config.emotions import EMOTION_TO_INTERVENTION, EMOTION_SEVERITY

logger = logging.getLogger(__name__)

# Mirror config thresholds here for clarity
_INCONGRUENCE_HIGH       = 0.7
_ESCALATION_THRESHOLD    = 3   # consecutive high-incongruence frames
_MAX_INTERVENTIONS       = 5   # stop generating after this many per session


# ---------------------------------------------------------------------------
# Routing node
# ---------------------------------------------------------------------------

def emotion_router(state: AgentState) -> AgentState:
    """
    Decide the routing path based on:
      1. Escalation: 3+ consecutive high-incongruence frames AND severity ≥ 2.
      2. Intervention: severity ≥ 2 OR incongruence is high.
      3. Reinforce:  emotion is happy or calm.
      4. Neutral:    everything else → affirmation (low-stakes nudge).
      5. Idle:       no current_frame (session not started).
    """
    if state.current_frame is None:
        state.routing_decision = "idle"
        return state

    emotion      = state.current_frame.dominant_emotion
    incongruence = state.current_frame.incongruence_score
    severity     = EMOTION_SEVERITY.get(emotion, 0)

    # Track consecutive high-incongruence frames
    if incongruence >= _INCONGRUENCE_HIGH:
        state.consecutive_high_incongruence += 1
    else:
        state.consecutive_high_incongruence = 0  # reset streak

    # Hard cap — don't spam interventions
    if state.intervention_count >= _MAX_INTERVENTIONS:
        state.routing_decision = "idle"
        return state

    # Priority order: escalation → intervention → reinforce → neutral
    if state.consecutive_high_incongruence >= _ESCALATION_THRESHOLD and severity >= 2:
        state.routing_decision = "escalate"
    elif severity >= 2 or incongruence >= _INCONGRUENCE_HIGH:
        state.routing_decision = "intervene"
    elif emotion in ("happy", "calm"):
        state.routing_decision = "reinforce"
    else:
        state.routing_decision = "neutral"

    logger.info(
        f"Router: {emotion} | severity={severity} | incongruence={incongruence:.2f} "
        f"| consecutive={state.consecutive_high_incongruence} "
        f"→ {state.routing_decision}"
    )
    return state


# ---------------------------------------------------------------------------
# Planner node — selects specific tool when routing_decision == "intervene"
# ---------------------------------------------------------------------------

def intervention_planner(state: AgentState) -> AgentState:
    """
    Choose which wellness tool to use, avoiding repetition.
    Sets routing_decision to the tool name string used by graph edges.
    """
    emotion = state.current_frame.dominant_emotion if state.current_frame else "neutral"
    preferred = EMOTION_TO_INTERVENTION.get(emotion, "affirmation_generator")

    # Rotate away from last-used tool to avoid identical consecutive responses
    if preferred == state.last_intervention:
        all_tools = list(set(EMOTION_TO_INTERVENTION.values()))
        alternatives = [t for t in all_tools if t != preferred]
        if alternatives:
            # Pick the first alternative alphabetically for determinism
            preferred = sorted(alternatives)[0]

    state.routing_decision = preferred
    logger.info(f"Planner selected: {preferred} (emotion={emotion})")
    return state


# ---------------------------------------------------------------------------
# Leaf tool nodes — each calls one specific wellness tool
# ---------------------------------------------------------------------------

def breathing_node(state: AgentState) -> AgentState:
    emotion = state.current_frame.dominant_emotion if state.current_frame else "stressed"
    state.agent_response  = breathing_exercise(emotion)
    state.last_intervention = "breathing_exercise"
    state.intervention_count += 1
    return state


def grounding_node(state: AgentState) -> AgentState:
    emotion = state.current_frame.dominant_emotion if state.current_frame else "anxious"
    state.agent_response  = grounding_technique(emotion)
    state.last_intervention = "grounding_technique"
    state.intervention_count += 1
    return state


def affirmation_node(state: AgentState) -> AgentState:
    emotion = state.current_frame.dominant_emotion if state.current_frame else "sad"
    state.agent_response  = affirmation_generator(emotion)
    state.last_intervention = "affirmation_generator"
    state.intervention_count += 1
    return state


def cognitive_reframe_node(state: AgentState) -> AgentState:
    frame     = state.current_frame
    emotion   = frame.dominant_emotion if frame else "neutral"
    user_text = frame.user_text        if frame else ""
    state.agent_response  = cognitive_reframe(emotion, user_text)
    state.last_intervention = "cognitive_reframe"
    state.intervention_count += 1
    return state


def music_node(state: AgentState) -> AgentState:
    emotion = state.current_frame.dominant_emotion if state.current_frame else "neutral"
    state.agent_response  = music_recommendation(emotion)
    state.last_intervention = "music_recommendation"
    state.intervention_count += 1
    return state


def reinforce_node(state: AgentState) -> AgentState:
    emotion = state.current_frame.dominant_emotion if state.current_frame else "happy"
    state.agent_response  = positive_reinforcement(emotion)
    state.last_intervention = "positive_reinforcement"
    # Positive reinforcement is not counted as an "intervention" (non-intrusive)
    return state


def escalation_node(state: AgentState) -> AgentState:
    """Delivers crisis resources. Marks session as escalated (no further LLM calls)."""
    state.agent_response  = escalation_response()
    state.escalated         = True
    state.last_intervention = "escalation"
    return state


# ---------------------------------------------------------------------------
# Memory and idle nodes
# ---------------------------------------------------------------------------

def memory_writer(state: AgentState) -> AgentState:
    """Persist the current frame to the session timeline."""
    if state.current_frame:
        state.session_history.append(state.current_frame)
    return state


def idle_node(state: AgentState) -> AgentState:
    """Friendly prompt shown when the session hasn't started or is capped."""
    state.agent_response = (
        "I'm here whenever you're ready. "
        "Position yourself in front of the camera and start speaking — "
        "I'll begin reading your emotional signals automatically."
    )
    return state
