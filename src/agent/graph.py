"""
src/agent/graph.py
──────────────────
LangGraph graph definition for WellnessAgent.

Graph topology:
  emotion_router
    ├─ "escalate"            → escalation_node
    ├─ "intervene"           → intervention_planner
    │     ├─ breathing_exercise    → breathing_node
    │     ├─ grounding_technique   → grounding_node
    │     ├─ affirmation_generator → affirmation_node
    │     ├─ cognitive_reframe     → cognitive_reframe_node
    │     ├─ music_recommendation  → music_node
    │     └─ positive_reinforcement→ reinforce_node
    ├─ "reinforce"           → reinforce_node
    ├─ "neutral"             → affirmation_node
    └─ "idle"                → idle_node
  all terminal nodes → memory_writer → END

The compiled graph is exposed as the `wellness_agent` singleton, which is
imported by the Streamlit dashboard and FastAPI routes.

Note on LangGraph state: we pass AgentState as a Pydantic model.
LangGraph supports dict-based state natively; we convert in the invoke wrapper.
"""

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    emotion_router,
    intervention_planner,
    breathing_node,
    grounding_node,
    affirmation_node,
    cognitive_reframe_node,
    music_node,
    reinforce_node,
    escalation_node,
    memory_writer,
    idle_node,
)


def build_wellness_agent():
    """
    Construct and compile the WellnessAgent state graph.

    Returns:
        A compiled LangGraph CompiledGraph ready for .invoke() calls.
    """
    graph = StateGraph(AgentState)

    # ---- Register all nodes ------------------------------------------------
    graph.add_node("emotion_router",        emotion_router)
    graph.add_node("intervention_planner",  intervention_planner)
    graph.add_node("breathing",             breathing_node)
    graph.add_node("grounding",             grounding_node)
    graph.add_node("affirmation",           affirmation_node)
    graph.add_node("cognitive_reframe",     cognitive_reframe_node)
    graph.add_node("music",                 music_node)
    graph.add_node("reinforce",             reinforce_node)
    graph.add_node("escalation",            escalation_node)
    graph.add_node("memory_writer",         memory_writer)
    graph.add_node("idle",                  idle_node)

    # ---- Entry point -------------------------------------------------------
    graph.set_entry_point("emotion_router")

    # ---- Conditional edges from emotion_router ----------------------------
    def route_from_router(state: AgentState) -> str:
        decision = state.routing_decision
        if decision == "escalate":
            return "escalation"
        elif decision == "intervene":
            return "intervention_planner"
        elif decision == "reinforce":
            return "reinforce"
        elif decision == "idle":
            return "idle"
        else:
            # "neutral" and any fallback → affirmation
            return "affirmation"

    graph.add_conditional_edges(
        "emotion_router",
        route_from_router,
        {
            "escalation":           "escalation",
            "intervention_planner": "intervention_planner",
            "reinforce":            "reinforce",
            "idle":                 "idle",
            "affirmation":          "affirmation",
        }
    )

    # ---- Conditional edges from intervention_planner ----------------------
    def route_from_planner(state: AgentState) -> str:
        # routing_decision is now the tool name string
        mapping = {
            "breathing_exercise":    "breathing",
            "grounding_technique":   "grounding",
            "affirmation_generator": "affirmation",
            "cognitive_reframe":     "cognitive_reframe",
            "music_recommendation":  "music",
            "positive_reinforcement":"reinforce",
        }
        return mapping.get(state.routing_decision, "affirmation")

    graph.add_conditional_edges(
        "intervention_planner",
        route_from_planner,
        {
            "breathing":        "breathing",
            "grounding":        "grounding",
            "affirmation":      "affirmation",
            "cognitive_reframe":"cognitive_reframe",
            "music":            "music",
            "reinforce":        "reinforce",
        }
    )

    # ---- All leaf nodes → memory_writer → END ----------------------------
    for node_name in [
        "breathing", "grounding", "affirmation",
        "cognitive_reframe", "music", "reinforce",
        "escalation", "idle"
    ]:
        graph.add_edge(node_name, "memory_writer")

    graph.add_edge("memory_writer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
wellness_agent = build_wellness_agent()
