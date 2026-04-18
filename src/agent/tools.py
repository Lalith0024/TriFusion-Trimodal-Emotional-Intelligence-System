"""
src/agent/tools.py
──────────────────
Wellness intervention tools for WellnessAgent.

Each tool is a thin wrapper that:
  1. Formats the appropriate prompt template with current emotional context.
  2. Calls the Groq LLM (LLaMA-3.3-70B via langchain-groq).
  3. Returns the raw string response.

Design notes:
  • get_llm() is called fresh inside each tool rather than at module import
    time. This avoids import errors when GROQ_API_KEY is not yet set
    (e.g., during testing without credentials).
  • All tools are plain functions (not LangChain @tool decorated) since
    the routing logic is handled entirely by LangGraph nodes — we don't
    need tool-calling / function-calling format here.
  • escalation_response() is the only tool that does NOT call the LLM —
    it returns the pre-written ESCALATION_MESSAGE directly for reliability.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.agent.prompts import (
    BREATHING_PROMPT,
    GROUNDING_PROMPT,
    AFFIRMATION_PROMPT,
    COGNITIVE_REFRAME_PROMPT,
    MUSIC_PROMPT,
    POSITIVE_REINFORCEMENT_PROMPT,
    ESCALATION_MESSAGE
)
import logging

logger = logging.getLogger(__name__)


def get_llm() -> ChatGroq:
    """
    Instantiate Groq LLM client.
    Raises ValueError if GROQ_API_KEY is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        max_tokens=512,
        temperature=0.7
    )


def _call_llm(prompt: str) -> str:
    """Shared LLM call with basic error handling."""
    try:
        llm = get_llm()
        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "I'm here with you. Take a deep breath — you've got this."


# ---------------------------------------------------------------------------
# Intervention tools
# ---------------------------------------------------------------------------

def breathing_exercise(emotion: str) -> str:
    """Guided breathing exercise tailored to the detected emotion."""
    return _call_llm(BREATHING_PROMPT.format(emotion=emotion))


def grounding_technique(emotion: str) -> str:
    """5-4-3-2-1 sensory grounding technique for agitation or anger."""
    return _call_llm(GROUNDING_PROMPT.format(emotion=emotion))


def affirmation_generator(emotion: str) -> str:
    """Three earned affirmations for sadness or low-energy states."""
    return _call_llm(AFFIRMATION_PROMPT.format(emotion=emotion))


def cognitive_reframe(emotion: str, user_text: str) -> str:
    """
    Gentle reframe of the user's spoken words.
    Passes the transcribed text so the LLM can respond to actual content.
    """
    return _call_llm(
        COGNITIVE_REFRAME_PROMPT.format(emotion=emotion, user_text=user_text or "(no speech detected)")
    )


def music_recommendation(current_emotion: str) -> str:
    """Three-track playlist to guide emotional regulation."""
    return _call_llm(MUSIC_PROMPT.format(current_emotion=current_emotion))


def positive_reinforcement(emotion: str) -> str:
    """Brief, warm celebration of a positive emotional state."""
    return _call_llm(POSITIVE_REINFORCEMENT_PROMPT.format(emotion=emotion))


def escalation_response() -> str:
    """
    Returns the pre-written crisis resource message.
    Does NOT call the LLM — reliability is critical here.
    """
    return ESCALATION_MESSAGE
