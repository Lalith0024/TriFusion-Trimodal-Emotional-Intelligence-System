"""
src/utils/redis_client.py
─────────────────────────
Redis client for session history persistence.

Used to store EmotionFrame snapshots per session_id so the
Session History dashboard page can display historical data
without re-running inference.

If Redis is unavailable (e.g., running dashboard without Docker),
all operations fall back to in-memory dict storage with a warning.
This ensures the dashboard never crashes due to missing Redis.
"""

import os
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_redis_client_cached = None
_redis_checked = False
_memory_store = {}

def _get_redis_client():
    """
    Attempt to create a Redis client with fast fallback caching.
    Returns None if Redis is unavailable.
    """
    global _redis_client_cached, _redis_checked
    if _redis_checked:
        return _redis_client_cached

    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        client = redis.from_url(url, decode_responses=True, socket_connect_timeout=0.2)
        client.ping()  # fast connectivity check
        _redis_client_cached = client
    except Exception as e:
        logger.info(f"Redis unavailable ({e}) — using high-performance in-memory fallback.")
        _redis_client_cached = None
    finally:
        _redis_checked = True

    return _redis_client_cached



def save_frame(session_id: str, frame_dict: dict, ttl_seconds: int = 3600) -> None:
    """
    Append an EmotionFrame dict to the session's Redis list.

    Args:
        session_id:   Unique session identifier.
        frame_dict:   Serialisable dict representation of EmotionFrame.
        ttl_seconds:  Key expiry — defaults to 1 hour.
    """
    key    = f"trifusion:session:{session_id}:frames"
    client = _get_redis_client()

    if client:
        try:
            client.rpush(key, json.dumps(frame_dict))
            client.expire(key, ttl_seconds)
            return
        except Exception as e:
            logger.error(f"Redis write failed: {e}")

    # In-memory fallback
    if key not in _memory_store:
        _memory_store[key] = []
    _memory_store[key].append(frame_dict)


def get_frames(session_id: str) -> List[dict]:
    """
    Retrieve all stored EmotionFrame dicts for a session.

    Returns:
        List of frame dicts (may be empty if no session data exists).
    """
    key    = f"trifusion:session:{session_id}:frames"
    client = _get_redis_client()

    if client:
        try:
            raw = client.lrange(key, 0, -1)
            return [json.loads(r) for r in raw]
        except Exception as e:
            logger.error(f"Redis read failed: {e}")

    return _memory_store.get(key, [])


def clear_session(session_id: str) -> None:
    """Delete all stored frames for a session."""
    key    = f"trifusion:session:{session_id}:frames"
    client = _get_redis_client()

    if client:
        try:
            client.delete(key)
            return
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")

    _memory_store.pop(key, None)
