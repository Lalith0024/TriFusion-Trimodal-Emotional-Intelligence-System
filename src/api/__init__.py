# src/api/__init__.py
# FastAPI app factory — imported by uvicorn entry point.
from src.api.main import app

__all__ = ["app"]
