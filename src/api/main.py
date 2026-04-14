"""
src/api/main.py
───────────────
FastAPI application entry point.

Startup behaviour:
  • CORS is fully open (allow_origins=["*"]) — restrict in production.
  • Routes are mounted at /api/v1 for versioning.
  • /health endpoint is mounted at root level for load-balancer checks.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from dotenv import load_dotenv

load_dotenv()   # load .env before any module reads os.getenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="TriFusion API",
    description=(
        "Trimodal Emotional Intelligence System — real-time emotion detection "
        "and LangGraph-powered wellness interventions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow all origins for development — tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount versioned routes
app.include_router(router, prefix="/api/v1", tags=["TriFusion"])


@app.get("/health", tags=["System"])
def health_check():
    """Root-level health endpoint for Docker / load-balancer liveness probes."""
    return {"status": "healthy", "service": "trifusion", "version": "1.0.0"}
