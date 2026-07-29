"""
src/api/main.py
───────────────
FastAPI application entry point.

Startup behaviour:
  • CORS is fully open (allow_origins=["*"]) — restrict in production.
  • Routes are mounted at /api/v1 for versioning.
  • /health endpoint is mounted at root level for load-balancer checks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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

@app.on_event("startup")
def check_resources():
    try:
        import psutil
        mem = psutil.virtual_memory()
        gb = mem.available / (1024 ** 3)
        if gb < 2.0:
            logging.error(f"Insufficient RAM: {gb:.2f}GB available. TriFusion requires at least 2GB free. Forcing SIMULATION_MODE.")
            os.environ["SIMULATION_MODE"] = "true"
    except ImportError:
        pass


@app.get("/health", tags=["System"])
def health_check():
    """Root-level health endpoint for Docker / load-balancer liveness probes."""
    from src.pipeline.startup_validator import validate_checkpoints
    
    checkpoints_ok, missing = validate_checkpoints()
    simulation_mode = os.environ.get("SIMULATION_MODE", str("STREAMLIT_SERVER_PORT" in os.environ)).lower() in ("true", "1", "t")
    
    return {
        "status": "healthy" if checkpoints_ok or simulation_mode else "degraded",
        "service": "trifusion",
        "version": "1.0.0",
        "checkpoints_loaded": checkpoints_ok,
        "missing_checkpoints": missing,
        "simulation_mode": simulation_mode
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
