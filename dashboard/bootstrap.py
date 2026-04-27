"""
dashboard/bootstrap.py
──────────────────────
CRITICAL: This module MUST be imported at the very top of every dashboard
page before any project imports. It ensures the project root is on sys.path
so that `from src.X import Y` and `from config.X import Y` always work
regardless of where Streamlit launches from (local, Streamlit Cloud, Docker).
"""
import sys
import os

# Resolve the absolute path to the project root (parent of this file)
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_DASHBOARD_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
