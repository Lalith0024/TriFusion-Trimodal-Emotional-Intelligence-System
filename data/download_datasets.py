"""
data/download_datasets.py
─────────────────────────
Downloads all three datasets required by TriFusion:
  • FER2013   — facial emotion dataset (Kaggle)
  • RAVDESS   — audio emotion dataset (Kaggle)
  • GoEmotions — text emotion dataset (HuggingFace)

Usage:
    python data/download_datasets.py

Requirements:
  • KAGGLE_USERNAME and KAGGLE_KEY set in .env (or kaggle.json placed at ~/.kaggle/)
  • HuggingFace Hub credentials (optional — GoEmotions is public)
"""

import os
import shutil
import zipfile
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_fer2013():
    """
    Download FER2013 via kagglehub.
    Produces: data/raw/fer2013/{train,test}/  with subdirs per emotion class.
    """
    logger.info("Downloading FER2013 dataset...")
    try:
        import kagglehub
        # Downloads to ~/.cache/kagglehub by default; we copy to data/raw
        path = kagglehub.dataset_download("msambare/fer2013")
        dest = RAW_DIR / "fer2013"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(path, dest)
        logger.info(f"FER2013 downloaded → {dest}")
    except Exception as e:
        logger.error(f"FER2013 download failed: {e}")
        logger.info("Manual alternative: kaggle datasets download -d msambare/fer2013 -p data/raw/")


def download_ravdess():
    """
    Download RAVDESS speech emotion dataset via kagglehub.
    Produces: data/raw/ravdess/ with actor folders Actor_01 … Actor_24
    """
    logger.info("Downloading RAVDESS dataset...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
        dest = RAW_DIR / "ravdess"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(path, dest)
        logger.info(f"RAVDESS downloaded → {dest}")
    except Exception as e:
        logger.error(f"RAVDESS download failed: {e}")
        logger.info("Manual alternative: kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p data/raw/")


def download_goemotions():
    """
    Download GoEmotions via HuggingFace datasets library.
    Saves raw train/validation/test splits to data/raw/goemotions/
    """
    logger.info("Downloading GoEmotions dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("go_emotions", "simplified")
        dest = RAW_DIR / "goemotions"
        dataset.save_to_disk(str(dest))
        logger.info(f"GoEmotions downloaded → {dest}")
    except Exception as e:
        logger.error(f"GoEmotions download failed: {e}")
        logger.info("Manual alternative: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('go_emotions')\"")


def verify_downloads():
    """Quick sanity check — list top-level contents of each dataset folder."""
    for name in ["fer2013", "ravdess", "goemotions"]:
        p = RAW_DIR / name
        if p.exists():
            entries = list(p.iterdir())[:5]
            logger.info(f"✓ {name}: {[e.name for e in entries]}")
        else:
            logger.warning(f"✗ {name}: NOT FOUND at {p}")


if __name__ == "__main__":
    download_fer2013()
    download_ravdess()
    download_goemotions()
    verify_downloads()
    logger.info("Dataset download complete. Run training scripts next.")
