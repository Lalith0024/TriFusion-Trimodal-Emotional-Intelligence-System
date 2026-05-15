"""
data/download_datasets.py
─────────────────────────
Downloads and organises all three datasets required for TriFusion training.

  FER2013     → data/raw/fer2013/train/{class}/ and /test/{class}/
                ImageFolder-compatible for torchvision
  RAVDESS     → data/raw/ravdess/Actor_XX/*.wav
                Downloaded from Zenodo (official source, no account needed)
  GoEmotions  → downloaded automatically by HuggingFace datasets during
                training — no manual step needed

Usage:
    python data/download_datasets.py

Requirements:
    pip install datasets kagglehub  (already in requirements.txt)
"""

import os
import sys
import shutil
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)


# ── FER2013 ──────────────────────────────────────────────────────────────────

def download_fer2013():
    """
    Downloads FER2013 via HuggingFace datasets and organises it into an
    ImageFolder-compatible directory tree for torchvision.ImageFolder:
        data/raw/fer2013/train/{emotion}/XXXXXX.png
        data/raw/fer2013/test/{emotion}/XXXXXX.png
    """
    fer_dir = RAW / "fer2013"
    if (fer_dir / "train").exists() and any((fer_dir / "train").rglob("*.png")):
        logger.info("FER2013 already exists — skipping download.")
        return

    logger.info("Downloading FER2013 from HuggingFace Hub...")
    try:
        from datasets import load_dataset
        from PIL import Image
        import numpy as np

        # msambare/fer2013 is a popular, public version on HF Hub
        ds = load_dataset("msambare/fer2013")

        label_names = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]

        split_map = {
            "train": ds.get("train", ds[list(ds.keys())[0]]),
            "test":  ds.get("test",  ds.get("validation", ds[list(ds.keys())[-1]])),
        }

        for split_name, split_data in split_map.items():
            for label in label_names:
                (fer_dir / split_name / label).mkdir(parents=True, exist_ok=True)

            for i, example in enumerate(split_data):
                label_idx  = example["label"]
                label_name = label_names[label_idx]

                img = example["image"]
                if not hasattr(img, "save"):
                    img = Image.fromarray(np.array(img)).convert("RGB")
                else:
                    img = img.convert("RGB")

                img.save(fer_dir / split_name / label_name / f"{i:06d}.png")
                if i % 2000 == 0:
                    logger.info(f"  FER2013 {split_name}: {i} images written...")

        logger.info(f"✓ FER2013 ready at {fer_dir}")

    except Exception as e:
        logger.error(f"FER2013 HuggingFace download failed: {e}")
        logger.info("Trying kagglehub fallback (requires ~/.kaggle/kaggle.json)...")
        # Programmatic Kaggle fallback (requires kaggle API key in ~/.kaggle/kaggle.json)
        try:
            import kagglehub
            logger.info("Trying kagglehub fallback...")
            path = kagglehub.dataset_download("msambare/fer2013")
            # kagglehub downloads to a cache dir — copy to our expected location
            import shutil
            if os.path.exists(fer_dir):
                shutil.rmtree(fer_dir)
            logger.info("Kaggle download complete! Now copying 35,000+ images to data/raw/fer2013...")
            logger.info("This might take a minute. PLEASE DO NOT PRESS CTRL+C!")
            shutil.copytree(path, str(fer_dir), dirs_exist_ok=True)
            logger.info(f"✓ FER2013 ready at {fer_dir}")
        except Exception as ke:
            logger.error(f"kagglehub also failed: {ke}")
            logger.info("Trying direct download from stable mirror...")
            try:
                import urllib.request
                # Direct link to a zip mirror of FER2013
                url = "https://github.com/git-disl/FER2013-Dataset/raw/master/fer2013.zip"
                zip_path = RAW / "fer2013.zip"
                
                def _progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        pct = min(downloaded / total_size * 100, 100)
                        print(f"\r  Progress: {pct:5.1f}%", end="", flush=True)

                urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
                print()
                
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(fer_dir)
                zip_path.unlink()
                logger.info(f"✓ FER2013 ready at {fer_dir}")
            except Exception as de:
                logger.error(f"Direct download also failed: {de}")
                logger.info("MANUAL STEP REQUIRED: Download FER2013 from kaggle.com/datasets/msambare/fer2013")
                logger.info("Extract so: data/raw/fer2013/train/angry/*.png etc exist")


# ── RAVDESS ──────────────────────────────────────────────────────────────────

def download_ravdess():
    """
    Downloads RAVDESS speech audio from Zenodo (official repository).
    No account or API key needed — direct HTTPS download.
    Zenodo record: https://zenodo.org/record/1188976
    Produces: data/raw/ravdess/Actor_XX/*.wav  (24 actors, ~1,440 files)
    """
    ravdess_dir = RAW / "ravdess"
    if ravdess_dir.exists() and any(ravdess_dir.rglob("*.wav")):
        logger.info("RAVDESS already exists — skipping download.")
        return

    ravdess_dir.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request

        url      = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        zip_path = RAW / "ravdess_speech.zip"

        logger.info("Downloading RAVDESS from Zenodo (~215 MB)...")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r  Progress: {pct:5.1f}% ({downloaded/1e6:.0f}/{total_size/1e6:.0f} MB)",
                      end="", flush=True)

        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()  # newline after progress bar

        logger.info("Extracting RAVDESS...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ravdess_dir)

        zip_path.unlink()   # clean up zip after extraction

        wav_count = len(list(ravdess_dir.rglob("*.wav")))
        logger.info(f"✓ RAVDESS ready at {ravdess_dir} ({wav_count} .wav files)")

    except Exception as e:
        logger.error(f"RAVDESS Zenodo download failed: {e}")
        logger.info("Manual alternative:")
        logger.info("  Download: https://zenodo.org/record/1188976")
        logger.info("  Extract Actor_XX/ folders into data/raw/ravdess/")


# ── GoEmotions ───────────────────────────────────────────────────────────────

def check_goemotions():
    """
    GoEmotions is streamed directly from HuggingFace Hub during
    src/text/train_text.py — no manual download required.
    This function just verifies the datasets library is installed.
    """
    try:
        import datasets
        logger.info(
            f"✓ GoEmotions: HuggingFace datasets v{datasets.__version__} ready. "
            "Will auto-download during train_text.py."
        )
    except ImportError:
        logger.error("HuggingFace datasets not installed. Run: pip install datasets")


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_downloads():
    """Quick sanity check — report file counts per dataset."""
    checks = {
        "FER2013 train": list((RAW / "fer2013" / "train").rglob("*.png")),
        "FER2013 test":  list((RAW / "fer2013" / "test").rglob("*.png")),
        "RAVDESS .wav":  list((RAW / "ravdess").rglob("*.wav")),
    }
    logger.info("\n── Verification ──────────────────────────────")
    for name, files in checks.items():
        status = "✓" if files else "✗ NOT FOUND"
        logger.info(f"  {status}  {name}: {len(files)} files")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== TriFusion Dataset Downloader ===\n")
    download_fer2013()
    download_ravdess()
    check_goemotions()
    verify_downloads()
    logger.info("\n✓ Datasets ready.  Run training scripts in order:")
    logger.info("  1. python src/vision/train_vision.py   (~2h GPU / ~8h CPU)")
    logger.info("  2. python src/audio/train_audio.py     (~1h GPU)")
    logger.info("  3. python src/text/train_text.py       (~45 min GPU)")
    logger.info("  4. python src/fusion/train_fusion.py   (~10 min)")
    logger.info("  5. streamlit run dashboard/app.py")
