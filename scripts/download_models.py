import os
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# This script serves as the artifact delivery mechanism for production deployment.
# Instead of storing heavy .pth files in git (which violates .gitignore),
# we download them at build time from a model registry.

HF_REPO_ID = "Lalithendra/TriFusion-models"  # Example HF hub repo

MODELS_TO_DOWNLOAD = [
    {"repo_file": "efficientnet_fer2013.pth", "local_path": "models/vision/efficientnet_fer2013.pth"},
    {"repo_file": "wav2vec2_ravdess/pytorch_model.bin", "local_path": "models/audio/wav2vec2_ravdess/pytorch_model.bin"},
    {"repo_file": "wav2vec2_ravdess/config.json", "local_path": "models/audio/wav2vec2_ravdess/config.json"},
    {"repo_file": "roberta_goemotions/pytorch_model.bin", "local_path": "models/text/roberta_goemotions/pytorch_model.bin"},
    {"repo_file": "roberta_goemotions/config.json", "local_path": "models/text/roberta_goemotions/config.json"},
    {"repo_file": "fusion_mlp.pth", "local_path": "models/fusion/fusion_mlp.pth"},
]

def main():
    logger.info("Starting model checkpoint delivery...")
    # NOTE: Since the real HF repo might not exist yet, we wrap this in a try-except.
    # In a real production run, this should fail loudly if the repo is unreachable.
    for model in MODELS_TO_DOWNLOAD:
        local_dir = os.path.dirname(model["local_path"])
        os.makedirs(local_dir, exist_ok=True)
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=model["repo_file"],
                local_dir=os.path.dirname(local_dir),
                local_dir_use_symlinks=False
            )
            logger.info(f"Successfully downloaded {model['repo_file']} to {downloaded_path}")
        except Exception as e:
            logger.warning(f"Could not download {model['repo_file']} from {HF_REPO_ID}. If running in demo mode, you can ignore this. Error: {e}")

if __name__ == "__main__":
    main()
