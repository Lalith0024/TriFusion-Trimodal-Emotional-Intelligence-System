"""
main.py
TriFusion Orchestrator & Setup Script.
Run this single file to:
1. Verify datasets
2. Train all missing models automatically
3. Launch the Streamlit dashboard
"""
import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress annoying transformers warnings for a cleaner terminal
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

def run_command(command, desc):
    logger.info(f"==> {desc}")
    # We use source venv/bin/activate to ensure the virtual environment is used
    full_command = f"source venv/bin/activate && {command}" if os.path.exists("venv/bin/activate") else command
    
    result = subprocess.run(full_command, shell=True, executable="/bin/zsh")
    if result.returncode != 0:
        logger.error(f"Failed during: {desc}. Please check the terminal output above.")
        sys.exit(1)

def main():
    logger.info("Starting TriFusion Setup & Orchestrator...")
    logger.info("-" * 50)
    
    # 1. Datasets
    if not os.path.exists("data/raw/fer2013/train"):
        run_command("python data/download_datasets.py", "Downloading Datasets")
        
    # 2. Train Models
    # We check if the saved weights exist. If not, we train.
    models_to_train = [
        ("models/vision/efficientnet_fer2013.pth", "python -m src.vision.train_vision", "Training Vision Model (~45 mins)"),
        ("models/audio/wav2vec2_ravdess/pytorch_model.bin", "python -m src.audio.train_audio", "Training Audio Model (~1 hour)"),
        ("models/text/roberta_goemotions/pytorch_model.bin", "python -m src.text.train_text", "Training Text Model (~45 mins)"),
        ("models/fusion/fusion_mlp.pth", "python -m src.fusion.train_fusion", "Training Fusion Model (~10 mins)")
    ]
    
    os.makedirs("models/vision", exist_ok=True)
    os.makedirs("models/audio", exist_ok=True)
    os.makedirs("models/text", exist_ok=True)
    os.makedirs("models/fusion", exist_ok=True)
    
    all_models_ready = True
    for path, cmd, desc in models_to_train:
        # For HuggingFace models, the script saves a directory, so we check for pytorch_model.bin inside it.
        # Alternatively, check if the directory exists and is not empty.
        is_hf_dir = path.endswith(".bin")
        target_path = path if not is_hf_dir else os.path.dirname(path)
        
        if not os.path.exists(target_path) or (is_hf_dir and not os.path.exists(path)):
            all_models_ready = False
            logger.info(f"Model missing: {target_path}")
            run_command(cmd, desc)
            
    if all_models_ready:
        logger.info("✅ All AI models are fully trained and ready!")
        # If everything is trained, we can turn off simulation mode automatically!
        manager_path = "src/pipeline/manager.py"
        with open(manager_path, "r") as f:
            content = f.read()
        if "SIMULATION_MODE = True" in content:
            logger.info("Disabling Simulation Mode since models are trained...")
            content = content.replace("SIMULATION_MODE = True", "SIMULATION_MODE = False")
            with open(manager_path, "w") as f:
                f.write(content)
            
    # 3. Launch Dashboard
    logger.info("-" * 50)
    run_command("streamlit run dashboard/app.py", "Launching TriFusion Dashboard")

if __name__ == "__main__":
    main()
