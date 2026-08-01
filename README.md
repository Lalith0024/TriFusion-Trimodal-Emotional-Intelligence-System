# 🧠 TriFusion: Trimodal Emotional Intelligence System
A production-ready real-time emotional analysis system using Vision, Audio, and Text fusion.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-purple)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)]()

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Analysis Modes](#-analysis-modes)
- [Tech Stack](#️-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Performance](#-performance)
- [Configuration](#️-configuration)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [Future Enhancements](#-future-enhancements)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview
TriFusion is a comprehensive, production-style emotional intelligence system built to analyze human emotion through three distinct lenses: Vision (Face), Audio (Voice), and Text (Words). By fusing these signals, it identifies **Emotional Incongruence**—the subtle statistical gap where a person's spoken words don't match their physiological expressions.

### Why This Project?
✅ **Zero-lag Analysis:** Threaded pipeline architecture for instant real-time feedback.
✅ **Trimodal Fusion:** Advanced late-fusion MLP for unified emotional profiling.
✅ **Global Hardware Support:** Automatic GPU acceleration (CUDA/MPS) with CPU fallback.
✅ **Agentic Interventions:** LangGraph-powered wellness agent for automated psychological support.
✅ **Production-Ready:** Modular architecture with robust error handling and telemetry.

---

## ✨ Key Features

### Core Capabilities
🎯 **Multiple Analysis Modes**
- **Live Dashboard:** Real-time high-frequency streaming analysis.
- **Demo Scenarios:** Controlled architectural testing without hardware.
- **Session History:** Historical trend analysis and post-session review.

🚀 **Performance Optimizations**
- **GPU Acceleration:** Automatic support for CUDA (NVIDIA) and MPS (Apple Silicon).
- **FPS Throttling:** Configurable processing rates (10-30 FPS) to match hardware.
- **Threaded Capture:** Hardware capture (cam/mic) decoupled from the UI thread.
- **Fragment Refresh:** Streamlit Fragments for smooth, flicker-free telemetry updates.

🎨 **Interactive UI Controls**
- **Incongruence Threshold:** Real-time sensitivity adjustment (0.0 - 1.0).
- **Modality Toggles:** Enable/Disable Vision, Audio, or Text inputs dynamically.
- **Agent Escalation:** Control the WellnessAgent's threshold for interventions.
- **Radar Sensitivity:** Fine-tune Plotly radar responsiveness.

📊 **Real-Time Monitoring**
- **Live Incongruence Meter:** Visual alert system for masked distress.
- **Modality Signal Badges:** Real-time status and confidence for each input.
- **Agent Response Box:** Direct feedback and clinical intervention prompts.
- **Timeline Visualization:** Scrolling history of fused emotional states.

🛡️ **Robust Error Handling**
- **Hardware Fallbacks:** Graceful handling of missing cameras or microphones.
- **Environment Verification:** Automatic checking for Groq API keys and models.
- **Thread Recovery:** Self-healing pipeline for interrupted hardware streams.

---

## 🎬 Analysis Modes

### 1. 🎥 Live Dashboard Mode (Recommended)
**Best for:** Real-time interaction, patient monitoring, and live coaching.
*   **How it works:**
    1. Select "Live Dashboard" in the sidebar.
    2. Click "▶ Start Session" to initialize threads.
    3. Monitor the Radar Chart and Incongruence Meter.
    4. View WellnessAgent interventions in real-time.

### 2. 🎭 Demo Scenarios
**Best for:** Batch analysis, software testing, and non-hardware demonstrations.
*   **How it works:**
    1. Select "Demo Scenarios".
    2. Choose a preset profile (e.g., "Anxious Masking").
    3. Observe how the Fusion MLP reconciles conflicting signals.
    4. Review the Agent's decision-making logic.

### 3. 📊 Session History
**Best for:** Longitudinal tracking and session reporting.
*   **How it works:**
    1. Navigate to "Session History".
    2. Review previous session logs and peaked incongruence events.
    3. Analyze the frequency of agent-triggered interventions.

---

## 🛠️ Tech Stack

| Component | Technology | Version |
| :--- | :--- | :--- |
| **Language** | Python | 3.10+ |
| **Vision Model** | EfficientNet-B0 | 2.2.0 |
| **Audio Model** | Wav2Vec2-Base | 4.38.0 |
| **Text Model** | RoBERTa-Base | 4.38.0 |
| **Orchestration** | LangGraph | 0.0.30 |
| **Web UI** | Streamlit | 1.56+ |
| **Backend API** | FastAPI / Uvicorn | 0.110+ |

### Device Support
✅ **NVIDIA RTX (CUDA):** Automatic high-performance acceleration.
✅ **Apple Silicon (MPS):** Native Metal acceleration for M1/M2/M3.
✅ **Standard CPU:** Seamless fallback for universal compatibility.

---

## 🚀 Getting Started

Follow these step-by-step instructions to get TriFusion running on your local machine.

### Step 1: Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/Lalith0024/TriFusion-Trimodal-Emotional-Intelligence-System.git
cd TriFusion-Trimodal-Emotional-Intelligence-System
```

### Step 2: Set Up Python Environment
TriFusion requires **Python 3.10+**. It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies
With your virtual environment activated, install all required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Download Neural Models
TriFusion relies on specific pre-trained weights for Vision, Audio, and Text processing. Run the automated download script to fetch them into the correct directories:
```bash
python3 scripts/download_models.py
```
> **Note:** This may take a few minutes depending on your internet connection.

### Step 5: Configure Environment Variables
TriFusion's WellnessAgent is powered by Groq's high-speed inference. You must provide an API key.
1. Create a file named `.env` in the root directory.
2. Add your Groq API key to the file:
```env
GROQ_API_KEY=your_api_key_here
```
*(You can get a free API key at [console.groq.com](https://console.groq.com))*

---

### Step 6: Launch the Application
TriFusion operates with a decoupled architecture. You will need to start the Backend API and the Frontend Dashboard in **two separate terminal windows**.

#### 🖥️ Terminal 1: Start the Backend API
Keep your virtual environment activated in this terminal and start the FastAPI server:
```bash
# Ensure you are in the project root with the venv activated
python3 src/api/main.py
```
*Wait until you see `Uvicorn running on http://0.0.0.0:8000` in the logs.*

#### 🎨 Terminal 2: Start the Frontend Dashboard
Open a **new** terminal window, navigate to the project folder, activate the virtual environment, and start Streamlit:
```bash
# Navigate to the project root
cd /path/to/TriFusion-Trimodal-Emotional-Intelligence-System

# Activate the virtual environment again
# (On macOS/Linux)
source venv/bin/activate

# Launch the Streamlit dashboard
streamlit run dashboard/app.py
```

### Step 7: Analyze!
The dashboard should automatically open in your default web browser at `http://localhost:8501`. 
1. Navigate to the **Live Dashboard** in the sidebar.
2. Click **▶ Start Session** to initialize the trimodal capture pipeline.
3. Observe the fused emotional analysis in real-time!

---

## 🛠️ Verifying Your Setup
If you want to ensure your environment is configured correctly before launching the full app, you can run the diagnostic unit tests using `pytest`:
```bash
# Ensure you are in the project root with the venv activated
pytest tests/test_vision.py
pytest tests/test_audio.py
```

---

## 📖 Usage Guide

### Performance Tips
*   **Lighting:** Ensure the subject's face is well-lit for the EfficientNet module.
*   **Audio:** Use a dedicated microphone for better Wav2Vec2 performance.
*   **GPU:** Ensure `torch.backends.mps.is_available()` is True on Mac for 30 FPS.

---

## ⚡ Performance

### Device-Specific Performance
| Device Type | Expected FPS | Notes |
| :--- | :--- | :--- |
| **NVIDIA RTX 4090** | 50-60 FPS | Maximum performance |
| **Apple M3 Pro** | 25-35 FPS | GPU accelerated (MPS) |
| **Apple M1** | 15-20 FPS | GPU accelerated (MPS) |
| **Intel i7 CPU** | 5-10 FPS | CPU-only fallback |

### Optimization Strategies
- **MPS Optimization:** Native Metal support for Apple Silicon.
- **CUDA Support:** Fully compatible with NVIDIA's GPU architecture.
- **Throttled Sync:** UI updates every 10ms to balance load and smoothness.

---

## ⚙️ Configuration

### Quick Configuration (`config/config.yaml`)
```yaml
# Model paths and thresholds
vision_path: "models/vision/efficientnet.pth"
incongruence_threshold: 0.7
```

### Advanced Configuration
Adjust the `PipelineManager` in `src/pipeline/manager.py` to change capture buffer sizes or audio sampling rates.

---

## 📁 Project Structure

```text
TriFusion/
├── dashboard/           # Streamlit UI & Components
├── src/
│   ├── agent/           # LangGraph reasoning
│   ├── api/             # FastAPI backend
│   ├── audio/           # Vocal analysis
│   ├── fusion/          # Neural fusion
│   ├── pipeline/        # Threaded manager
│   ├── text/            # Linguistic analysis
│   └── vision/          # Facial analysis
└── tests/               # Validation suite
```

### Module Responsibilities
- **src/fusion:** Merges 23-dimensional vectors into unified emotional states.
- **src/agent:** Provides clinical interventions via LLaMA-3.3 reasoning.
- **src/pipeline:** Handles high-speed data flow without UI blocking.

---

## 🐛 Troubleshooting

❌ **"StreamlitDuplicateElementId"**
> Ensure you are using the updated `radar_chart` with static key bindings.

❌ **"Failed to detect GPU"**
> Run `python3 -c "import torch; print(torch.cuda.is_available())"` to verify drivers.

❌ **"Groq Authentication Error"**
> Double-check your `.env` for hidden spaces or missing characters.

---

## 🎓 Advanced Usage
For researchers, the `FusionMLP` training weights can be extracted from `models/fusion/` to analyze modality-weight bias.

## 🔮 Future Enhancements
- [ ] Real-time region-of-interest (ROI) tracking for micro-expressions.
- [ ] Support for multiple simultaneous subjects.
- [ ] Exportable session analytics in CSV/JSON.

---

## 🙏 Acknowledgments
- **RAVDESS & FER2013** for the primary training datasets.
- **LangChain** for the agentic orchestration.
- **HuggingFace** for the Wav2Vec2 and RoBERTa foundations.

Built with ❤️ by [Lalithendra Kasula](https://github.com/Lalith0024)
