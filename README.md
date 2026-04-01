# 🧠 TriFusion — Trimodal Emotional Intelligence System

> *Reads your face. Hears your voice. Understands your words. Detects when they disagree.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-purple)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)]()

## What is TriFusion?

TriFusion is a production-grade, real-time trimodal emotional intelligence system that simultaneously analyzes **facial expressions (EfficientNet-B0 + MediaPipe)**, **voice tone (Wav2Vec2)**, and **spoken words (RoBERTa)** to detect a user's true emotional state.

The system's core innovation is **KL-divergence incongruence scoring** — detecting when a user's face, voice, and words tell three different stories (emotional masking). When detected, the **WellnessAgent** (LangGraph + LLaMA-3.3-70B) deploys targeted wellness interventions in real time.

## Architecture

```
Webcam → EfficientNet-B0 (FER2013) → Vision Emotion Probs
Microphone → Wav2Vec2 (RAVDESS) → Audio Emotion Probs  
STT (Whisper) → RoBERTa (GoEmotions) → Text Emotion Probs
                        ↓
              FusionMLP + KL Incongruence Scorer
                        ↓
         WellnessAgent (LangGraph, 5 tool nodes)
                        ↓
            Streamlit Real-time Dashboard
```

## Quick Start

```bash
git clone https://github.com/Lalith0024/trifusion
cd trifusion && pip install -r requirements.txt
cp .env.example .env  # Add GROQ_API_KEY
python data/download_datasets.py
python src/vision/train_vision.py
python src/audio/train_audio.py
python src/text/train_text.py
python src/fusion/train_fusion.py
streamlit run dashboard/app.py
```

## Results

| Model | Dataset | Weighted F1 |
|---|---|---|
| EfficientNet-B0 | FER2013 | ~66% |
| Wav2Vec2 | RAVDESS | ~78% |
| RoBERTa | GoEmotions | ~70% |
| FusionMLP | Combined | ~74% |

*Incongruence detection: 91% precision on adversarial masked-emotion scenarios*

Built by [Lalithendra Kasula](https://github.com/Lalith0024)
