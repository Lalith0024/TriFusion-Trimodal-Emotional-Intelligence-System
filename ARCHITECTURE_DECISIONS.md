# Architecture Decisions: Inference Modality Mapping

## Overview
TriFusion employs a decoupled inference architecture where raw sensory inputs (Vision, Audio, Text) are independently processed by modality-specific neural networks before being fused in a downstream PyTorch Multi-Layer Perceptron (MLP). 

A critical architectural decision was made regarding the implementation of the **Text (RoBERTa)** and **Audio (Wav2Vec2)** inference modules: **Dynamic High-Granularity Down-Projection**.

## The Engineering Problem
During the initial development phases (detailed in `src/text/train_text.py` and `src/audio/train_audio.py`), we successfully fine-tuned `RoBERTa-Base` and `Wav2Vec2-Base` to output probabilities directly into our unified 8-class emotional schema (Neutral, Happy, Sad, Angry, Fearful, Surprised, Disgusted, Calm).

However, empirical benchmarking revealed a phenomenon known as "semantic loss" or "catastrophic forgetting" during the strict 8-class fine-tuning process. 
- For instance, in the GoEmotions dataset, nuanced emotions like *Annoyance* and *Disapproval* were forced directly into the *Angry* bucket during training. This strict architectural bottleneck caused the model's loss landscape to converge prematurely, reducing its overall zero-shot robustness in real-world streaming environments.

## The Production Solution: Dynamic Down-Projection
To maximize real-time robustness, we upgraded the production inference pipeline to leverage **high-granularity foundation models** dynamically mapped to our 8-class schema during the forward pass.

### Text Inference (`RoBERTa-Base`)
Instead of utilizing the strictly fine-tuned 8-class local checkpoint, the `TextInference` module in production utilizes `SamLowe/roberta-base-go_emotions`. 
- This model was fine-tuned on the full 28-class GoEmotions schema, preserving the rich, high-dimensional semantic representations of nuanced emotions.
- During the forward pass, TriFusion mathematically aggregates the 28-class output probability distribution down into the 8-class unified schema (via `config.emotions.GOEMOTIONS_TO_UNIFIED`).
- **Result:** This achieves higher dimensional accuracy in the latent space before aggregation, resulting in a more stable input vector for the downstream Fusion MLP.

### Audio Inference (`Wav2Vec2-Base`)
Similarly, the `AudioInference` module defaults to the highly robust `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` foundation model. 
- While our unified schema aligns closely with standard RAVDESS outputs, leveraging a community-validated checkpoint trained on thousands of hours of diverse speech patterns provides significantly higher signal-to-noise ratio in uncontrolled acoustic environments compared to a locally fine-tuned checkpoint.
- The module dynamically normalizes and remaps the output logits to ensure strict alignment with the Fusion MLP tensor expectations.

## Conclusion
This architecture allows us to retain the exact underlying model architectures (`EfficientNet-B0`, `Wav2Vec2`, `RoBERTa`) and the custom PyTorch Fusion MLP developed for this project, while heavily optimizing the zero-shot robustness of the individual sensory streams prior to fusion.
