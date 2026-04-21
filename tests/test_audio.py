import torch
import pytest
from src.audio.emotion_model import build_wav2vec2_model
from config.emotions import RAVDESS_NUM_CLASSES

def test_wav2vec2_model_output_shape():
    model = build_wav2vec2_model(num_labels=RAVDESS_NUM_CLASSES, pretrained=False)
    model.eval()
    # Mock input_values shape (batch_size, sequence_length)
    dummy_input = torch.randn(2, 16000 * 3) 
    with torch.no_grad():
        output = model(dummy_input)
    assert output.logits.shape == (2, RAVDESS_NUM_CLASSES)
