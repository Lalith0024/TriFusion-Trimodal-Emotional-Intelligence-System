import torch
import pytest
from src.vision.emotion_model import FacialEmotionNet
from config.emotions import FER2013_NUM_CLASSES

def test_facial_emotion_net_output_shape():
    model = FacialEmotionNet(pretrained=False, num_classes=FER2013_NUM_CLASSES)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, FER2013_NUM_CLASSES)

def test_facial_emotion_net_probabilities():
    model = FacialEmotionNet(pretrained=False, num_classes=FER2013_NUM_CLASSES)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        probs = model.get_probabilities(dummy_input)
    assert probs.shape == (1, FER2013_NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
