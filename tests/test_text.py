import torch
from transformers import RobertaForSequenceClassification
from src.text.emotion_model import build_roberta_model
from config.emotions import TEXT_NUM_CLASSES

def test_roberta_model_output_shape():
    model = build_roberta_model(num_labels=TEXT_NUM_CLASSES, pretrained=False)
    model.eval()
    
    # Mock input tensor shape (batch_size, sequence_length)
    dummy_input = torch.randint(0, 50000, (2, 32))
    
    with torch.no_grad():
        output = model(input_ids=dummy_input)
        
    assert output.logits.shape == (2, TEXT_NUM_CLASSES)
