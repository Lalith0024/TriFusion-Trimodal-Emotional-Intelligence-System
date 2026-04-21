import pytest
import torch
from src.fusion.fusion_model import FusionMLP
from src.fusion.incongruence import compute_incongruence
from config.emotions import UNIFIED_EMOTIONS

def test_fusion_mlp_output_shape():
    model = FusionMLP(input_dim=23, output_dim=8)
    model.eval()
    
    # Batch of 2
    v_probs = torch.randn(2, 7)
    a_probs = torch.randn(2, 8)
    t_probs = torch.randn(2, 8)
    
    with torch.no_grad():
        output = model(v_probs, a_probs, t_probs)
        
    assert output.shape == (2, 8)
    assert torch.allclose(output.sum(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-5)

def test_compute_incongruence_aligned():
    probs = {e: 0.1 for e in UNIFIED_EMOTIONS}
    probs["happy"] = 0.8
    score = compute_incongruence(probs, probs, probs)
    assert score == 0.0

def test_compute_incongruence_high():
    v_probs = {e: 0.01 for e in UNIFIED_EMOTIONS}
    v_probs["sad"] = 0.9
    
    a_probs = {e: 0.01 for e in UNIFIED_EMOTIONS}
    a_probs["angry"] = 0.9
    
    t_probs = {e: 0.01 for e in UNIFIED_EMOTIONS}
    t_probs["happy"] = 0.9
    
    score = compute_incongruence(v_probs, a_probs, t_probs)
    assert score > 0.5 # Should be quite high
