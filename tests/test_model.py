import torch
import numpy as np
import sys
import os

# Ensure foxzero is in path
sys.path.append(os.getcwd())

def test_model():
    print("Testing FoxZeroNet...")
    try:
        from foxzero.model import FoxZeroNet
    except ImportError as e:
        print(f"Failed to import FoxZeroNet: {e}")
        return

    net = FoxZeroNet()
    print("Model initialized.")
    
    # Create dummy input based on dimension [Batch, 11, 4, 13]
    batch_size = 2
    dummy_input = torch.randn(batch_size, 11, 4, 13)
    
    print(f"Input shape: {dummy_input.shape}")
    
    p, v = net(dummy_input)
    
    print(f"Policy Output shape: {p.shape}")
    print(f"Value Output shape: {v.shape}")
    
    assert p.shape == (batch_size, 52)
    assert v.shape == (batch_size, 1)
    
    # Check values range
    assert torch.all(p >= 0) and torch.all(p <= 1)
    # Check softmax sum
    sums = torch.sum(p, dim=1)
    print(f"Policy sums: {sums}")
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    assert torch.all(v >= -1) and torch.all(v <= 1)
    
    print("Model verification passed!")

if __name__ == "__main__":
    test_model()
