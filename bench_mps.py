import torch
import time
from foxzero.common import FoxZeroResNet

model = FoxZeroResNet()
model.eval()

# Move model to MPS
device = torch.device("mps")
model = model.to(device)

dummy_input = torch.randn(1, 11, 4, 13, device=device)

# Warmup
with torch.no_grad():
    for _ in range(50):
        model(dummy_input)

# Benchmark
with torch.no_grad():
    start = time.time()
    for _ in range(800):
        model(dummy_input)
    # MPS is asynchronous, we must synchronize to get accurate timing for a synchronous loop
    torch.mps.synchronize()
    elapsed = time.time() - start

print(f"MPS (Batch 1, 800 times): {elapsed:.4f}s")
