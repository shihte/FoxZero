import torch
import time
import numpy as np
from foxzero.common import FoxZeroResNet

model = FoxZeroResNet()
model.eval()
model = torch.jit.script(model)

dummy_s_t = np.random.rand(11 * 4 * 13).astype(np.float32)

device = next(model.parameters()).device

start = time.time()
for _ in range(3000):
    inp = torch.from_numpy(dummy_s_t).view(11, 4, 13).unsqueeze(0).to(device)
    with torch.no_grad():
        l, v = model(inp)
        val = v.item()
        p_dist = torch.softmax(l, dim=1).cpu().numpy().flatten()
elapsed = time.time() - start

print(f"Callback Simulation (3000 calls): {elapsed:.4f} s")
