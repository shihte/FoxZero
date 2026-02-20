import torch
import time
from foxzero.common import FoxZeroResNet

model = FoxZeroResNet()
model.eval()
model = torch.jit.script(model)

dummy_input = torch.randn(1, 11, 4, 13)

def bench(name, ctx, threads):
    if threads is not None:
        torch.set_num_threads(threads)
    with ctx():
        # Warmup
        for _ in range(10):
            model(dummy_input)
        
        start = time.time()
        for _ in range(800):
            model(dummy_input)
        elapsed = time.time() - start
        
    print(f"{name}: {elapsed:.4f}s")

bench('no_grad + default threads', torch.no_grad, None)
bench('inference_mode + default threads', torch.inference_mode, None)
bench('inference_mode + 1 thread', torch.inference_mode, 1)
bench('inference_mode + 2 threads', torch.inference_mode, 2)
bench('inference_mode + 4 threads', torch.inference_mode, 4)
