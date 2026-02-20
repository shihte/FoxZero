import time
import numpy as np
import onnxruntime as ort

onnx_path = "foxzero_model.onnx"

session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
dummy_numpy = np.random.rand(1, 11, 4, 13).astype(np.float32)

# Warmup
for _ in range(10):
    session.run(None, {'input': dummy_numpy})

start = time.time()
for _ in range(3000):
    session.run(None, {'input': dummy_numpy})
elapsed = time.time() - start

print(f"ONNX CPU (Default threads, 3000 calls): {elapsed:.4f}s")
