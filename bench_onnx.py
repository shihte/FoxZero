import torch
import time
import numpy as np
import onnxruntime as ort
from foxzero.common import FoxZeroResNet
import os

model = FoxZeroResNet()
model.eval()

dummy_input = torch.randn(1, 11, 4, 13)

# Export to ONNX
onnx_path = "foxzero_model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['policy', 'value'],
    dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
)

# Benchmark ONNX Runtime
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
dummy_numpy = dummy_input.numpy()

# Warmup
for _ in range(10):
    session.run(None, {'input': dummy_numpy})

start = time.time()
for _ in range(800):
    session.run(None, {'input': dummy_numpy})
elapsed = time.time() - start

print(f"ONNX CPU (1 thread): {elapsed:.4f}s")
os.remove(onnx_path)
