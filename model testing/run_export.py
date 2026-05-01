"""
Export & Quantization Benchmark
Steps:
  1. Load trained PyTorch model
  2. Baseline  — PyTorch FP32 inference speed + file size
  3. Conversion — export to ONNX (FP32)
  4. Quantization — ONNX Dynamic INT8
  5. Report — compare all three side-by-side
"""

import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import models
from PIL import Image

from train.utils.transforms import val_transform

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = Path("models/best_model.pth")
ONNX_PATH    = Path("models/model.onnx")
QUANT_PATH   = Path("models/model_int8.onnx")
TEST_IMAGE   = Path("dataset/test/chiikawa head.png")   # รูปทดสอบ
WARMUP_RUNS  = 10
BENCH_RUNS   = 100
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────
def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def load_pytorch_model(device: torch.device):
    ckpt        = torch.load(CHECKPOINT, map_location=device)
    class_names = ckpt["classes"]
    model = models.mobilenet_v3_small()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_names


def prepare_input(device: torch.device):
    img    = Image.open(TEST_IMAGE).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(device)
    return tensor


def bench(fn, warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS) -> float:
    """Returns mean latency in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def print_row(label: str, size_mb: float, latency_ms: float):
    print(f"  {label:<20} {size_mb:>8.2f} MB   {latency_ms:>8.2f} ms")


# ── Step 1 — Load ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading model...")
device = torch.device("cpu")
model, class_names = load_pytorch_model(device)
tensor = prepare_input(device)
print(f"Classes : {class_names}")
print(f"Test image: {TEST_IMAGE.name}")


# ── Step 2 — Baseline: PyTorch FP32 ───────────────────────────────────────────
print("\n[2] Baseline — PyTorch FP32")
with torch.no_grad():
    out = model(tensor)
pred = class_names[out.argmax().item()]
print(f"  Prediction: {pred}")

pytorch_latency = bench(lambda: model(tensor) if True else None)
# wrap in no_grad for fair benchmark
def _pt():
    with torch.no_grad():
        model(tensor)

pytorch_latency = bench(_pt)
pytorch_size    = file_size_mb(CHECKPOINT)
print(f"  Size    : {pytorch_size:.2f} MB")
print(f"  Latency : {pytorch_latency:.2f} ms  (avg over {BENCH_RUNS} runs)")


# ── Step 3 — Export to ONNX ───────────────────────────────────────────────────
print("\n[3] Conversion — Export to ONNX (FP32)")
torch.onnx.export(
    model,
    tensor,
    str(ONNX_PATH),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
    dynamo=False,
)
onnx.checker.check_model(str(ONNX_PATH))
print(f"  Saved  : {ONNX_PATH}")

sess_fp32 = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
np_input  = tensor.numpy()

with torch.no_grad():
    out_onnx = sess_fp32.run(None, {"input": np_input})[0]
pred_onnx = class_names[out_onnx.argmax()]
print(f"  Prediction: {pred_onnx}")

onnx_latency = bench(lambda: sess_fp32.run(None, {"input": np_input}))
onnx_size    = file_size_mb(ONNX_PATH)
print(f"  Size    : {onnx_size:.2f} MB")
print(f"  Latency : {onnx_latency:.2f} ms  (avg over {BENCH_RUNS} runs)")


# ── Step 4 — Dynamic Quantization (INT8) ──────────────────────────────────────
print("\n[4] Quantization — ONNX Dynamic INT8")
ONNX_PREP = Path("models/model_prep.onnx")
quant_pre_process(str(ONNX_PATH), str(ONNX_PREP), skip_symbolic_shape=True)
quantize_dynamic(
    model_input=str(ONNX_PREP),
    model_output=str(QUANT_PATH),
    weight_type=QuantType.QInt8,
)
print(f"  Saved  : {QUANT_PATH}")

sess_int8    = ort.InferenceSession(str(QUANT_PATH), providers=["CPUExecutionProvider"])
out_int8     = sess_int8.run(None, {"input": np_input})[0]
pred_int8    = class_names[out_int8.argmax()]
print(f"  Prediction: {pred_int8}")

quant_latency = bench(lambda: sess_int8.run(None, {"input": np_input}))
quant_size    = file_size_mb(QUANT_PATH)
print(f"  Size    : {quant_size:.2f} MB")
print(f"  Latency : {quant_latency:.2f} ms  (avg over {BENCH_RUNS} runs)")


# ── Step 5 — Report ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY COMPARISON")
print("=" * 60)
print(f"  {'Model':<20} {'Size':>10}   {'Latency':>10}")
print("-" * 48)
print_row("PyTorch FP32",  pytorch_size,  pytorch_latency)
print_row("ONNX FP32",     onnx_size,     onnx_latency)
print_row("ONNX INT8",     quant_size,    quant_latency)
print("=" * 48)
