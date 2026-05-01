from pathlib import Path

import json
import numpy as np
import onnxruntime as ort
from PIL import Image

ONNX_PATH  = Path("models/model.onnx")
LABELS     = Path("models/labels.json")
TEST_DIR   = Path("dataset/test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # NCHW
    return arr


def predict(sess, arr: np.ndarray, classes: list[str]):
    logits = sess.run(None, {"input": arr})[0][0]
    probs  = np.exp(logits) / np.exp(logits).sum()
    idx    = probs.argmax()
    return classes[idx], float(probs[idx]), probs.tolist()


def main():
    with open(LABELS, encoding="utf-8") as f:
        classes = json.load(f)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print(f"Model  : {ONNX_PATH}")
    print(f"Classes: {classes}\n")
    print(f"{'File':<40} {'Prediction':<12} {'Confidence':>10}  (all probs)")
    print("-" * 80)

    images = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print("No images found in dataset/test/")
        return

    for img_path in images:
        try:
            arr              = preprocess(img_path)
            label, conf, probs = predict(sess, arr, classes)
            prob_str         = "  ".join(f"{c}:{p:.2f}" for c, p in zip(classes, probs))
            print(f"{img_path.name:<40} {label:<12} {conf:>10.2%}  [{prob_str}]")
        except Exception as e:
            print(f"{img_path.name:<40} ERROR: {e}")


if __name__ == "__main__":
    main()
