import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import json

class ChiikawaModel:
    def __init__(self, model_path: str, labels_path: str, threshold: float = 0.7):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2 
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=options,
            providers=['CPUExecutionProvider']
        )
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.classes = json.load(f)
            
        self.threshold = threshold
        # ค่า MEAN และ STD ตามที่เพื่อนใช้เทรน (สำคัญมาก!)
        self.MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"--- [INIT] Model Loaded: {model_path} ---")
        print(f"--- [INIT] Labels: {self.classes} ---")

    def predict(self, image_bytes: bytes):
        try:
            # --- PRE-PROCESSING ---
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            
            # ทำ Normalization ให้ตรงกับที่เพื่อนเทรน
            arr = (arr - self.MEAN) / self.STD
            arr = arr.transpose(2, 0, 1)[np.newaxis]  # NCHW

            # --- INFERENCE ---
            # ใช้ชื่อ input ว่า "input" ตามโค้ดที่เพื่อนใช้
            raw_outputs = self.session.run(None, {"input": arr})
            logits = np.squeeze(raw_outputs[0]) 

            # --- POST-PROCESSING ---
            probs = np.exp(logits) / np.exp(logits).sum()
            idx = probs.argmax()
            conf_score = float(probs[idx])
            label = self.classes[idx]

            # --- THRESHOLD CHECK ---
            if conf_score < self.threshold:
                return f"Unknown (Maybe {label}?)", conf_score
            
            return label, conf_score

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return "Error", 0.0