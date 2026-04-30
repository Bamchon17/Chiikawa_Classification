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
        print(f"--- [INIT] Model Loaded: {model_path} ---")
        print(f"--- [INIT] Labels: {self.classes} (Total: {len(self.classes)}) ---")
        print(f"--- [INIT] Current Threshold: {self.threshold} ---")

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def predict(self, image_bytes: bytes):
        try:
            print("\n" + "="*50)
            print("🔍 DEBUG: Start Prediction Process")
            
            # --- PRE-PROCESSING ---
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize((224, 224)) 
            
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Debug Pixel Range: เช็คว่า ToTensor ทำงานถูกไหม
            print(f"DEBUG: Image Scale Min: {img_array.min():.4f}, Max: {img_array.max():.4f}")
            
            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0)

            # --- INFERENCE ---
            input_name = self.session.get_inputs()[0].name
            raw_outputs = self.session.run(None, {input_name: img_array})
            
            logits = np.squeeze(raw_outputs[0]) 
            print(f"DEBUG: Raw Logits (Top 5): {logits[:5]}") # ดูค่าดิบก่อนเข้า Softmax

            # --- POST-PROCESSING ---
            probabilities = self._softmax(logits)
            idx = np.argmax(probabilities)
            conf_score = float(probabilities[idx])

            # Debug ผลลัพธ์ทั้งหมด
            print(f"DEBUG: All Probabilities: {['{:.4f}'.format(p) for p in probabilities]}")
            print(f"DEBUG: Winner Index: {idx} ({self.classes[idx]})")
            print(f"DEBUG: Confidence Score: {conf_score:.4f}")

            # --- LOGIC: THRESHOLD CHECK ---
            if conf_score < self.threshold:
                print(f"⚠️ RESULT: Unknown (Reason: {conf_score:.4f} < {self.threshold})")
                # ส่งค่ากลับไปพร้อมระบุว่าจริงๆ แล้วมันคือตัวอะไรแต่ Confidence ไม่ถึง
                return f"Unknown (Maybe {self.classes[idx]}?)", conf_score
            
            print(f"✅ RESULT: Prediction Success -> {self.classes[idx]}")
            return self.classes[idx], conf_score

        except Exception as e:
            print(f"❌ ERROR: Prediction Failed -> {str(e)}")
            return "Error", 0.0