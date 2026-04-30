import io
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image
from typing import Dict
import random, time
from concurrent.futures import ProcessPoolExecutor
from ai_engine import ChiikawaModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# จัดการ Path ใช้ Environment Variables + pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "mobilenetv3_quant.onnx"))
LABELS_PATH = os.getenv("LABELS_PATH", str(BASE_DIR / "models" / "labels.json"))
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", 0.7))


app = FastAPI(
    title="Chiikawa Classification Service",
    description="API สำหรับจำแนกตัวละคร Chiikawa โดยใช้ MobileNetV3",
    version="0.1.0"
)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- ส่วนจัดการ Traffic  ---
# จำกัดให้ CPU Bound ได้พร้อมกันแค่ 4 งาน ใครที่มาเกินกว่านี้จะ Wait in Queue แทนการโดนไล่
service_lock = asyncio.Semaphore(4)
executor = ProcessPoolExecutor(max_workers=4)

# ตัวแปรสำหรับเก็บโมเดลในแต่ละ Process 
engine = None

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    message: str

# ส่วนการประมวลผลโมเดล AI Classification 
def AI_inference_worker(image_bytes: bytes):
    """
    ฟังก์ชันคนงานที่จะถูกรันใน Process แยกต่างหาก 
    """
    global engine
    if engine is None:
        # เช็คก่อนว่าไฟล์โมเดลมีอยู่จริงไหม (ข้อดีของ pathlib object)
        m_path = Path(MODEL_PATH)
        if not m_path.exists():
            raise FileNotFoundError(f"หาไฟล์โมเดลไม่เจอที่: {m_path}")
            
        engine = ChiikawaModel(
            model_path=str(m_path), 
            labels_path=LABELS_PATH,
            threshold=THRESHOLD
        )
    label, score = engine.predict(image_bytes)
    return label, score


# Background Task
def log_model_processing(filename: str):
    print(f"กำลังส่งไฟล์ {filename} เข้าสู่กระบวนการรันโมเดลใน Background...")

# สร้าง Route สำหรับหน้าแรก (Frontend)
@app.get("/")
async def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/predict", response_model=PredictionResponse, tags=["MLOps"])
async def predict(file: UploadFile= File(...), 
                  backgroud_tasks: BackgroundTasks = BackgroundTasks(),
                  description="กรุณาอัปโหลไฟล์ภาพ Chiikawa ที่มีขนาดไม่เกิน 5 MB"):

    # --- Step 1 & 2: I/O Bound ---

    # Step 1 ตรวจนามสกุลไฟล์
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "ประเภทไฟล์ไม่ถูกต้อง",
                "received_type": file.content_type,
                "allowed_types": ["jpg", "jpeg", "png", "webp"],
                "suggestion": "กรุณาเปลี่ยนนามสกุลไฟล์ให้ถูกต้องก่อนอัปโหลด"
            }
        )

    # Step 2 อ่าน Binary จากรูปที่ได้ แล้ว Check File Size (I/O bound)
    content = await file.read()
    MAX_FILE_SIZE_MB = 5
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024 # แปลง bytes - kb - mb

    if len(content) > MAX_FILE_SIZE_BYTES:
        file_size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "message": "ไฟล์มีขนาดใหญ่เกินกว่าที่กำหนด",
                "current_size": f"{file_size_mb: .2f} MB",
                "limit": f"{MAX_FILE_SIZE_MB} MB",
                "suggestion": "กรุณาลดขนาดภาพให้เล็กลงหรือเท่ากับ 5MB"
            }
        )

    # --- Step 3: จัดการลำดับคิวและแสดงข้อความบอก User ---
    if service_lock.locked():
        current_message = "ขณะนี้มีผู้ใช้งานจำนวนมาก ไฟล์ของคุณถูกเข้าคิวไว้แล้ว กรุณารอสักครู่..."
    else:
        current_message = "คิวว่างแล้ว! กำลังประมวลผลไฟล์ของคุณทันที"

    async with service_lock:
        # 3.1 ตรวจสอบไส้ในไฟล์ กันการแปลงนามสกุลไฟล์หลอกๆหรือส่งไวรัสมา  CPU Bound
        def verify_image(image_content):
            try:
                # ใช้ ฺ BytesIO เพื่อให้ PILLOW อ่านจาก Memory โดยตรง
                img = Image.open(io.BytesIO(image_content))
                img.verify()
                return True
            except Exception:
                return False
            
        # 3.2 ส่งคนงานไปทำใน Thread Pool (ไม่บล็อก Event Loop)
        is_valid = await run_in_threadpool(verify_image, content)

        # 3.3 ตรวจสอบผลลัพธ์จากคนงาน ถ้า False ก็แจ้ง User ทันที 
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail="ไฟล์รูปภาพเสียหายหรือไม่ถูกต้อง Invalid Image"
            )
        
        # Step 4: การประมวลผลโมเดล  ---
        # ดึง Event Loop มาเพื่อสั่งให้ executor ทำงานแบบไม่ขวาง API
        loop = asyncio.get_running_loop()
        label, score = await loop.run_in_executor(executor, AI_inference_worker, content)
        
        # --- Background Task ---
        # ส่งงานเข้าคิวหลังบ้านทันที เพื่อให้ API คืนResponse ได้ไวที่สุด
        backgroud_tasks.add_task(log_model_processing, filename=file.filename)
        
        return PredictionResponse(
            filename=file.filename,
            prediction=label, 
            confidence=round(score, 4),
            message=current_message,
        )