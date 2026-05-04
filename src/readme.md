# 🐹 Chiikawa Classification Service (Backend)

## 📌 Project Overview
API นี้ทำหน้าที่เป็นด่านหน้า (**Gateway**) สำหรับการจำแนกภาพตัวละคร **Chiikawa, Usagi, และ Hachiware** โดยใช้โมเดล **MobileNetV3** ระบบถูกออกแบบภายใต้แนวคิด MLOps ที่เน้นความเสถียร รองรับปริมาณ Traffic สูง (High-Throughput) และมีระบบตรวจสอบความปลอดภัยของไฟล์ภาพที่เข้มงวด เพื่อให้มั่นใจว่า Backend จะทำงานได้อย่างราบรื่นในสภาพแวดล้อมจริง

---

## 🏗️ System Architecture & Logic

โปรเจกต์นี้ให้ความสำคัญกับ **Efficiency** และ **Security** โดยแบ่งโครงสร้างการทำงานออกเป็นส่วนๆ ดังนี้:

### 1. API Core & Data Validation ⚙️
ใช้ **FastAPI** เป็นหัวใจหลักในการคุม Route และจัดการ Data Schema
* **Async Implementation:** ใช้ `async def` เพื่อบริหารจัดการ Event Loop อย่างมีประสิทธิภาพ
* **Pydantic Guard:** ล็อกสเปก Output ด้วย `PredictionResponse` (filename, prediction, confidence, message)
* **Auto-Docs:** รองรับ Swagger UI สำหรับทดสอบ API ได้ทันทีผ่าน endpoint `/docs`

### 2. High-Performance Optimization ⚡
จัดการปัญหาคอขวด (Bottleneck) ด้วยกลยุทธ์ **Hybrid Task Management**:
* **I/O Bound Task:** การรับและอ่านไฟล์ใช้ `await` เพื่อไม่ให้บล็อกการทำงานของ Request อื่น
* **CPU Bound Task:** การรันโมเดลและการตรวจสอบไฟล์ (Heavy Computation) จะถูกส่งไปทำที่ `run_in_threadpool` เพื่อกระจายโหลดไปยัง Worker Threads (เช่น Intel Core Ultra 5) ป้องกัน API ค้าง
* **Concurrency:** รองรับการประมวลผลภาพหลายภาพพร้อมกันโดยไม่ต้องเข้าคิว (Anti-Blocking)

### 3. Deep Security Inspection 🛡️
เราไม่ได้ตรวจสอบแค่ "นามสกุลไฟล์" แต่เราเช็คไปถึงโครงสร้างข้อมูลภายใน (File Integrity):
* **Magic Bytes Validation:** ตรวจสอบ Header ในระดับ Hex เพื่อยืนยันว่าเป็นไฟล์ JPEG, PNG หรือ WebP จริง (ป้องกัน Malicious Script)
* **Pillow Deep Verification:** ใช้ `img.verify()` ตรวจสอบว่าไฟล์ไม่เสียหาย (Corrupted) และมีโครงสร้าง Chunk ข้อมูลที่ถูกต้อง
* **Payload Control:** จำกัดขนาดไฟล์สูงสุดที่ 5MB ($5 \text{ MB} \times 1024^2 \text{ Bytes}$) เพื่อป้องกันการโจมตีแบบ DoS

[Planned] Step 2: Model Inference & Multiprocessing
เนื่องจากการรันโมเดล AI เป็นงานหนักระดับ Heavy CPU Bound เราจึงวางแผนจัดการดังนี้:

Concurrency Strategy: เลือกใช้ ProcessPoolExecutor หรือ ONNX Runtime Optimization เพื่อแยกการคำนวณออกจาก Main Event Loop

Parallel Execution: เพื่อให้รูปที่ 1 และรูปที่ 2 สามารถทำ Inference บน Core CPU ที่แยกกันได้จริง (Parallelism) ป้องกันการรอคิวในระดับระดับโมเดล

Memory Management: มีการจัดการ Pre-processing (Resize/Normalize) ในระดับ Worker เพื่อให้ Main Process ทำหน้าที่รับ-ส่งข้อมูลเพียงอย่างเดียว

---

## ⚡ Performance Matrix (DevOps Way)

| ประเภทงาน | ขั้นตอน | วิธีการจัดการ (Strategy) |
| :--- | :--- | :--- |
| **I/O Bound** | รับไฟล์จาก User, อ่าน Binary | Non-blocking Event Loop (`await`) |
| **CPU Bound** | Image Verification, Model Inference | Worker Threadpool (Parallel Processing) |

---

## 🛠️ Installation & Running
```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload


---

