<p align="center">
  <img src="assets/images/ChiikawaBanner.jpeg" width="100%" alt="Project Banner">
</p>



# 🐹 Chiikawa Image Classification Service (MLOps Challenge)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)

## 📌 Project Overview
This project is developed for **AIE494 MLOps**. It focuses on building a high-throughput image classification service for **Chiikawa characters** (Usagi, Hachiware, and Chiikawa). The system features a highly optimized model using **ONNX conversion and Dynamic Quantization** to achieve low latency and a small footprint without compromising accuracy. 

The API is built with **FastAPI**, leveraging **multiprocessing** to efficiently handle high-concurrency user traffic, and is fully integrated with a **CI/CD pipeline** for automated deployment to Hugging Face Spaces.

---

## 🛠 Project Structure & Responsibilities

โปรเจกต์นี้พัฒนาด้วยภาษา **Python** เป็นหลัก โดยแบ่งการทำงานออกเป็น 3 ส่วนหลัก ดังนี้:

### 1. Model Optimization & Performance (คิว - Kew)
เน้นการเตรียมโมเดลให้มีประสิทธิภาพสูงสุดสำหรับการใช้งานจริง (Production)
* **Model Selection:** เลือกโมเดล MobileNetV3 (Pre-trained) จาก Hugging Face
* **Conversion & Quantization:** แปลงโมเดลเป็น ONNX Format และทำ Dynamic Quantization เพื่อลดขนาดไฟล์และเพิ่มความเร็ว
* **Performance Analysis:** * ทดสอบ Baseline (Speed vs Size) ของโมเดลแต่ละเวอร์ชัน
    * ทำ Load Testing ด้วย JMeter เพื่อวัดค่า TPS และ Latency P95
* **Deliverables:** ตารางเปรียบเทียบประสิทธิภาพ, ไฟล์ `.jmx`, และ JMeter Dashboard

### 2. API Development & Production Handling (แบม - Bam)
เน้นการสร้างช่องทางการเชื่อมต่อที่รัดกุมและรองรับ Traffic ปริมาณมาก
* **Backend Framework:** พัฒนา API ด้วย FastAPI (async def)
* **Concurrency Management:** ใช้ `ProcessPoolExecutor` หรือ Multiprocessing เพื่อป้องกัน API Frozen จากงาน CPU-bound
* **Robust Error Handling:** ใช้ Pydantic ตรวจสอบความถูกต้องของ Input และจัดการ HTTP Status Codes (400, 413, 500)
* **Containerization:** เขียน Dockerfile สำหรับแพ็กแอปพลิเคชันให้มีขนาดเล็กที่สุด (Small Footprint)
* **API Documentation:** จัดทำ Postman Collection และ cURL commands

### 3. Automation, CI/CD & Reporting (เจแปน - Japan)
เน้นการตรวจสอบคุณภาพและการส่งมอบงานแบบอัตโนมัติ
* **Quality Assurance:** เขียน Unit Test ด้วย `pytest` ตรวจสอบ API Endpoint และความแม่นยำของโมเดล
* **CI/CD Pipeline:** ตั้งค่า GitHub Actions สำหรับรัน Unit Test อัตโนมัติทุกครั้งที่มีการ Push โค้ด
* **Deployment:** ระบบ Auto-Deploy ไปยัง Hugging Face Spaces เมื่อผ่านการทดสอบ 100%
* **Documentation:** วาด System Architecture Diagram, เขียนรายงานโปรเจกต์ (PDF) และเตรียม Presentation สรุปงาน

---

## 🗂 Git LFS Setup (สำหรับ Dataset)

โปรเจกต์นี้ใช้ **Git LFS (Large File Storage)** สำหรับจัดการไฟล์รูปภาพใน `dataset/` เพื่อนที่ clone ไปต้องติดตั้ง Git LFS ก่อน ไม่งั้นจะได้แค่ pointer file แทนรูปจริง

### ติดตั้ง Git LFS

**Windows (แนะนำ):**
```bash
# ดาวน์โหลดจาก https://git-lfs.com/ แล้วติดตั้ง
# หรือใช้ winget
winget install GitHub.GitLFS
```

**Mac:**
```bash
brew install git-lfs
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install git-lfs
```

### ใช้งาน Git LFS

```bash
# 1. เปิดใช้งาน LFS ใน machine (ทำครั้งเดียว)
git lfs install

# 2. Clone โปรเจกต์ตามปกติ (LFS จะ download รูปให้อัตโนมัติ)
git clone <repo-url>

# 3. ถ้า clone ไปแล้วแต่รูปยังไม่โหลด ให้รัน
git lfs pull
```

> **หมายเหตุ:** ไฟล์ที่ track ด้วย LFS ได้แก่ `.png`, `.jpg`, `.jpeg`, `.webp`

---

## 📁 Directory Structure (Proposed)
```text
.
├── src/                # FastAPI Application source code
│   └── main.py             # FastAPI entry point
├── train/              # Training pipeline
│   ├── dataset.py          # Dataset loader & augmentation
│   ├── engine.py           # Train/validation loop
│   ├── evaluate.py         # Evaluation & metrics
│   ├── train.py            # Training entry point
│   └── utils/              # Shared utilities
│       ├── transforms.py   # Image transforms
│       ├── preprocess.py   # Image preprocessing
│       ├── metrics.py      # Evaluation metrics
│       ├── visualization.py# Plotting & visualization
│       ├── data_prepare.py # Dataset preparation helpers
│       └── seed.py         # Reproducibility seed
├── models/             # Exported ONNX & Quantized models
│   ├── best_model.pth      # PyTorch checkpoint (training output)
│   ├── model.onnx          # ONNX FP32 (exported)
│   ├── model_prep.onnx     # ONNX pre-processed (quantization intermediate)
│   ├── model_int8.onnx     # ONNX INT8 Quantized (production)
│   ├── labels.json         # Class label mapping
│   ├── training_curves.png # Loss & accuracy curves (training artifact)
│   └── confusion_matrix.png# Confusion matrix (testing artifact)
├── tests/              # Pytest unit tests
├── .github/workflows/  # CI/CD configuration (GitHub Actions)
├── performance/        # JMeter scripts and Load test results
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🤖 Model Artifacts

โมเดลผ่าน pipeline ครบทุกขั้นตอน ผลลัพธ์ทั้งหมดอยู่ใน `models/`

| File | Description |
|---|---|
| `best_model.pth` | PyTorch checkpoint (ต้นทาง, ใช้สำหรับ export) |
| `model.onnx` | ONNX FP32 (export จาก PyTorch) |
| `model_prep.onnx` | ONNX ที่ผ่าน `quant_pre_process` (intermediate step ก่อน quantize) |
| `model_int8.onnx` | ONNX Dynamic INT8 Quantized (ใช้งานใน production) |

**Training pipeline:**
```
Train → Export (ONNX FP32) → Pre-process → Quantize (INT8) → Deploy
```