# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# ติดตั้งอุปกรณ์เสริมสำหรับ Compile ไลบรารี
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Run stage
FROM python:3.11-slim

WORKDIR /app

# ติดตั้งเฉพาะ Runtime dependencies สำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# คัดลอกเฉพาะ Libraries ที่ติดตั้งแล้วมาจาก Stage Builder
COPY --from=builder /root/.local /root/.local

# --- ส่วนสำคัญ: เลือกคัดลอกเฉพาะไฟล์ที่ใช้รันจริง ---
COPY src/ ./src/
COPY models/model.onnx ./models/
COPY models/labels.json ./models/

# ตั้งค่า Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# รัน Server โดยชี้ไปที่ main.py ในโฟลเดอร์ src
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]