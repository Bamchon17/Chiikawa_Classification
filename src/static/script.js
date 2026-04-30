const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('image-preview');
const uploadInner = document.getElementById('upload-text-group');

const resultImg = document.getElementById('result-img');
const charName = document.getElementById('char-name');
const confNum = document.getElementById('conf-num');
const progressFill = document.getElementById('progress-fill');

const toast = document.getElementById('status-toast');
const statusMsg = document.getElementById('status-message');
const statusIcon = document.getElementById('status-icon');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

function showNotification(message, type = 'info') {
    statusMsg.innerText = message;
    toast.classList.remove('hidden', 'toast-error', 'toast-success');
    
    if (type === 'error') {
        toast.classList.add('toast-error');
        statusIcon.innerText = '❌';
    } else {
        toast.classList.add('toast-success');
        statusIcon.innerText = '🍃';
    }
    setTimeout(() => toast.classList.add('hidden'), 4000);
}

dropZone.onclick = () => fileInput.click();
fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) handleUpload(file);
};

async function handleUpload(file) {
    // 1. แสดงรูป Preview ทันที
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.classList.remove('hidden');
        uploadInner.classList.add('hidden');
    };
    reader.readAsDataURL(file);

    // 2. เริ่มสถานะ Loading
    loadingOverlay.classList.remove('hidden');
    loadingText.innerText = "กำลังมองหาตัวละครให้คุณนะ...";

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });

        if (!response.ok) {
            const errData = await response.json();
            const errorMessage = errData.detail.suggestion || errData.detail.message || "โอ๊ะโอ! มีบางอย่างผิดพลาด";
            throw new Error(errorMessage);
        }

        const data = await response.json();

        // 3. แสดงผลสำเร็จในกล่อง Result
        resultImg.src = previewImg.src;
        resultImg.classList.remove('hidden');
        charName.innerText = data.prediction;
        
        // อัปเดต Progress Bar พร้อมอนิเมชั่น
        const score = data.confidence;
        confNum.innerText = score.toFixed(2);
        progressFill.style.width = (score * 100) + '%';
        
        showNotification("ทำนายเสร็จแล้ว! น่ารักสุดๆ ✨", "success");

    } catch (error) {
        showNotification(error.message, "error");
        charName.innerText = "Error";
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

// รีเซ็ตค่าทั้งหมด
document.getElementById('predict-btn').onclick = () => {
    fileInput.value = "";
    previewImg.src = "";
    previewImg.classList.add('hidden');
    uploadInner.classList.remove('hidden');
    resultImg.src = "";
    resultImg.classList.add('hidden');
    charName.innerText = "-";
    confNum.innerText = "0.00";
    progressFill.style.width = "0%";
    showNotification("พร้อมเริ่มทำนายใหม่แล้ว 🍃", "success");
};