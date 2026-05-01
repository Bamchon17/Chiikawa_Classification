const fileInput = document.getElementById('file-input');
const predictBtn = document.getElementById('predict-btn');

// ฟังก์ชันส่งรูปไป Predict
async function uploadImage(file) {
    // Validation: ตรวจสอบขนาดไฟล์ตามโจทย์ MLOps
    if (file.size > 5 * 1024 * 1024) { 
        alert("ไฟล์ใหญ่เกินไป! กรุณาใช้ไฟล์ไม่เกิน 5MB");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.status === 200) {
            updateUI(result.prediction, result.confidence);
        } else {
            alert("เกิดข้อผิดพลาด: " + result.detail);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("ไม่สามารถติดต่อ Server ได้");
    }
}

function updateUI(name, conf) {
    document.getElementById('winner-name').innerText = name;
    const confPercent = (conf * 100).toFixed(0) + "%";
    document.getElementById('confidence-fill').style.width = confPercent;
    document.getElementById('confidence-text').innerText = `${conf.toFixed(2)} / 1.00`;
}