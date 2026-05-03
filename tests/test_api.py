from io import BytesIO
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.main as main


client = TestClient(main.app)
TEST_DATA_DIR = PROJECT_ROOT / "dataset" / "test"


def make_png_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint_returns_json():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "online",
        "message": "Chiikawa API is ready!",
    }


def test_root_serves_frontend_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/static/style.css" in response.text
    assert "/static/script.js" in response.text


def test_static_css_is_available():
    response = client.get("/static/style.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


@pytest.mark.parametrize(
    ("image_name", "expected_prediction"),
    [
        ("chiikawa head.png", "chiikawa"),
        ("hachiware head.png", "hachiware"),
        ("Usagi-pointing.jpg", "usagi"),
    ],
)
def test_predict_with_real_dataset_images(image_name, expected_prediction):
    image_path = TEST_DATA_DIR / image_name

    assert image_path.exists(), f"Missing test image: {image_path}"

    content_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    response = client.post(
        "/predict",
        files={"file": (image_path.name, image_path.read_bytes(), content_type)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == image_path.name
    assert data["prediction"] == expected_prediction
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["message"], str)


def test_predict_rejects_non_image_file_type():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["received_type"] == "text/plain"


def test_predict_rejects_corrupted_image():
    response = client.post(
        "/predict",
        files={"file": ("broken.png", b"not a real png", "image/png")},
    )

    assert response.status_code == 422


def test_predict_rejects_file_larger_than_5mb():
    oversized_content = b"0" * (5 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={"file": ("large.png", oversized_content, "image/png")},
    )

    assert response.status_code == 413
