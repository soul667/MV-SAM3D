"""Tests for the MV-SAM3D FastAPI application."""

import io

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal in-memory PNG image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _make_mask_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal in-memory mask PNG (grayscale)."""
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[16:48, 16:48] = 255  # simple square foreground
    img = Image.fromarray(arr, "L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---- Health ----

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "gpu_available" in data


# ---- Reconstruct ----

def test_reconstruct_no_images(client):
    resp = client.post("/api/reconstruct")
    assert resp.status_code == 422  # validation error


def test_reconstruct_with_images_and_masks(client):
    img_bytes = _make_png_bytes()
    mask_bytes = _make_mask_bytes()

    resp = client.post(
        "/api/reconstruct",
        files=[
            ("images", ("0.png", img_bytes, "image/png")),
            ("images", ("1.png", img_bytes, "image/png")),
            ("masks", ("0.png", mask_bytes, "image/png")),
            ("masks", ("1.png", mask_bytes, "image/png")),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "queued"


def test_reconstruct_images_only(client):
    img_bytes = _make_png_bytes()
    resp = client.post(
        "/api/reconstruct",
        files=[("images", ("0.png", img_bytes, "image/png"))],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"


# ---- Segment and Reconstruct ----

def test_segment_and_reconstruct(client):
    img_bytes = _make_png_bytes()
    resp = client.post(
        "/api/segment-and-reconstruct",
        files=[
            ("images", ("0.png", img_bytes, "image/png")),
            ("images", ("1.png", img_bytes, "image/png")),
        ],
        data={"object_prompt": "stuffed toy"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert "stuffed toy" in data["message"]


def test_segment_and_reconstruct_empty_prompt(client):
    img_bytes = _make_png_bytes()
    resp = client.post(
        "/api/segment-and-reconstruct",
        files=[("images", ("0.png", img_bytes, "image/png"))],
        data={"object_prompt": "  "},
    )
    assert resp.status_code == 400


def test_segment_and_reconstruct_no_images(client):
    resp = client.post(
        "/api/segment-and-reconstruct",
        data={"object_prompt": "toy"},
    )
    assert resp.status_code == 422


# ---- Tasks ----

def test_task_not_found(client):
    resp = client.get("/api/tasks/nonexistent-id")
    assert resp.status_code == 404


def test_list_tasks(client):
    resp = client.get("/api/tasks")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_task_status_after_submit(client):
    img_bytes = _make_png_bytes()
    resp = client.post(
        "/api/reconstruct",
        files=[("images", ("0.png", img_bytes, "image/png"))],
    )
    task_id = resp.json()["task_id"]

    # The task should exist now
    status_resp = client.get(f"/api/tasks/{task_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["task_id"] == task_id


# ---- Frontend ----

def test_frontend_served(client):
    resp = client.get("/app/")
    assert resp.status_code == 200
    assert b"MV-SAM3D" in resp.content
