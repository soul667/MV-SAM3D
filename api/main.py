"""
MV-SAM3D FastAPI Application

Provides REST API endpoints for multi-view 3D reconstruction:
- Upload multiple images with masks for direct reconstruction
- Upload multiple images with a text prompt for automatic segmentation + reconstruction
- Query task status and download results
"""

import io
import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    TaskResponse,
    TaskStatusResponse,
    ErrorResponse,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MV-SAM3D API",
    description="Multi-view 3D reconstruction API with mask and text-prompt support",
    version="0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Simple in-memory task store (task_id -> status dict)
_tasks: dict = {}

# Lazy-loaded inference pipeline (heavy; only load when needed)
_pipeline = None
_pipeline_lock = asyncio.Lock()

GROUNDED_SAM2_URL = os.environ.get("GROUNDED_SAM2_URL", "http://grounded-sam2:8080")

# Mount outputs directory for serving result files
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount frontend directory if it exists
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


async def _get_pipeline():
    """Lazy-load the inference pipeline (thread-safe)."""
    global _pipeline
    async with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            import sys
            sys.path.insert(0, "notebook")
            from inference import Inference

            config_path = os.environ.get(
                "SAM3D_CONFIG", "checkpoints/sam3d_config.yaml"
            )
            _pipeline = Inference(config_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load inference pipeline: {exc}. "
                "Make sure model checkpoints are available."
            ) from exc
    return _pipeline


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    """Save an uploaded file to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(upload.file.read())
    return dest


def _load_image_array(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.uint8)


def _load_mask_array(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if img.mode == "RGBA":
        return arr[..., 3] > 0
    if arr.ndim == 3:
        return arr[..., -1] > 0
    return arr > 0


async def _segment_with_grounded_sam2(
    image_path: Path, prompt: str
) -> Optional[np.ndarray]:
    """
    Call the Grounded-SAM-2 service to generate a segmentation mask.

    The external service is expected to accept a multipart POST with
    an image file and a text prompt, returning a PNG mask.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(image_path, "rb") as f:
                resp = await client.post(
                    f"{GROUNDED_SAM2_URL}/segment",
                    files={"image": (image_path.name, f, "image/png")},
                    data={"prompt": prompt},
                )
            if resp.status_code == 200:
                mask_img = Image.open(
                    io.BytesIO(resp.content)
                )
                return np.array(mask_img) > 0
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------


async def _run_reconstruction(task_id: str, task_dir: Path, mode: str):
    """Run inference in the background and update task status."""
    try:
        _tasks[task_id]["status"] = "running"

        images_dir = task_dir / "images"
        masks_dir = task_dir / "masks"
        result_dir = OUTPUTS_DIR / task_id

        image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        mask_paths = sorted(masks_dir.glob("*.png")) if masks_dir.exists() else []

        if not image_paths:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["message"] = "No images found"
            return

        # If mode is "prompt" and no masks yet, try to generate them
        if mode == "prompt" and not mask_paths:
            prompt = _tasks[task_id].get("object_prompt", "")
            if prompt:
                masks_dir.mkdir(parents=True, exist_ok=True)
                for img_path in image_paths:
                    mask = await _segment_with_grounded_sam2(img_path, prompt)
                    if mask is not None:
                        mask_uint8 = (mask.astype(np.uint8) * 255)
                        if mask_uint8.ndim == 3:
                            mask_uint8 = mask_uint8[..., 0]
                        mask_img = Image.fromarray(mask_uint8, mode="L")
                        mask_img.save(masks_dir / img_path.name)
                mask_paths = sorted(masks_dir.glob("*.png"))

        # Attempt to run pipeline if available
        result_dir.mkdir(parents=True, exist_ok=True)

        try:
            pipeline = await _get_pipeline()

            images = [_load_image_array(p) for p in image_paths]
            masks = [_load_mask_array(p) for p in mask_paths] if mask_paths else [None] * len(images)

            # NOTE: Currently runs single-view reconstruction on the first image.
            # Full multi-view weighted fusion requires Depth Anything 3 (DA3) pointmaps
            # and should be invoked via run_inference_weighted.py for production use.
            result = pipeline(images[0], masks[0] if masks else None)

            # Save result GLB if available
            if result and "gaussian" in result:
                _tasks[task_id]["status"] = "completed"
                _tasks[task_id]["message"] = "Reconstruction completed"
                _tasks[task_id]["result_url"] = f"/outputs/{task_id}"
            else:
                _tasks[task_id]["status"] = "completed"
                _tasks[task_id]["message"] = "Pipeline returned no gaussian output"

        except RuntimeError:
            # Pipeline not available — save uploaded data for reference
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["message"] = (
                "Model checkpoint not loaded. "
                "Uploaded data saved. Run inference manually with run_inference_weighted.py"
            )

        # Copy images/masks to result dir for preview
        import shutil
        if images_dir.exists():
            shutil.copytree(images_dir, result_dir / "images", dirs_exist_ok=True)
        if masks_dir.exists():
            shutil.copytree(masks_dir, result_dir / "masks", dirs_exist_ok=True)

        if "result_url" not in _tasks[task_id] or not _tasks[task_id].get("result_url"):
            _tasks[task_id]["result_url"] = f"/outputs/{task_id}"

    except Exception as exc:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["message"] = str(exc)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.0.1",
        gpu_available=_gpu_available(),
    )


@app.post("/api/reconstruct", response_model=TaskResponse)
async def reconstruct(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(..., description="Input images (PNG/JPG)"),
    masks: List[UploadFile] = File(
        default=[], description="Corresponding mask images (PNG, RGBA alpha or grayscale)"
    ),
):
    """
    Reconstruct a 3D object from multiple images and their masks.

    Upload multiple view images along with their corresponding segmentation masks.
    Masks should be PNG images where non-zero pixels indicate the foreground object.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    task_id = str(uuid.uuid4())
    task_dir = OUTPUTS_DIR / f".tmp_{task_id}"
    images_dir = task_dir / "images"
    masks_dir = task_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i, img_file in enumerate(images):
        ext = Path(img_file.filename or f"{i}.png").suffix or ".png"
        _save_upload(img_file, images_dir / f"{i}{ext}")

    for i, mask_file in enumerate(masks):
        ext = Path(mask_file.filename or f"{i}.png").suffix or ".png"
        _save_upload(mask_file, masks_dir / f"{i}{ext}")

    _tasks[task_id] = {
        "status": "queued",
        "message": "Task queued for reconstruction",
        "task_dir": str(task_dir),
    }

    background_tasks.add_task(_run_reconstruction, task_id, task_dir, "masks")

    return TaskResponse(
        task_id=task_id,
        status="queued",
        message=f"Reconstruction task created with {len(images)} images and {len(masks)} masks",
    )


@app.post("/api/segment-and-reconstruct", response_model=TaskResponse)
async def segment_and_reconstruct(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(..., description="Input images (PNG/JPG)"),
    object_prompt: str = Form(..., description="Text description of the object to segment"),
):
    """
    Automatically segment an object from images using a text prompt, then reconstruct in 3D.

    This endpoint uses the Grounded-SAM-2 service to generate segmentation masks from
    the provided text prompt, then runs the MV-SAM3D reconstruction pipeline.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    if not object_prompt.strip():
        raise HTTPException(status_code=400, detail="object_prompt is required")

    task_id = str(uuid.uuid4())
    task_dir = OUTPUTS_DIR / f".tmp_{task_id}"
    images_dir = task_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, img_file in enumerate(images):
        ext = Path(img_file.filename or f"{i}.png").suffix or ".png"
        _save_upload(img_file, images_dir / f"{i}{ext}")

    _tasks[task_id] = {
        "status": "queued",
        "message": "Task queued for segmentation and reconstruction",
        "object_prompt": object_prompt,
        "task_dir": str(task_dir),
    }

    background_tasks.add_task(_run_reconstruction, task_id, task_dir, "prompt")

    return TaskResponse(
        task_id=task_id,
        status="queued",
        message=f"Segmentation + reconstruction task created with {len(images)} images, prompt: '{object_prompt}'",
    )


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Query the status of a reconstruction task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        message=task.get("message", ""),
        result_url=task.get("result_url"),
    )


@app.get("/api/tasks")
async def list_tasks():
    """List all tasks."""
    return {
        tid: {
            "status": t.get("status"),
            "message": t.get("message", ""),
            "result_url": t.get("result_url"),
        }
        for tid, t in _tasks.items()
    }
