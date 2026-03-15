"""Pydantic schemas for the MV-SAM3D API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.0.1"
    gpu_available: bool = False


class TaskResponse(BaseModel):
    """Response after submitting a reconstruction task."""
    task_id: str
    status: str = "processing"
    message: str = ""


class TaskStatusResponse(BaseModel):
    """Response for querying task status."""
    task_id: str
    status: str
    message: str = ""
    result_url: Optional[str] = None
    preview_url: Optional[str] = None


class ReconstructRequest(BaseModel):
    """Request body for reconstruction with text prompt (JSON mode)."""
    object_prompt: str = Field(..., description="Text description of the object to segment")
    mask_prompt: Optional[str] = Field(None, description="Mask prompt name for pre-existing masks")


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
