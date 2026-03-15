# ==============================================================================
# MV-SAM3D Docker Image
# Multi-view 3D reconstruction with FastAPI serving
# ==============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget curl build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libglfw3-dev libgles2-mesa-dev libegl1-mesa-dev \
    libboost-all-dev libfreeimage-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ==============================================================================
# Install PyTorch with CUDA 12.1
# ==============================================================================
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ==============================================================================
# Install project dependencies
# ==============================================================================
WORKDIR /app

# Copy requirements files first for better caching
COPY requirements.txt requirements.inference.txt requirements.p3d.txt requirements.api.txt ./

# Install main requirements (skip packages that need special handling)
RUN pip install --no-cache-dir -r requirements.api.txt

# Install main requirements (best-effort; some packages may need CUDA at build)
RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.ngc.nvidia.com \
    -r requirements.txt || true

# Install inference requirements
RUN pip install --no-cache-dir -r requirements.inference.txt || true

# Install PyTorch3D and flash attention
RUN pip install --no-cache-dir -r requirements.p3d.txt || true

# ==============================================================================
# Copy application code
# ==============================================================================
COPY . .

# Install the sam3d_objects package
RUN pip install -e . || true

# ==============================================================================
# Expose ports and set entrypoint
# ==============================================================================
# FastAPI server port
EXPOSE 8000
# Frontend static file server port
EXPOSE 3000

# Default command: start the FastAPI server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
