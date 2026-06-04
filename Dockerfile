# Box Pose Estimation System — Production Dockerfile
# Base: CUDA 12.1 + Ubuntu 22.04 for x86 GPU workstation / server
# For Jetson: use Dockerfile.jetson

FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    # OpenCV runtime
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Open3D dependencies
    libgl1-mesa-glx \
    libglu1-mesa \
    # USB access for RealSense
    udev \
    libusb-1.0-0 \
    # General
    wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── RealSense SDK ──────────────────────────────────────────────────────────
# Install librealsense2 from Intel repository
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
        | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
        https://librealsense.intel.com/Debian/apt-repo $(. /etc/os-release && echo $VERSION_CODENAME) main" \
        | tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y librealsense2=2.55.1* librealsense2-utils=2.55.1* && \
    rm -rf /var/lib/apt/lists/*

# ── Python environment ─────────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY . .

# Create runtime directories
RUN mkdir -p logs output/frames models

# Download YOLO11 model if not bundled
# (In production, mount models/ volume with pre-downloaded weights)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')" || true
RUN mv -f yolo11n-seg.pt models/ 2>/dev/null || true

# ── Runtime configuration ──────────────────────────────────────────────────
EXPOSE 8080 9090

# udev rules for RealSense USB access
COPY deploy/udev/99-realsense-libusb.rules /etc/udev/rules.d/ 2>/dev/null || true

ENV BOX_POSE__LOGGING__FORMAT=json \
    BOX_POSE__VISUALIZATION__ENABLED=false \
    BOX_POSE__API__HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health | python3 -c \
    "import sys, json; d=json.load(sys.stdin); sys.exit(0 if d['status']!='unhealthy' else 1)" || exit 1

CMD ["python3", "main.py", "--config", "config/default.yaml", "--no-vis"]
