#!/usr/bin/env bash
# install.sh — Install Box Pose Estimation on Ubuntu 22.04/24.04
# Usage: sudo bash install.sh [--jetson]
set -euo pipefail

INSTALL_DIR=/opt/box-pose
CONFIG_DIR=/etc/box-pose
LOG_DIR=/var/log/box-pose
SERVICE_USER=box-pose
JETSON=false

for arg in "$@"; do
    [[ "$arg" == "--jetson" ]] && JETSON=true
done

echo "==> Installing Box Pose Estimation System"
echo "    Install dir: $INSTALL_DIR"
echo "    Jetson mode:  $JETSON"

# ── System dependencies ────────────────────────────────────────────────────
apt-get update
apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libgl1-mesa-glx libusb-1.0-0 udev curl wget

# ── Intel RealSense SDK ────────────────────────────────────────────────────
if ! dpkg -l | grep -q librealsense2; then
    echo "==> Installing librealsense2"
    mkdir -p /etc/apt/keyrings
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
        | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
        https://librealsense.intel.com/Debian/apt-repo $(. /etc/os-release && echo $VERSION_CODENAME) main" \
        > /etc/apt/sources.list.d/librealsense.list
    apt-get update
    apt-get install -y librealsense2 librealsense2-utils
fi

# ── Create service user ────────────────────────────────────────────────────
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --shell /bin/false \
        --home "$INSTALL_DIR" \
        --groups plugdev,video \
        "$SERVICE_USER"
fi

# ── Install application ────────────────────────────────────────────────────
install -d -m 755 "$INSTALL_DIR" "$CONFIG_DIR" "$LOG_DIR"

# Copy application files
cp -r . "$INSTALL_DIR/"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$LOG_DIR"

# ── Python virtual environment ─────────────────────────────────────────────
python3.11 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip

if [[ "$JETSON" == "true" ]]; then
    "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
else
    "$INSTALL_DIR/venv/bin/pip" install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121
    "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
fi

# ── Download YOLO model ────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR/models"
if [[ ! -f "$INSTALL_DIR/models/yolo11n-seg.pt" ]]; then
    echo "==> Downloading YOLO11n-Seg model weights"
    cd "$INSTALL_DIR"
    "$INSTALL_DIR/venv/bin/python" -c \
        "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')" || true
    mv -f yolo11n-seg.pt models/ 2>/dev/null || true
fi

# ── Configuration ──────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
    cp "$INSTALL_DIR/config/default.yaml" "$CONFIG_DIR/config.yaml"
    echo "==> Config installed to $CONFIG_DIR/config.yaml — edit before starting"
fi

# ── udev rules for RealSense ───────────────────────────────────────────────
if [[ -d /etc/udev/rules.d ]]; then
    wget -qO /etc/udev/rules.d/99-realsense-libusb.rules \
        https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
        || true
    udevadm control --reload-rules
    udevadm trigger
fi

# ── systemd service ────────────────────────────────────────────────────────
cp "$INSTALL_DIR/deploy/systemd/box-pose.service" /etc/systemd/system/
# Patch WorkingDirectory and ExecStart to use actual install path
sed -i "s|/opt/box-pose|$INSTALL_DIR|g" /etc/systemd/system/box-pose.service
systemctl daemon-reload
systemctl enable box-pose.service

echo ""
echo "==> Installation complete!"
echo "    Edit config:   $CONFIG_DIR/config.yaml"
echo "    Start service: systemctl start box-pose"
echo "    View logs:     journalctl -u box-pose -f"
echo "    REST API:      http://localhost:8080/docs"
echo "    Metrics:       http://localhost:9090/metrics"
