#!/bin/bash
# Isaac Sim 5.1.0 Cloud Setup — RTX 6000
# Run once SSH'd into the instance
set -e

echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi

echo ""
echo "=== Docker Check ==="
if command -v docker &> /dev/null; then
    echo "Docker: $(docker --version)"
else
    echo "Installing Docker..."
    sudo apt update -qq && sudo apt install -y -qq docker.io
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to re-login for group changes."
fi

echo ""
echo "=== NVIDIA Container Toolkit ==="
if dpkg -l nvidia-container-toolkit &> /dev/null; then
    echo "nvidia-container-toolkit: installed"
else
    echo "Installing nvidia-container-toolkit..."
    sudo apt install -y -qq nvidia-container-toolkit
    sudo systemctl restart docker
fi

echo ""
echo "=== Test GPU Docker ==="
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>&1 | head -15

echo ""
echo "=== NGC Login ==="
# Use the key you already have
NGC_KEY='bW9hdTJmanNidWlwcHJzMjM1dXVibXZzb3Q6OWJhZWM3M2MtZDcyYy00OGY3LTkxYzYtMTU0NjMyYzgyZDNl'
echo "$NGC_KEY" | base64 -d | docker login nvcr.io --username '$oauthtoken' --password-stdin

echo ""
echo "=== Pull Isaac Sim 5.1.0 ==="
docker pull nvcr.io/nvidia/isaac-sim:5.1.0

echo ""
echo "=== Run Demo ==="
docker run --gpus all --rm --network=host \
    -e ACCEPT_EULA=Y \
    -v $(pwd):/workspace \
    --entrypoint /isaac-sim/python.sh \
    nvcr.io/nvidia/isaac-sim:5.1.0 \
    /workspace/demo_cloud.py
