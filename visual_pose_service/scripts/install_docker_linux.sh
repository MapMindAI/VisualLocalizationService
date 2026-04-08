#!/bin/bash
set -e

echo "Checking if Docker is installed..."

DOCKER_INSTALLED=false
if command -v docker &> /dev/null; then
    echo "Docker is installed: $(docker --version)"
    DOCKER_INSTALLED=true
else
    echo "Docker not found; installing (Debian/Ubuntu)..."

    # Update APT index and install prerequisites
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg lsb-release

    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Add Docker apt repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Enable and start Docker
    sudo systemctl enable docker
    sudo systemctl start docker

    echo "Docker installation finished: $(docker --version)"
fi

echo ""
echo "Checking NVIDIA driver..."

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver present:"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader,nounits
else
    echo "NVIDIA driver not found. Install a driver first, e.g.:"
    echo "  sudo ubuntu-drivers autoinstall"
    echo "Or download from NVIDIA's website."
    exit 1
fi

echo ""
echo "Checking NVIDIA Container Toolkit..."

if command -v nvidia-container-runtime &> /dev/null; then
    echo "NVIDIA Container Toolkit present:"
    nvidia-container-runtime --version
    echo "Reconfiguring Docker integration..."
else
    echo "NVIDIA Container Toolkit not found; installing..."

    # Repository and GPG key
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    echo "NVIDIA Container Toolkit installation finished"
fi

echo ""
echo "Configuring Docker runtime..."

sudo nvidia-ctk runtime configure --runtime=docker

echo "Restarting Docker..."
sudo systemctl restart docker

echo ""
echo "Testing NVIDIA Container Toolkit..."

echo "Pulling test image..."
docker pull nvidia/cuda:12.2.0-base-ubuntu20.04

echo "Testing GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "GPU test passed."
    echo ""
    echo "Full test:"
    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi
else
    echo "GPU test failed; trying manual device mapping..."
    if docker run --rm --gpus all --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
        echo "Manual device mapping works."
        echo ""
        echo "Use a command like:"
        echo "docker run --rm --gpus all --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm [IMAGE] [CMD]"
        echo ""
        docker run --rm --gpus all --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi
    else
        echo "GPU test failed. Check:"
        echo "  1. NVIDIA driver installation"
        echo "  2. User in docker group: sudo usermod -aG docker \$USER"
        echo "  3. Re-login or: newgrp docker"
        exit 1
    fi
fi

echo ""
echo "Setup complete."
echo ""
echo "Examples:"
echo "  docker run --rm --gpus all [IMAGE] [CMD]"
echo "  docker run --rm -it --gpus all [IMAGE] bash"
echo "  docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi"
echo ""
echo "Useful commands:"
echo "  docker --version"
echo "  docker info | grep -i runtime"
echo "  nvidia-smi"
