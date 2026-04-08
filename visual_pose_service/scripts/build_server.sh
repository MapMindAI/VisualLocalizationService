#!/bin/bash
set -e

IMAGE_NAME="visual_pose_server_image"
IMAGE_TAR="visual_pose_server_image.tar"

SCRIPT_PATH=$(realpath "$0")
ROOT_DIR=$(dirname $(dirname "$SCRIPT_PATH"))

echo "[*] Building Docker image..."
cd $ROOT_DIR
docker build -f deployment/dockerfile.dev -t $IMAGE_NAME .

echo "[*] Saving Docker image to $IMAGE_TAR..."
docker save -o $IMAGE_TAR $IMAGE_NAME

echo "[*] Done. Distribute $IMAGE_TAR and project folder to deployment machines."
