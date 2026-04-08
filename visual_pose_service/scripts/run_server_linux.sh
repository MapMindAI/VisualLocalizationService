#!/bin/bash
set -e

IMAGE_TAR_FILE="visual_pose_server_image.tar"
IMAGE_NAME="visual_pose_server_image"
IMAGE_PATH="aliyunregistry-gz-mobi-registry.cn-guangzhou.cr.aliyuncs.com/dm/mobili-visual-pose-server-image:presubmit-20250714-bafe3"
CONTAINER_NAME="visual_pose_server_container"

DATA_DIR="/home/dmgz/Data"
DB_NAME="NanshaOffice/database_3d.db"
DB_PATH="$DATA_DIR/$DB_NAME"

MODEL_DIR="/Mobili/python/visual_pose_service/model_repository"

# Server log directory; leave empty to disable file logging
LOG_DIR=""

# Ensure Docker is installed
CUR_DIR=$(dirname "$(realpath "$0")")
bash $CUR_DIR/install_docker_linux.sh

# Pull image if missing locally
if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "$IMAGE_NAME exists in docker!"
else
    echo "$IMAGE_NAME does not exist in docker!"
    echo "Attempting to pull $IMAGE_PATH..."
    docker pull "$IMAGE_PATH"
    if [ $? -eq 0 ]; then
        echo "Successfully pulled the image!"
        docker tag "$IMAGE_PATH" "$IMAGE_NAME"
        echo "Successfully tagged the image as $IMAGE_NAME"
    else
        echo "[ERROR] Fail to pull image!"
        exit 1
    fi
fi

# Require SuperPoint/SuperGlue model repo under MODEL_DIR
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR")" ]; then
    echo "[ERROR] Model directory $MODEL_DIR does not exist or is empty!"
    exit 1
fi

# Launch Triton in a new terminal
echo "[*] Starting Triton Inference Server..."
gnome-terminal -- bash -c "docker run --gpus='device=0' --rm -p8001:8001 --name 'tritonserver' -v $MODEL_DIR:/models aliyunregistry.deepmirror.com.cn/dm/inference-server-app:0.3.1 tritonserver --model-repository=/models; exec bash"

# Require database file; images should live alongside the DB
if [ ! -f "$DB_PATH" ]; then
    echo "[ERROR] Database not found at $DB_PATH"
    echo "Please ensure the data folder contains .db file before running."
    exit 1
fi

# Start visual_pose_server container
echo "[*] Running container from image: $IMAGE_NAME"
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    docker rm -f "$CONTAINER_NAME"
fi
docker run \
    --name $CONTAINER_NAME \
    --network=host \
    -v "$DATA_DIR":/data \
    -v "/home/dmgz/Downloads/entrypoint.sh":/app/deployment/entrypoint.sh \
    -e DB_NAME=$DB_NAME \
    -e LOG_DIR=$LOG_DIR \
    $IMAGE_NAME
