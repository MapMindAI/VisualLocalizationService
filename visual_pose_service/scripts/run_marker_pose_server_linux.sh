#!/bin/bash
set -e

IMAGE_TAR_FILE="visual_pose_server_image.tar"
IMAGE_NAME="visual_pose_server_image"
IMAGE_PATH="aliyunregistry-gz-mobi-registry.cn-guangzhou.cr.aliyuncs.com/dm/mobili-visual-pose-server-image:presubmit-20250714-bafe3"
CONTAINER_NAME="marker_pose_server_container"

DATA_DIR="/home/dmgz/Data"
MARKER_CONFIG_NAME="marker_config.json"  # Marker config file name in DATA_DIR (optional)
MARKER_CONFIG_PATH="$DATA_DIR/$MARKER_CONFIG_NAME"
MARKER_IMAGE_DIR="$DATA_DIR/marker_image"  # Marker image directory in DATA_DIR

# Server port
PORT="${PORT:-40011}"

# Pool size and max workers
POOL_SIZE="${POOL_SIZE:-4}"
MAX_WORKERS="${MAX_WORKERS:-10}"

# Log level
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Server log directory; leave empty to disable file logging
LOG_DIR="${LOG_DIR:-}"

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

# Optional marker config; fall back to defaults if missing
if [ -n "$MARKER_CONFIG_NAME" ] && [ ! -f "$MARKER_CONFIG_PATH" ]; then
    echo "[WARNING] Marker config not found at $MARKER_CONFIG_PATH. Will use default configuration."
    MARKER_CONFIG_NAME=""
fi

# Warn if marker_image directory is missing
if [ ! -d "$MARKER_IMAGE_DIR" ]; then
    echo "[WARNING] Marker image directory not found at $MARKER_IMAGE_DIR"
    echo "Please create the directory and place marker images there."
fi

# Start marker_pose_server container
echo "[*] Running marker_pose_server container from image: $IMAGE_NAME"
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    docker rm -f "$CONTAINER_NAME"
fi

docker run \
    --name $CONTAINER_NAME \
    --network=host \
    -p ${PORT}:${PORT} \
    -v "$DATA_DIR":/data \
    -v "/home/dmgz/Downloads/entrypoint_marker_pose.sh":/app/deployment/entrypoint_marker_pose.sh \
    -e MARKER_CONFIG_NAME="$MARKER_CONFIG_NAME" \
    -e PORT="$PORT" \
    -e POOL_SIZE="$POOL_SIZE" \
    -e MAX_WORKERS="$MAX_WORKERS" \
    -e LOG_LEVEL="$LOG_LEVEL" \
    -e LOG_DIR="$LOG_DIR" \
    $IMAGE_NAME \
    bash deployment/entrypoint_marker_pose.sh

