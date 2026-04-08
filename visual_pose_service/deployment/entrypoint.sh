#!/bin/bash
export PYTHONPATH=/app

DB_PATH="/data/$DB_NAME"
# Normalize backslashes from Windows paths
DB_PATH="${DB_PATH//\\//}"

if [ ! -f "$DB_PATH" ]; then
    echo "[✘] Database not found at $DB_PATH. Please mount your /data folder correctly."
    exit 1
fi

if [ -z "$LOG_DIR" ]; then
    LOG_FLAG=0
    LOG_DIR="none"
else
    LOG_FLAG=1
fi

# Parent directory of the map database file
DB_DIR=$(dirname "$DB_PATH")

echo "[✓] Database found in $DB_PATH, LOG_DIR: $LOG_DIR, LOG_FLAG: $LOG_FLAG"

# Triton gRPC for SuperPoint/SuperGlue (must be reachable from this container)
SP_ADDRESS="${SP_ADDRESS:-192.168.11.194:8001}"
# gRPC listen port for VisualPoseService
PORT="${PORT:-40010}"

echo "Starting server (port=$PORT, sp_address=$SP_ADDRESS)..."
exec python /app/server.pyc \
    --model_path "$DB_DIR" \
    --port "$PORT" \
    --sp_address "$SP_ADDRESS" \
    --log_flag $LOG_FLAG \
    --logs_dir $LOG_DIR \
    --pool_size 6