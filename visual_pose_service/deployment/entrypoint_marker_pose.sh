#!/bin/bash
export PYTHONPATH=/app

# Marker config file path in /data (similar to DB_NAME for visual_pose_server)
MARKER_CONFIG_PATH="/data/$MARKER_CONFIG_NAME"
# Normalize backslashes from Windows paths
MARKER_CONFIG_PATH="${MARKER_CONFIG_PATH//\\//}"

if [ -z "$LOG_DIR" ]; then
    LOG_FLAG=0
    LOG_DIR="none"
else
    LOG_FLAG=1
fi

# Check if marker config file exists (optional, will use default if not provided)
if [ -n "$MARKER_CONFIG_NAME" ] && [ ! -f "$MARKER_CONFIG_PATH" ]; then
    echo "[✘] Marker config not found at $MARKER_CONFIG_PATH. Will use default configuration."
    MARKER_CONFIG_NAME=""  # Clear so it won't be passed as argument
elif [ -n "$MARKER_CONFIG_NAME" ] && [ -n "$MARKER_CONFIG_PATH" ] && [ -f "$MARKER_CONFIG_PATH" ]; then
    echo "[✓] Marker config found at $MARKER_CONFIG_PATH"
else
    echo "[✓] No marker config specified, using default configuration"
    MARKER_CONFIG_NAME=""  # Clear so it won't be passed as argument
fi

echo "[✓] Starting marker_pose_server..."
echo "    LOG_DIR: $LOG_DIR, LOG_FLAG: $LOG_FLAG"
echo "    Port: ${PORT:-40011}"

# Build command arguments
CMD_ARGS=(
    "--port" "${PORT:-40011}"
    "--log_flag" "$LOG_FLAG"
    "--logs_dir" "$LOG_DIR"
    "--pool_size" "${POOL_SIZE:-4}"
    "--max_workers" "${MAX_WORKERS:-10}"
    "--log_level" "${LOG_LEVEL:-INFO}"
)

# Add marker_config argument if provided
if [ -n "$MARKER_CONFIG_NAME" ] && [ -f "$MARKER_CONFIG_PATH" ]; then
    CMD_ARGS=("--marker_config" "$MARKER_CONFIG_PATH" "${CMD_ARGS[@]}")
fi

exec python /app/marker_pose_service/marker_pose_server.pyc "${CMD_ARGS[@]}"

