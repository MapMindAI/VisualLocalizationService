#!/usr/bin/env bash
# One-launch helper for visual_pose_server on the host (from source).
# Usage:
#   MODEL_PATH=/path/to/map ./scripts/run_vlserver.sh
#   ./scripts/run_vlserver.sh --model_path /path/to/map --sp_address 127.0.0.1:8001
# Environment (used when no CLI args are passed):
#   MODEL_PATH     Required. Directory containing database_3d.db (and map assets).
#   PORT           gRPC port (default: 40010)
#   SP_ADDRESS     Triton gRPC host:port (default: 127.0.0.1:8001)
#   POOL_SIZE      Worker processes (default: 4)
#   TOP_K          Retrieval top-k (default: 3)
#   MAX_WORKERS    gRPC thread pool (default: 10)
#   LOG_LEVEL      DEBUG|INFO|WARNING|ERROR (default: INFO)
#   LOG_FLAG       0|1 — 1 saves extra debug images/logs (default: 0)
#   LOGS_DIR       Used when LOG_FLAG=1 (default: ./logs/visual_pose_server)
#   SP_MODEL_NAME  Triton model name for SuperPoint (default: superpoint_onnx)
#   SG_MODEL_NAME  Triton model name for SuperGlue/LightGlue (default: lightglue_onnx)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$VPS_ROOT"

SERVER_PY="${VPS_ROOT}/visual_pose_server.py"

if [[ $# -gt 0 ]]; then
  exec python3 "$SERVER_PY" "$@"
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Set MODEL_PATH to your map directory (must contain database_3d.db), e.g.:"
  echo "  MODEL_PATH=/data/my_map ./scripts/run_vlserver.sh"
  echo "Or pass arguments directly:"
  echo "  ./scripts/run_vlserver.sh --model_path /data/my_map --sp_address 127.0.0.1:8001"
  exit 1
fi

exec python3 "$SERVER_PY" \
  --model_path "$MODEL_PATH" \
  --port "${PORT:-40010}" \
  --sp_address "${SP_ADDRESS:-127.0.0.1:8001}" \
  --pool_size "${POOL_SIZE:-4}" \
  --top_k "${TOP_K:-3}" \
  --max_workers "${MAX_WORKERS:-10}" \
  --log_level "${LOG_LEVEL:-INFO}" \
  --log_flag "${LOG_FLAG:-0}" \
  --logs_dir "${LOGS_DIR:-${VPS_ROOT}/logs/visual_pose_server}" \
  --sp_model_name "${SP_MODEL_NAME:-superpoint_onnx}" \
  --sg_model_name "${SG_MODEL_NAME:-lightglue_onnx}"
