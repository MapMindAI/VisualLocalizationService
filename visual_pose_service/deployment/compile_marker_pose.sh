#!/bin/bash
set -e

echo "[*] Compiling marker_pose_server Python files..."
cd /app/marker_pose_service
python -m compileall -b .

echo "[*] Verifying compiled marker_pose_server..."
# Recursively collect .pyc files
FOUND_PYC=$(find . -type f -name "marker_pose_server*.pyc" | head -n 1)

if [ -z "$FOUND_PYC" ]; then
    echo "[✘] Compiled marker_pose_server .pyc file not found."
    exit 1
fi

echo "[✓] Found marker_pose_server.pyc at: $FOUND_PYC"
echo "[*] Keeping marker_pose_server.pyc in /app/marker_pose_service/ for proper imports"

echo "[*] Removing marker_pose_service source .py files..."
# Keep config/ and tools/ directories
# marker_image/ will be mounted from /data at runtime, so we can remove it from image
# Keep .pyc files so that marker_pose_estimator can be imported
find . -name "*.py" ! -path "./tools/*" ! -path "./config/*" -type f -delete

cd /app
echo "[✓] marker_pose_server compilation done."
echo "[*] marker_pose_server.pyc is in /app/marker_pose_service/ for proper module imports"

