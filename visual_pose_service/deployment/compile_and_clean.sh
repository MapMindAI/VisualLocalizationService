#!/bin/bash
set -e

echo "[*] Compiling Python files..."
python -m compileall -b .

echo "[*] Locating compiled server entry..."
# Find compiled visual_pose_server .pyc
FOUND_PYC=$(find . -type f -name "visual_pose_server*.pyc" | head -n 1)

if [ -z "$FOUND_PYC" ]; then
    echo "[✘] Compiled .pyc file not found."
    exit 1
fi

cp "$FOUND_PYC" /app/server.pyc

echo "[*] Removing source .py files..."
find . -name "*.py" ! -name "entrypoint.sh" ! -path "./deployment/*" -type f -delete

echo "[✓] Done."