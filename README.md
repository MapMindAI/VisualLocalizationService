# VisualLocalizationService

gRPC service for visual localization: the server estimates camera pose using a prebuilt map and a Triton-hosted SuperPoint/SuperGlue stack. This repository includes a **test client** (`visual_pose_client.py`) for smoke tests and load experiments.

---

## Prerequisites

1. **Map directory** (`--model_path`): must contain the localization database and assets (at minimum `database_3d.db` and related data expected by `visual_localizer.py` — see `visual_pose_service/README.md` for map preparation).
2. **Triton Inference Server** with the SuperPoint / SuperGlue models expected by this codebase (model names such as `superpoint_trt`; see `visual_pose_service/feature/superpoint.py`). The server connects to Triton at **`--sp_address`** (gRPC).

---

## Environment with Conda (recommended)

Create and activate an environment, then install Python dependencies:

```bash
conda create -n vlpose python=3.9 -y
conda activate vlpose
cd visual_pose_service
pip install -r requirements.txt
```

On **Linux**, if OpenCV or Open3D fail to load native libraries, install the usual graphics/runtime packages for your distro (for example `libgl1`, `libglib2.0-0`, `libgomp1` on Debian/Ubuntu).

Always run commands from `visual_pose_service` **or** set:

```bash
export PYTHONPATH="/absolute/path/to/VisualLocalizationService/visual_pose_service"
```

---

## Run the visual pose server

### Option A: helper script (recommended)

From `visual_pose_service` after `conda activate` and `pip install -r requirements.txt`:

```bash
chmod +x scripts/run_vlserver.sh   # once
MODEL_PATH=/absolute/path/to/your/map \
SP_ADDRESS=YOUR_TRITON_HOST:8001 \
./scripts/run_vlserver.sh
```

Defaults: `PORT=40010`, `SP_ADDRESS=127.0.0.1:8001`, `POOL_SIZE=4`, `TOP_K=3`. See comment header in `scripts/run_vlserver.sh` for all environment variables.

You can also pass arguments directly to the server (same as `visual_pose_server.py`):

```bash
./scripts/run_vlserver.sh --model_path /path/to/map --sp_address 127.0.0.1:8001 --port 40010
```

### Option B: run Python directly

```bash
conda activate vlpose
cd visual_pose_service
export PYTHONPATH="$(pwd)"

python visual_pose_server.py \
  --model_path /absolute/path/to/your/map \
  --port 40010 \
  --sp_address YOUR_TRITON_HOST:8001 \
  --pool_size 4 \
  --top_k 3
```

- **`--model_path`**: directory that contains `database_3d.db` (the directory, not the file path).
- **`--sp_address`**: Triton gRPC `host:port` reachable from this machine.
- **`--port`**: gRPC port for this service (default `40010`).

The server listens on **`0.0.0.0:<port>`**.

---

## Call the server with the test client

The bundled client is **not** a production SDK; paths and intrinsics are **hard-coded**.

1. Set **`SINGLE_CLIENT_ADDRESS`** / **`MULTI_CLIENTS_ADDRESS`** in `visual_pose_service/visual_pose_client.py` to your server `host:port`.
2. Set **`self.folder`** / **`self.image_path`** to a real image file.
3. Run:

```bash
conda activate vlpose
cd visual_pose_service
export PYTHONPATH="$(pwd)"
python visual_pose_client.py
```

---

## Production clients

Use **gRPC** and **`GetPoseFromImage`**. Python stubs live under `visual_pose_service/proto/`. Original `.proto` files are not in this repository.

---

## Related tools (not part of the localization server)

- **Marker pose** — separate gRPC service for fiducial/marker pose: [`visual_pose_service/marker_pose_service/README.md`](visual_pose_service/marker_pose_service/README.md)
