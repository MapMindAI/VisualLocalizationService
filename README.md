# VisualLocalizationService

gRPC service for visual localization: the server estimates camera pose using a prebuilt map and a Triton-hosted SuperPoint/SuperGlue stack. This repository includes a **test client** (`visual_pose_client.py`) for smoke tests and load experiments.

---

## What you need before running

1. **Map directory** (`--model_path`): must contain the localization database and assets (at minimum `database_3d.db` and related data expected by `visual_localizer.py` — see `visual_pose_service/README.md` for map preparation).
2. **Triton Inference Server** with the SuperPoint / SuperGlue models expected by this codebase (model names such as `superpoint_trt`; see `visual_pose_service/feature/superpoint.py`). The server connects to Triton at **`--sp_address`** (gRPC).
3. **Python** (3.9+ recommended) and dependencies from `visual_pose_service/requirements.txt` if you run from source.

---

## Deploy the server (host, from source)

This is the most straightforward path and matches the Python entrypoint.

```bash
cd visual_pose_service
export PYTHONPATH="$(pwd)"

python3 visual_pose_server.py \
  --model_path /absolute/path/to/your/map \
  --port 40010 \
  --sp_address YOUR_TRITON_HOST:8001 \
  --pool_size 4 \
  --top_k 3
```

- **`--model_path`**: directory that contains `database_3d.db` (not the file path alone).
- **`--sp_address`**: host and gRPC port where Triton serves the feature models (must be reachable from the machine running the server).
- **`--port`**: gRPC port for **this** service (default `40010`).

The server listens on **`0.0.0.0:<port>`** (see `visual_pose_server.py`).

---

## Call the server from the test client

The bundled client is **not** a production SDK; it is **hard-coded** for image path and intrinsics. To get a **working** call against your server:

1. **Match the address and port**  
   In `visual_pose_service/visual_pose_client.py`, set **`SINGLE_CLIENT_ADDRESS`** and **`MULTI_CLIENTS_ADDRESS`** to `host:port` of the **visual pose server** (e.g. `127.0.0.1:40010` if local).

2. **Point to a real image**  
   Set **`self.folder`** (or **`self.image_path`**) to a JPEG/PNG that exists on disk (defaults under `/Mobili/...` will not work unless you create them).

3. **Align intrinsics (optional but recommended)**  
   `create_request()` uses fixed `fx, fy` and `cx, cy` from `cv2.imread(self.image_path).shape` — adjust for your camera if you care about accuracy.

4. **Run the client** from `visual_pose_service` with the same `PYTHONPATH`:

```bash
cd visual_pose_service
export PYTHONPATH="$(pwd)"
python3 visual_pose_client.py
```

Follow the prompts (interval, single vs multi-client). The client calls **`GetPoseFromImage`** over gRPC.

---

## Deploy the server (Docker)

Build (from `visual_pose_service`):

```bash
docker build -f deployment/dockerfile.dev -t visual-pose-service:dev .
```

Run (example: map data on host at `./mapdata`, DB file at `./mapdata/session/database_3d.db`):

```bash
docker run --rm \
  --network host \
  -e DB_NAME="session/database_3d.db" \
  -e SP_ADDRESS="127.0.0.1:8001" \
  -e PORT="40010" \
  -v "$(pwd)/mapdata:/data" \
  visual-pose-service:dev
```

- **`DB_NAME`**: path **inside** `/data` to the **database file** (e.g. `myrun/database_3d.db`). The entrypoint checks that **`/data/$DB_NAME`** exists.
- **`SP_ADDRESS`**: Triton gRPC URL **as seen from inside the container** (use `host.docker.internal`, host LAN IP, or `--network host` with `127.0.0.1` if Triton is on the host).
- **`PORT`**: gRPC port for **this** service (default `40010`).

Optional file logging: set **`LOG_DIR`** to a writable path (e.g. mount a volume and pass `-e LOG_DIR=/logs`).

Then call the server from the host test client using `127.0.0.1:40010` if you used `--network host`, or the published host port if you map ports instead.

---

## Production clients

Use **gRPC** and the **`GetPoseFromImage`** RPC. **Protobuf stubs** live under `visual_pose_service/proto/` (Python). Original `.proto` files are not in this repo; generate stubs for other languages from your team’s `.proto` sources.

---

## Other modules

- Map alignment (QR codes): [`visual_pose_service/map_alignment/README.md`](visual_pose_service/map_alignment/README.md)
- Marker pose (separate gRPC service, different port): `visual_pose_service/marker_pose_service/`
