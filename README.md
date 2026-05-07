# VisualLocalizationService

gRPC service for visual localization: the server estimates camera pose from an encoded query image, camera intrinsics, and an optional localization prior. It uses a prebuilt map, BoW retrieval, SuperPoint, LightGlue, and PnP with optional gravity prior. This repository also includes a **test client** (`visual_pose_client.py`) for smoke tests and simple load experiments.

---

## Prerequisites

1. **Map directory** (`--model_path`): must contain the localization database, database images, and BoW assets (see [Map and database preparation](#map-and-database-preparation) and [Bag-of-words retrieval](#bag-of-words-bow-retrieval)).
2. **Triton Inference Server** with the EasyTensorRT SuperPoint / LightGlue models expected by this codebase. Defaults are `superpoint_onnx` and `lightglue_onnx`; TensorRT deployments usually use `superpoint_trt` and `lightglue_trt`. The server connects to Triton at **`--sp_address`** (gRPC).

---

## Map and database preparation

**Building the map and producing the SQLite / COLMAP-style assets used by this service (including paths toward `database_3d.db`) is documented and automated in a separate pipeline:**

[**MapMindAI / EasyGaussianSplatting**](https://github.com/MapMindAI/EasyGaussianSplatting)

That repository provides an end-to-end Gaussian Splatting workflow (e.g. drone / 360 data, Docker or local environment, scripts under `mindmap/`). After the pipeline finishes, you should have a **session / map folder** on disk that your team aligns with what `visual_localizer.py` expects. The current implementation reads:

```text
/path/to/map/
  database_3d.db
  images/
  retrieval_vocab_kmeans.pkl
  retrieval_ids.npy
  retrieval_bow_vectors.npy
  output/tsdf_mesh.ply        # only needed when LOG_FLAG=1 draws mesh projections
```

Any extra steps to fill poses, 3D points from depth, and keypoints into the database are part of **your mapping pipeline** (or tooling shipped with EasyGaussianSplatting); this service repo does not re-run full map building.

Example of reconstruction result:

<table>
  <tr>
    <td align="center" width="50%"><img src="asset/gs_img.png" alt="GS reconstruction example" width="400"/></td>
    <td align="center" width="50%"><img src="asset/colmap.gif" alt="COLMAP animation" width="400"/></td>
  </tr>
</table>


---

## Bag-of-words (BoW) retrieval

**BoW retrieval files are produced by the [EasyGaussianSplatting](https://github.com/MapMindAI/EasyGaussianSplatting) map pipeline** alongside `database_3d.db` and the rest of the map layout. After a successful run, the map root you pass to this service as `--model_path` should contain `retrieval_vocab_kmeans.pkl`, `retrieval_ids.npy`, and `retrieval_bow_vectors.npy` — no separate BoW step is required in the usual workflow.

If you need to **regenerate** retrieval data (e.g. after manual edits to the database, or when debugging), you can run:

```bash
conda activate vlpose
cd visual_pose_service
export PYTHONPATH="$(pwd)"

python retrieval/make_retrieval_db.py --model_path /path/to/your/map
```

---

## Environment with Conda (recommended)

Create and activate an environment, then install Python dependencies:

```bash
conda create -n vlpose python=3.10 -y
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

## Run with Docker Compose

Requires **Docker Compose** — install it following the [official guide](https://docs.docker.com/compose/install/).

Docker Compose starts both `visual_pose_server` and a Triton server together. Choose the Triton backend via `--profile`.

**Required environment variables:**

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Map root directory (must contain `database_3d.db`) |
| `TRITON_MODEL_DIR` | Triton model directory — clone [MapMindAI/EasyTensorRT](https://github.com/MapMindAI/EasyTensorRT) and point to the cloned path |

```bash
git clone https://github.com/MapMindAI/EasyTensorRT
```

```bash
# CPU (no GPU required)
MODEL_PATH=/path/to/map TRITON_MODEL_DIR=/path/to/EasyTensorRT \
  docker compose --profile cpu up

# ONNX + GPU
MODEL_PATH=/path/to/map TRITON_MODEL_DIR=/path/to/EasyTensorRT \
  docker compose --profile onnx up

# TensorRT + GPU (converts models on first launch)
MODEL_PATH=/path/to/map TRITON_MODEL_DIR=/path/to/EasyTensorRT \
SP_MODEL_NAME=superpoint_trt SG_MODEL_NAME=lightglue_trt \
  docker compose --profile trt up
```

The `visual_pose_server` image is built automatically on first launch; subsequent runs reuse the cached image. To force a rebuild: `docker compose build visual_pose_server`.

Optional overrides (pass as environment variables): `VPS_PORT` (host port, default `40010`), `SP_ADDRESS` (default `tritonserver:8001` inside Compose), `POOL_SIZE` (default `4`), `TOP_K` (default `3`), `MAX_WORKERS` (default `10`), `LOG_LEVEL` (default `INFO`), `LOG_FLAG` (default `0`), `LOGS_DIR` (host log directory, default `./logs`), `SP_MODEL_NAME` (default `superpoint_onnx`), `SG_MODEL_NAME` (default `lightglue_onnx`).

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

Defaults: `PORT=40010`, `SP_ADDRESS=127.0.0.1:8001`, `POOL_SIZE=4`, `TOP_K=3`, `SP_MODEL_NAME=superpoint_onnx`, `SG_MODEL_NAME=lightglue_onnx`. See comment header in `scripts/run_vlserver.sh` for all environment variables.

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
  --top_k 3 \
  --sp_model_name superpoint_onnx \
  --sg_model_name lightglue_onnx
```

The server listens on **`0.0.0.0:<port>`**.

---

## Server CLI reference (`visual_pose_server.py`)

| Argument | Description |
|----------|-------------|
| `--model_path` | Map root directory (must contain `database_3d.db`, `images/`, and retrieval files used by `VisualLocalizer`) |
| `--port` | gRPC listen port for this service (default `40010`) |
| `--max_workers` | gRPC thread pool size (default `10`) |
| `--log_level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--sp_address` | Triton gRPC `host:port` for SuperPoint / SuperGlue |
| `--top_k` | Number of top retrieval images (default `3`) |
| `--log_flag` | `0` — console logging only; `1` — also write debug logs / match images under `--logs_dir` |
| `--logs_dir` | Directory used when `--log_flag` is `1` |
| `--pool_size` | Number of worker processes; each loads one `VisualLocalizer` (default `4`) |
| `--sp_model_name` | Triton model name for SuperPoint (default `superpoint_onnx`) |
| `--sg_model_name` | Triton model name for LightGlue / SuperGlue wrapper (default `lightglue_onnx`) |

---

## Call the server with the test client

The bundled client is **not** a production SDK; server addresses, image paths, and intrinsics are **hard-coded** in `visual_pose_service/visual_pose_client.py`.

1. Set **`SINGLE_CLIENT_ADDRESS`** / **`MULTI_CLIENTS_ADDRESS`** to your server `host:port`.
2. Set **`self.folder`** / **`self.image_path`** for single-client mode. The current default is `../Data/NS_hks_260414_1/test_img/1.jpg`.
3. For multi-client mode, also update the hard-coded `image_path` inside `run_multi_clients()`.
4. Adjust the sample intrinsics if needed. The client currently sends `fx=1302`, `fy=1302`, `cx=width/2`, `cy=height/2`, and four zero distortion coefficients.
5. Run:

```bash
conda activate vlpose
cd visual_pose_service
export PYTHONPATH="$(pwd)"
python visual_pose_client.py
```

The script prompts for request interval and mode:

- `1`: single client, repeatedly sends one image through `run_continuous()`.
- `2`: multi clients, starts a thread pool and repeatedly sends the multi-client test image.

Successful responses include `status=1`, `pose`, 3x3 rotation / translation covariance matrices encoded as `float64`, the original request timestamp, and `server_log`.

Projection results:

The left side of the image shows the mesh projection result and map point reprojection error, while the right side shows candidate image frames similar to the query image found by the bag-of-words model.

<table>
  <tr>
    <td align="center" width="50%"><img src="asset/proj_1.jpg" alt="projection 1" width="400"/></td>
    <td align="center" width="50%"><img src="asset/proj_2.jpg" alt="projection 2" width="400"/></td>
  </tr>
</table>

---

## Production clients

Use **gRPC** and **`GetPoseFromImage`**. Python stubs live under `visual_pose_service/proto/`. Original `.proto` files are not in this repository.

Request payload:

- `image_timestamp`: microsecond timestamp.
- `image.image.encoded_str`: encoded image bytes that OpenCV can decode.
- `intrinsics`: `fx`, `fy`, `cx`, `cy`, and optional four-value distortion coefficients.
- `localization_prior` (optional): prior pose, gravity vector, and VIO pose. The current localizer can use gravity for PnP; pose-prior retrieval is disabled by default in `visual_localizer.py`.

Response payload:

- `status=1` on success; failed responses carry `server_log` and leave `status` at the proto default.
- `pose`: translation plus quaternion rotation (`w`, `x`, `y`, `z`).
- `translation_covariance` / `rotation_covariance`: 3x3 `float64` matrices serialized in `Matrix.data`.
- `timestamp`: the request timestamp echoed back by the server.

---

## Application Example

Global visual localization service combined with local tracking of VR devices achieves global tracking.

<img src="asset/office_test.gif" alt="office test result" width="400"/>

## Related tools (not part of the localization server)

- **Marker pose** — separate gRPC service for fiducial/marker pose: [`visual_pose_service/marker_pose_service/README.md`](visual_pose_service/marker_pose_service/README.md)
