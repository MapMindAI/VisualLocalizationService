# Visual Pose Service

Demo and deployment assets for the visual pose gRPC service.

## Quick start

### Dependencies

```bash
pip install -r requirements.txt
```

### Start server

```bash
python visual_pose_server.py
```

### Run test client

```bash
python visual_pose_client.py
```

## Preparing map data

### 1. Build map and generate database

Record the environment (e.g. with GoPro) and run the mapping pipeline to produce a localization database.

Pipeline reference: [`https://github.com/deepmirrorinc/GaussianSplatting`](https://github.com/deepmirrorinc/GaussianSplatting).

### 2. Prepare retrieval (BoW)

```bash
python retrieval/make_retrieval_db.py --model_path /path/to/your/map
```

Retrieval preparation may already be integrated in the Gaussian Splatting pipeline.

### 3. Prepare `database_3d`

Fill poses, 3D points (from depth), and keypoints in the database file as required by your pipeline.

```bash
# Parameters are documented in the script
python /path/to/update_map_database.py
```

## Configuration

### Server (`visual_pose_server.py`)

| Argument | Description |
|----------|-------------|
| `--model_path` | COLMAP / map root directory |
| `--port` | gRPC port (default `40010`) |
| `--max_workers` | gRPC thread pool size (default `10`) |
| `--log_level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--sp_address` | Triton address for SuperPoint/SuperGlue |
| `--top_k` | Number of retrieval candidates |
| `--log_flag` | `0` console only; `1` also write logs under `--logs_dir` |
| `--logs_dir` | Log directory when `--log_flag` is `1` |

### Test client (`visual_pose_client.py`)

Parameters are mostly hard-coded in the file for local simulation.

When you run `python visual_pose_client.py`, choose `1` for single-client loop or `2` for multi-client stress test.

- **Single client**: `server_address`, `image_path`, `interval`, `max_num` (in code).
- **Multi client**: number of clients, then `server_address`, `image_path`, `interval`, `run_time` (in code).

## Typical workflow

```text
# After map generation: ensure database and images exist under the map directory

cd visual_pose_service   # adjust paths as needed

python visual_pose_server.py   # with args above

python visual_pose_client.py   # in another terminal
```

## Deploying on Linux / Windows

- Allow inbound connections through the firewall for the gRPC port.
- Docker images can speed up deployment (images may be built in CI and stored in a registry).

Example build (paths adjusted to your environment):

```bash
chmod +x build_server.sh
./path/to/build_server.sh
```

Before running, ensure:

- Docker is installed.
- Map data exists locally (`DATA_DIR` in your run script).
- `run_server_linux.sh` or `run_server_windows.ps1` paths are updated:
  - `DATA_DIR`: map directory to mount into the container
  - `DB_NAME`: database file path relative to `DATA_DIR`
  - `IMAGE_PATH`: container image reference
  - `MODEL_DIR`: Triton model repository (if applicable)
  - Optional: override `deployment/entrypoint.sh` via a bind mount for debugging
- `install_docker_*.sh` / `install_docker_*.ps1` alongside the run scripts if you use them.
- Authenticate to your container registry (`docker login ...`) before pulling private images.

Then run your `run_server_*.sh` or `run_server_*.ps1`.

Deployed layout example:

![folder.jpg](./images/folder.jpg)
