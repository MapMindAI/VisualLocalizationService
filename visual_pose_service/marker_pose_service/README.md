# Marker Pose Estimation Service

Real-time marker pose estimation over gRPC using SIFT matching and PnP.

## Features

- **Pose accuracy**: SIFT matching with refined PnP
- **Gravity-aware refinement**: Uses gravity direction when available
- **Feature matching**: FLANN with Lowe’s ratio test
- **PnP strategy**: Prefer homography corner points plus high-quality inliers
- **Optional VIO retry**: Second pass with VIO-based image warping if the first pass fails
- **Quality gates**: Thresholds on matches, inliers, reprojection error
- **Multi-process workers**: Configurable pool for throughput
- **Debug visuals**: Optional visualization

## Layout

```
marker_pose_service/
├── marker_pose_server.py      # gRPC server
├── marker_pose_client.py      # gRPC client
├── marker_pose_estimator.py   # Core estimator
├── config/
│   └── marker_config_example.json
├── marker_image/              # Reference marker images
│   ├── dm_final.jpg
│   └── marker_test.png
└── tools/
    └── transform_image_by_gravity.py  # Gravity-based image warp helper
```

## Dependencies

```bash
pip install numpy opencv-python grpcio
```

## Configuration

JSON config; see `config/marker_config_example.json`:

```json
{
  "marker_image": "/path/to/marker/image.jpg",
  "marker_width": 0.3,
  "min_matches": 15,
  "min_inliers": 8,
  "min_inlier_ratio": 0.3,
  "max_avg_reproj_error": 15.0,
  "max_reproj_error": 20.0,
  "enable_vio_retry": false
}
```

| Field | Description |
|-------|-------------|
| `marker_image` | Path to marker template image (required) |
| `marker_width` | Physical marker width in meters |
| `min_matches` | Minimum matches; below this, reject |
| `min_inliers` | Minimum inlier count |
| `min_inlier_ratio` | Minimum inlier ratio (0–1) |
| `max_avg_reproj_error` | Max mean reprojection error (pixels) |
| `max_reproj_error` | Max per-point reprojection error (pixels) |
| `enable_vio_retry` | If `true`, on failure with VIO prior, warp and retry; if `false`, fail fast |

## Usage

### 1. Start server

```bash
cd visual_pose_service
export PYTHONPATH="$(pwd)"

python marker_pose_service/marker_pose_server.py \
  --marker_config /path/to/marker_config.json \
  --port 40011 \
  --logs_dir /path/to/logs \
  --pool_size 4
```

### 2. Client

**Single image**

```bash
python marker_pose_client.py \
  --mode 1 \
  --image /path/to/image.jpg \
  --marker_width 0.3 \
  --marker_image /path/to/marker.jpg
```

**Live camera**

```bash
python marker_pose_client.py \
  --mode 2 \
  --camera_id 0 \
  --marker_width 0.3 \
  --marker_image /path/to/marker.jpg
```

**Python API**

```python
from marker_pose_client import MarkerPoseClient
import cv2

client = MarkerPoseClient(server_address='127.0.0.1:40011')
image = cv2.imread('test_image.jpg')

pose_info = client.send_image_request(
    image,
    marker_width=0.3,
    intrinsics=None  # optional
)

if pose_info:
    print(f"Translation: {pose_info['translation']}")
    print(f"Rotation: {pose_info['rotation']}")
```

## Pipeline (high level)

1. SIFT on marker template and query image  
2. FLANN matching + ratio test  
3. Homography (RANSAC) for corners  
4. Refined PnP (corners + up to 10 strong inliers; fallback if homography fails)  
5. Gravity-aware refinement  
6. Threshold checks  
7. Optional VIO retry (warp, re-match, map back)  
8. Return quaternion + translation  

## Tool: `transform_image_by_gravity.py`

Warps the marker toward a fronto-parallel view using gravity and pose; uses `H = K * R * K^{-1}`.

Edit paths and pose in the script, then run:

```bash
python transform_image_by_gravity.py
```

## API: `GetPoseFromImage`

gRPC: image in, marker pose out.

**Request**

- `image`: JPEG bytes  
- `image_timestamp`: microseconds  
- `intrinsics`: optional camera intrinsics  

**Response**

- `status`: `0` = success  
- `rotation`: quaternion `[w, x, y, z]`  
- `translation`: `[x, y, z]`  
- `server_log`: server-side debug string  

## Performance

- Multi-process pool (`--pool_size`, default 4)  
- Queued tasks and result thread  
- Cached marker features at startup  
- Corner-first PnP to limit point count  

## Troubleshooting

1. **Too few matches** — improve marker texture; tune `min_matches`; ensure full marker in view  
2. **High reprojection error** — verify intrinsics and `marker_width`; relax error thresholds if appropriate  
3. **No response** — confirm server running, logs, and free port  

## Notes

- Marker images should be textured for SIFT  
- `marker_width` must be accurate for 3D scale  
- Intrinsics recommended  
- Gravity parameters must match your world frame  
- VIO retry needs `image_prior`; adds latency; default off  

## Environment (Conda)

Use the same Conda env as the main `visual_pose_service` (see repository root `README.md`), or a minimal env with the dependencies above. From the repo root:

```bash
conda create -n vlpose python=3.9 -y
conda activate vlpose
cd visual_pose_service
pip install -r requirements.txt
```

Run the marker server from `visual_pose_service` so `proto` and shared modules resolve (or set `PYTHONPATH` to `visual_pose_service`):

```bash
conda activate vlpose
cd /path/to/VisualLocalizationService/visual_pose_service
export PYTHONPATH="$(pwd)"
python marker_pose_service/marker_pose_server.py --port 40011 --pool_size 4
```

Point `--marker_config` / JSON `marker_image` to **real paths on your machine** (not container paths).
