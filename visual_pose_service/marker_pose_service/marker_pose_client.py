#!/usr/bin/env python3
import grpc
import time
import signal
import os
import cv2
import numpy as np
import sys

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from proto.map import visual_pose_service_pb2, visual_pose_service_pb2_grpc
from proto.internal.data import image_data_pb2
from proto.api.vision import image_pb2
from proto.api.vision import sensor_pb2

TIMEOUT = 30.0  # seconds, gRPC timeout for each request

# Default server address
DEFAULT_SERVER_ADDRESS = '127.0.0.1:40011'


class MarkerPoseClient:
    # Default camera intrinsics (can be customized)
    DEFAULT_FX = 745.6702238297661
    DEFAULT_FY = 744.6129004520485
    DEFAULT_CX = 349.9650377498923
    DEFAULT_CY = 241.6554278950715
    DEFAULT_DIST_COEF = [0.005384192656204398, -0.8058426332593978, -0.01725660206090937, 0.001368436315894204]

    def __init__(self, server_address=DEFAULT_SERVER_ADDRESS):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = visual_pose_service_pb2_grpc.VisualPoseServiceStub(self.channel)
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def create_request(self, image, timestamp=None, intrinsics=None):
        """Create gRPC request from image."""
        if timestamp is None:
            timestamp = int(time.time() * 1000000)

        success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("Failed to encode image to JPEG")

        image_msg = image_pb2.Image(encoded_str=buffer.tobytes())
        image_data_msg = image_data_pb2.ImageData(image=image_msg)

        if intrinsics is None:
            height, width = image.shape[:2]
            fx = fy = max(width, height) * 0.8
            intrinsics = sensor_pb2.CameraIntrinsics(fx=fx, fy=fy, cx=width/2.0, cy=height/2.0)
        intrinsics.distortion_coef.extend([0.0, 0.0, 0.0, 0.0])

        return visual_pose_service_pb2.ImageRequest(image=image_data_msg, image_timestamp=timestamp, intrinsics=intrinsics)

    def decode_covariance(self, covariance):
        """Decode covariance matrix from protobuf."""
        rows = covariance.row
        cols = covariance.column
        data_flat = np.frombuffer(covariance.data, dtype=np.float64)
        cov = data_flat.reshape(rows, cols)
        return cov

    def parse_pose_response(self, response):
        """Parse pose response and return pose information."""
        if response.status <= 0:
            return None

        pose = response.pose
        rot = pose.rotation
        trans = pose.translation

        # Convert to numpy arrays
        rotation_quat = np.array([rot.w, rot.x, rot.y, rot.z])  # [w, x, y, z]
        translation = np.array([trans.x, trans.y, trans.z])

        # Decode covariance
        rot_cov = self.decode_covariance(response.rotation_covariance)
        tran_cov = self.decode_covariance(response.translation_covariance)

        return {
            'rotation': rotation_quat,  # [w, x, y, z]
            'translation': translation,  # [x, y, z]
            'rotation_cov': rot_cov,
            'translation_cov': tran_cov,
            'timestamp': response.timestamp,
            'status': response.status,
            'server_log': response.server_log,
        }

    def draw_marker_boundary_and_axes(
        self, image, pose_info, marker_width, camera_matrix, dist_coeffs=None, marker_height=None, marker_image_path=None
    ):
        """
        Draw marker boundary (4 corners) and coordinate axes on image.
        
        Args:
            image: OpenCV image to draw on
            pose_info: Pose information dict from parse_pose_response
            marker_width: Real width of marker in meters
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients (optional)
            marker_height: Real height of marker in meters (optional, calculated from marker_image_path if not provided)
            marker_image_path: Path to marker image file (optional, used to calculate aspect ratio)
        
        Returns:
            Image with drawn boundary and axes
        """
        if pose_info is None:
            return image

        if dist_coeffs is None:
            dist_coeffs = np.zeros(4)

        if marker_height is None:
            if marker_image_path and os.path.exists(marker_image_path):
                marker_image = cv2.imread(marker_image_path)
                if marker_image is not None:
                    h_px, w_px = marker_image.shape[:2]
                    marker_height = marker_width * (h_px / w_px)
                else:
                    marker_height = marker_width
            else:
                marker_height = marker_width

        hw, hh = marker_width / 2.0, marker_height / 2.0
        marker_corners_3d = np.array([
            [-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]
        ], dtype=np.float32)

        axis_len = marker_width * 0.3
        marker_axes_3d = np.array([
            [0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]
        ], dtype=np.float32)

        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from utils.math_utils import quaternion_to_rotation_matrix

        R = quaternion_to_rotation_matrix(pose_info['rotation'])
        rvec, _ = cv2.Rodrigues(R)

        corners_2d, _ = cv2.projectPoints(marker_corners_3d, rvec, pose_info['translation'], camera_matrix, dist_coeffs)
        corners_2d = corners_2d.reshape(-1, 2).astype(np.int32)

        axes_2d, _ = cv2.projectPoints(marker_axes_3d, rvec, pose_info['translation'], camera_matrix, dist_coeffs)
        axes_2d = axes_2d.reshape(-1, 2).astype(np.int32)

        # Draw boundary and corners
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i + 1) % 4]), (0, 255, 255), 3)
            cv2.circle(image, tuple(corners_2d[i]), 5, (0, 255, 0), -1)

        # Draw axes: X(red), Y(green), Z(blue)
        origin = tuple(axes_2d[0])
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ['X', 'Y', 'Z']
        for i in range(1, 4):
            end = tuple(axes_2d[i])
            cv2.line(image, origin, end, colors[i-1], 3)
            cv2.putText(image, labels[i-1], (end[0] + 5, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i-1], 2)
        cv2.circle(image, origin, 5, (255, 255, 255), -1)

        return image

    def send_image_request(self, image, marker_width=0.1, intrinsics=None):
        """Send image request to server and get pose."""
        try:
            start_time = time.time()
            response = self.stub.GetPoseFromImage(self.create_request(image, intrinsics=intrinsics), timeout=TIMEOUT)
            pose_info = self.parse_pose_response(response)

            if pose_info:
                print(f"Response status: {response.status}")
                print(f"Translation: {pose_info['translation']}")
                print(f"Rotation (w,x,y,z): {pose_info['rotation']}")
                print(f"Response time: {time.time() - start_time:.4f}s")
                print(f"Server log: {response.server_log}")
                print("-" * 50)

            return pose_info
        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def _get_default_intrinsics(self):
        """Get default camera intrinsics."""
        intrinsics = sensor_pb2.CameraIntrinsics(
            fx=self.DEFAULT_FX, fy=self.DEFAULT_FY, cx=self.DEFAULT_CX, cy=self.DEFAULT_CY
        )
        intrinsics.distortion_coef.extend(self.DEFAULT_DIST_COEF)
        return intrinsics

    def _get_camera_matrix(self):
        """Get default camera matrix."""
        return np.array([
            [self.DEFAULT_FX, 0, self.DEFAULT_CX],
            [0, self.DEFAULT_FY, self.DEFAULT_CY],
            [0, 0, 1]
        ], dtype=np.float32)

    def mode1_single_image(self, image_path, marker_width=0.1, marker_image_path=None):
        """Mode 1: Process a single image file."""
        print(f"Mode 1: Processing single image: {image_path}")
        print("=" * 50)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image from {image_path}")
            return

        print(f"Image size: {image.shape[1]}x{image.shape[0]}")

        intrinsics = self._get_default_intrinsics()
        pose_info = self.send_image_request(image, marker_width, intrinsics)

        if pose_info:
            result_image = self.draw_marker_boundary_and_axes(
                image.copy(), pose_info, marker_width, self._get_camera_matrix(), marker_image_path=marker_image_path
            )
            cv2.namedWindow("Marker Detection Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Marker Detection Result", result_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to detect marker in image")

    def mode2_realtime_camera(self, marker_width=0.1, camera_id=0, marker_image_path=None):
        """Mode 2: Real-time camera feed."""
        print(f"Mode 2: Real-time camera feed (camera ID: {camera_id})")
        print("Press 'q' to quit")
        print("=" * 50)

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Failed to open camera {camera_id}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}, FPS: {cap.get(cv2.CAP_PROP_FPS)}")

        intrinsics = self._get_default_intrinsics()
        camera_matrix = self._get_camera_matrix()
        frame_count = 0
        last_request_time = 0
        request_interval = 0.1
        last_pose_info = None

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break

                frame_count += 1
                current_time = time.time()

                if current_time - last_request_time >= request_interval:
                    pose_info = self.send_image_request(frame, marker_width, intrinsics)
                    last_request_time = current_time
                    last_pose_info = pose_info if pose_info else None

                if last_pose_info:
                    display_frame = self.draw_marker_boundary_and_axes(
                        frame.copy(), last_pose_info, marker_width, camera_matrix, marker_image_path=marker_image_path
                    )
                else:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "No marker detected", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(display_frame, f"Frame: {frame_count}", (20, display_frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Marker Detection - Real-time", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit requested by user")
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released")

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping...")
        self.running = False

    def close(self):
        if hasattr(self, 'channel'):
            self.channel.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Marker Pose Client')
    parser.add_argument(
        '--server',
        type=str,
        default='127.0.0.1:40010',
        help=f'Server address (default: 127.0.0.1:40011)',
    )
    parser.add_argument(
        '--marker_width',
        type=float,
        default=0.2,
        help='Real width of marker in meters (default: 0.1)',
    )
    parser.add_argument(
        '--image',
        type=str,
        default='/Mobili/Data/marker2_a.jpg',
        help='Image file path (for mode 1)',
    )
    parser.add_argument(
        '--camera_id',
        type=int,
        default=2,
        help='Camera device ID (for mode 2, default: 0)',
    )
    parser.add_argument(
        '--marker_image',
        type=str,
        default=None,
        help='Path to marker image file (used to calculate aspect ratio for accurate projection)',
    )

    args = parser.parse_args()

    client = MarkerPoseClient(args.server)

    print("=" * 50)
    print("Marker Pose Client")
    print(f"Server address: {args.server}")
    print("=" * 50)

    # Interactive mode selection
    print("\nSelect mode:")
    print("1. Single image file")
    print("2. Real-time camera feed")
    mode = input("Enter mode (1 or 2): ").strip()

    if mode == '1':
        # Use command line argument if provided, otherwise prompt for input
        if args.image:
            image_path = args.image
            print(f"Using image path from argument: {image_path}")
        else:
            image_path = input("Enter image file path: ").strip()
        
        if image_path:
            client.mode1_single_image(image_path, args.marker_width, args.marker_image)
        else:
            print("Error: No image path provided")
    elif mode == '2':
        client.mode2_realtime_camera(args.marker_width, args.camera_id, args.marker_image)
    else:
        print(f"Error: Invalid mode '{mode}'. Please enter 1 or 2.")

    client.close()


if __name__ == '__main__':
    main()

