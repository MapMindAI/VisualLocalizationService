import numpy as np
import cv2
import os
import time
import logging
import math
import sys

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from optimization.pnp_optimization import pnp_ransac_optimization_with_gravity
from utils.math_utils import rotation_matrix_to_quaternion

logger = logging.getLogger(__name__)

GRAVITY_WORLD = np.array([0, 0, -1])
GRAVITY_WEIGHT = 1000.0
GRAVITY_ERROR_THRESHOLD_DEGREE = 40.0

# Default thresholds for quality control
DEFAULT_MIN_MATCHES = 20  # Minimum number of matches required
DEFAULT_MIN_INLIERS = 10  # Minimum number of inliers required
DEFAULT_MIN_INLIER_RATIO = 0.3  # Minimum inlier ratio (0.0-1.0)
DEFAULT_MAX_AVG_REPROJ_ERROR = 5.0  # Maximum average reprojection error in pixels
DEFAULT_MAX_REPROJ_ERROR = 10.0  # Maximum reprojection error for individual points in pixels


class MarkerPoseEstimator:
    def __init__(self, marker_config: dict = None):
        self.marker_config = marker_config or {}
        self._initialized = False
        self.current_task_id = None
        self.marker_image_path = self.marker_config.get('marker_image', None)
        self.marker_width = self.marker_config.get('marker_width', 0.1)
        self.sift = None
        self.flann_matcher = None
        self.marker_image = None
        self.marker_keypoints = None
        self.marker_descriptors = None
        self.marker_scores = None
        self.marker_scale_x = None
        self.marker_scale_y = None
        self.marker_height = None  # Real height in meters (calculated from image aspect ratio)
        
        # Quality control thresholds
        self.min_matches = self.marker_config.get('min_matches', DEFAULT_MIN_MATCHES)
        self.min_inliers = self.marker_config.get('min_inliers', DEFAULT_MIN_INLIERS)
        self.min_inlier_ratio = self.marker_config.get('min_inlier_ratio', DEFAULT_MIN_INLIER_RATIO)
        self.max_avg_reproj_error = self.marker_config.get('max_avg_reproj_error', DEFAULT_MAX_AVG_REPROJ_ERROR)
        self.max_reproj_error = self.marker_config.get('max_reproj_error', DEFAULT_MAX_REPROJ_ERROR)
        
        # VIO retry flag: whether to retry with VIO pose transformation after first attempt fails
        self.enable_vio_retry = self.marker_config.get('enable_vio_retry', True)

    def initialize(self, marker_config: dict = None):
        if marker_config:
            self.marker_config = marker_config
            self.marker_image_path = self.marker_config.get('marker_image', self.marker_image_path)
            self.marker_width = self.marker_config.get('marker_width', self.marker_width)
            # Update quality control thresholds if provided
            self.min_matches = self.marker_config.get('min_matches', self.min_matches)
            self.min_inliers = self.marker_config.get('min_inliers', self.min_inliers)
            self.min_inlier_ratio = self.marker_config.get('min_inlier_ratio', self.min_inlier_ratio)
            self.max_avg_reproj_error = self.marker_config.get('max_avg_reproj_error', self.max_avg_reproj_error)
            self.max_reproj_error = self.marker_config.get('max_reproj_error', self.max_reproj_error)
            self.enable_vio_retry = self.marker_config.get('enable_vio_retry', self.enable_vio_retry)

        # Validate configuration
        if not self.marker_image_path:
            error_msg = "marker_image not specified in config"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not os.path.exists(self.marker_image_path):
            error_msg = f"Marker image file not found: {self.marker_image_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Marker image: {self.marker_image_path}")
        logger.info(f"Marker width: {self.marker_width} meters")
        logger.info(f"Enable VIO retry: {self.enable_vio_retry}")
        
        try:
            self.sift = cv2.SIFT_create()
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
            logger.info("SIFT and FLANN matcher initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize SIFT: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Load marker image
        self.marker_image = cv2.imread(self.marker_image_path)
        if self.marker_image is None:
            error_msg = f"Failed to load marker image: {self.marker_image_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        marker_height_px, marker_width_px = self.marker_image.shape[:2]
        logger.info(f"Marker image size: {marker_width_px}x{marker_height_px} pixels")
        
        # Calculate real height based on image aspect ratio
        aspect_ratio = marker_height_px / marker_width_px
        self.marker_height = self.marker_width * aspect_ratio
        logger.info(f"Marker real size: {self.marker_width:.4f}m (width) x {self.marker_height:.4f}m (height)")
        
        self.marker_scale_x = self.marker_width / marker_width_px
        self.marker_scale_y = self.marker_height / marker_height_px
        
        logger.info("Extracting SIFT features from marker image...")
        start_time = time.time()
        self.marker_keypoints, self.marker_descriptors, self.marker_scores = self._extract_features_sift(self.marker_image)
        
        if self.marker_keypoints is None or self.marker_descriptors is None:
            error_msg = "Failed to extract SIFT features from marker image"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        extract_time = time.time() - start_time
        logger.info(
            f"Extracted {self.marker_keypoints.shape[1]} SIFT features in {extract_time:.5f} seconds"
        )

        self._initialized = True
        logger.info("Marker pose estimator initialization completed!")

    def set_current_task_id(self, task_id):
        self.current_task_id = task_id
    
    def _extract_features_sift(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if keypoints is None or descriptors is None or len(keypoints) == 0:
            return None, None, None
        
        kpts_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        kpts_array = np.expand_dims(kpts_array, axis=0)
        descriptors = descriptors.astype(np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        return kpts_array, descriptors, scores
    
    def _match_features_sift(self, query_keypoints, query_descriptors, marker_keypoints, marker_descriptors):
        if query_descriptors is None or marker_descriptors is None:
            return None, None
        
        query_desc_float = query_descriptors.astype(np.float32)
        marker_desc_float = marker_descriptors.astype(np.float32)
        if len(query_desc_float) == 0 or len(marker_desc_float) == 0:
            num_query = len(query_keypoints[0])
            return np.full(num_query, -1, dtype=np.int32), np.zeros(num_query, dtype=np.float32)
        
        try:
            matches = self.flann_matcher.knnMatch(query_desc_float, marker_desc_float, k=2)
        except cv2.error as e:
            logger.warning(f"Task {self.current_task_id} : FLANN match error: {e}")
            num_query = len(query_keypoints[0])
            return np.full(num_query, -1, dtype=np.int32), np.zeros(num_query, dtype=np.float32)
        
        good_matches = [m for match_pair in matches if len(match_pair) == 2 
                       for m, n in [match_pair] if m.distance < 0.7 * n.distance]
        
        num_query = len(query_keypoints[0])
        match_indices = np.full(num_query, -1, dtype=np.int32)
        match_scores = np.zeros(num_query, dtype=np.float32)
        
        for match in good_matches:
            if match.queryIdx < num_query and match.trainIdx < len(marker_keypoints[0]):
                match_indices[match.queryIdx] = match.trainIdx
                match_scores[match.queryIdx] = 1.0 / (match.distance + 1e-6)
        
        return match_indices, match_scores

    def estimate_pose_from_byte(self, image_bytes: bytes, image_prior, intrinsic, save_dir, draw_flag=0):
        if image_bytes is None or not self._initialized:
            error_msg = f"Task {self.current_task_id} : Invalid input or not initialized"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        np_img = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            error_msg = f"Task {self.current_task_id} : Failed to decode image"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        # Try normal estimation first
        result = self._estimate_pose_from_image(image, image_prior, intrinsic, save_dir, draw_flag)
        ok = result[0]  # Only check the first element (success flag)
        if ok:
            return result
        
        # If failed, try with VIO pose transformation if available and enabled
        if not self.enable_vio_retry:
            logger.info(f"Task {self.current_task_id} : VIO retry is disabled, skipping VIO pose transformation")
        elif image_prior is not None and len(image_prior) >= 3 and image_prior[2] is not None:
            vio_pose = image_prior[2]  # [w, x, y, z, tx, ty, tz]
            logger.info(f"Task {self.current_task_id} : First attempt failed, trying with VIO pose transformation...")
            
            # Start timing for VIO pose transformation
            vio_transform_start_time = time.time()
            
            # Transform image using VIO pose
            camera_matrix = intrinsic[0]
            image_transform_start_time = time.time()
            image_transformed, H_final = self._transform_image_by_gravity(image, vio_pose, camera_matrix)
            image_transform_time = time.time() - image_transform_start_time
            logger.info(f"Task {self.current_task_id} : Image transformation took {image_transform_time:.5f} seconds")
            
            if image_transformed is not None and H_final is not None:
                logger.info(f"Task {self.current_task_id} : Image transformed successfully, retrying pose estimation...")
                # Save transformed image for debugging
                if save_dir:
                    try:
                        transformed_image_path = save_dir + '_transformed.jpg'
                        cv2.imwrite(transformed_image_path, image_transformed)
                        logger.info(f"Task {self.current_task_id} : Saved transformed image to {transformed_image_path}")
                        logger.info(f"Task {self.current_task_id} : Transformed image shape: {image_transformed.shape}, dtype: {image_transformed.dtype}")
                        logger.info(f"Task {self.current_task_id} : Original image shape: {image.shape}, dtype: {image.dtype}")
                        # Log homography matrix for debugging
                        logger.debug(f"Task {self.current_task_id} : H_final matrix:\n{H_final}")
                    except Exception as e:
                        logger.warning(f"Task {self.current_task_id} : Failed to save transformed image: {e}")
                
                # Retry with transformed image
                original_image_shape = (image.shape[0], image.shape[1])
                # Use different save_dir suffix for VIO transformation
                save_dir_vio = save_dir + '_vio' if save_dir else save_dir
                pose_estimation_start_time = time.time()
                result = self._estimate_pose_from_image(image_transformed, image_prior, intrinsic, save_dir_vio, draw_flag, 
                                                       use_transformed_image=True, H_final=H_final, 
                                                       original_image_shape=original_image_shape,
                                                       original_image=image)
                pose_estimation_time = time.time() - pose_estimation_start_time
                ok = result[0]  # Only check the first element (success flag)
                if ok:
                    total_vio_time = time.time() - vio_transform_start_time
                    logger.info(f"Task {self.current_task_id} : Pose estimation succeeded after VIO pose transformation!")
                    logger.info(f"Task {self.current_task_id} : VIO pose transformation total time: {total_vio_time:.5f} seconds (image transform: {image_transform_time:.5f}s, pose estimation: {pose_estimation_time:.5f}s)")
                    return result
                else:
                    total_vio_time = time.time() - vio_transform_start_time
                    logger.warning(f"Task {self.current_task_id} : Pose estimation still failed after VIO pose transformation")
                    logger.info(f"Task {self.current_task_id} : VIO pose transformation total time: {total_vio_time:.5f} seconds (image transform: {image_transform_time:.5f}s, pose estimation: {pose_estimation_time:.5f}s)")
            else:
                total_vio_time = time.time() - vio_transform_start_time
                logger.warning(f"Task {self.current_task_id} : Failed to transform image using VIO pose")
                logger.info(f"Task {self.current_task_id} : VIO pose transformation total time: {total_vio_time:.5f} seconds (image transform: {image_transform_time:.5f}s)")
        
        # Return the original failure result
        return result 

    def estimate_pose_from_image(self, image: np.ndarray, intrinsic, save_dir, draw_flag=0):
        if image is None or not self._initialized:
            error_msg = f"Task {self.current_task_id} : Invalid input or not initialized"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg
        return self._estimate_pose_from_image(image, None, intrinsic, save_dir, draw_flag)

    def _pixel_to_marker_coords(self, pixel_x: float, pixel_y: float) -> np.ndarray:
        """
        Convert pixel coordinates in marker image to marker 3D coordinates.
        
        Marker coordinate system:
        - Origin (0,0,0) is at the center of the marker
        - X-axis: right (positive x direction in image)
        - Y-axis: down (positive y direction in image)
        - Z-axis: up (perpendicular to marker plane, pointing outward)
        
        Args:
            pixel_x: X coordinate in marker image (0 to marker_width_px)
            pixel_y: Y coordinate in marker image (0 to marker_height_px)
        
        Returns:
            3D coordinates in marker frame [x, y, 0.0] in meters
        """
        marker_height_px, marker_width_px = self.marker_image.shape[:2]
        center_x = marker_width_px / 2.0
        center_y = marker_height_px / 2.0
        # Convert to marker coordinates: center at origin, x right, y down
        marker_x = (pixel_x - center_x) * self.marker_scale_x
        marker_y = (pixel_y - center_y) * self.marker_scale_y
        return np.array([marker_x, marker_y, 0.0], dtype=np.float32)

    def _should_draw(self, draw_flag, save_dir):
        return draw_flag == 1 and save_dir != ''
    
    def _transform_points_from_transformed_to_original(self, points_2d, H_final):
        """
        Transform 2D points from transformed image coordinates to original image coordinates.
        
        Args:
            points_2d: 2D points in transformed image coordinates, shape (N, 2)
            H_final: Homography matrix from original to transformed image
        
        Returns:
            2D points in original image coordinates, shape (N, 2)
        """
        points_homogeneous = np.hstack([
            points_2d,
            np.ones((len(points_2d), 1), dtype=np.float32)
        ])
        H_final_inv = np.linalg.inv(H_final)
        points_original_homogeneous = (H_final_inv @ points_homogeneous.T).T
        points_original = points_original_homogeneous[:, :2] / points_original_homogeneous[:, 2:3]
        return points_original
    
    def _transform_query_keypoints_to_original(self, query_keypoints, H_final):
        """
        Transform query_keypoints from transformed image coordinates to original image coordinates.
        
        Args:
            query_keypoints: Keypoints array with shape (1, N, 2) in transformed image coordinates
            H_final: Homography matrix from original to transformed image
        
        Returns:
            Keypoints array with shape (1, N, 2) in original image coordinates
        """
        query_kpts_transformed = query_keypoints[0].copy()  # Shape: (N, 2)
        query_kpts_original = self._transform_points_from_transformed_to_original(query_kpts_transformed, H_final)
        return np.expand_dims(query_kpts_original, axis=0)
    
    def _transform_points_from_original_to_transformed(self, points_2d, H_final):
        """
        Transform 2D points from original image coordinates to transformed image coordinates.
        
        Args:
            points_2d: 2D points in original image coordinates, shape (N, 2)
            H_final: Homography matrix from original to transformed image
        
        Returns:
            2D points in transformed image coordinates, shape (N, 2)
        """
        points_homogeneous = np.hstack([
            points_2d,
            np.ones((len(points_2d), 1), dtype=np.float32)
        ])
        points_transformed_homogeneous = (H_final @ points_homogeneous.T).T
        points_transformed = points_transformed_homogeneous[:, :2] / points_transformed_homogeneous[:, 2:3]
        return points_transformed
    
    def _get_marker_corners_and_axes_3d(self):
        """
        Get marker corners and axes in 3D marker coordinate system.
        
        Returns:
            tuple: (marker_corners_3d, marker_axes_3d) both as numpy arrays
        """
        half_width = self.marker_width / 2.0
        half_height = self.marker_height / 2.0
        marker_corners_3d = np.array([
            [-half_width, -half_height, 0.0],  # top-left: (-w/2, -h/2, 0)
            [half_width, -half_height, 0.0],   # top-right: (+w/2, -h/2, 0)
            [half_width, half_height, 0.0],    # bottom-right: (+w/2, +h/2, 0)
            [-half_width, half_height, 0.0],   # bottom-left: (-w/2, +h/2, 0)
        ], dtype=np.float32)
        
        axis_length = self.marker_width * 0.3
        marker_axes_3d = np.array([
            [0, 0, 0],  # origin
            [axis_length, 0, 0],  # x-axis (right)
            [0, axis_length, 0],  # y-axis (down)
            [0, 0, axis_length],  # z-axis (up)
        ], dtype=np.float32)
        
        return marker_corners_3d, marker_axes_3d
    
    def _draw_marker_boundary_and_axes_on_image(self, image, corners_2d, axes_2d):
        """
        Draw marker boundary and coordinate axes on image.
        
        Args:
            image: Image to draw on
            corners_2d: Marker corner points in image coordinates, shape (4, 2), dtype int32
            axes_2d: Marker axes points in image coordinates, shape (4, 2), dtype int32
        """
        # Draw marker boundary (yellow)
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[(i + 1) % 4])
            cv2.line(image, pt1, pt2, (0, 255, 255), 3)

        # Draw corner points (green)
        for pt in corners_2d:
            cv2.circle(image, tuple(pt), 5, (0, 255, 0), -1)

        # Draw coordinate axes
        origin = tuple(axes_2d[0])
        x_end = tuple(axes_2d[1])
        y_end = tuple(axes_2d[2])
        z_end = tuple(axes_2d[3])

        cv2.line(image, origin, x_end, (0, 0, 255), 3)  # X-axis (red)
        cv2.putText(image, 'X', (x_end[0] + 5, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(image, origin, y_end, (0, 255, 0), 3)  # Y-axis (green)
        cv2.putText(image, 'Y', (y_end[0] + 5, y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(image, origin, z_end, (255, 0, 0), 3)  # Z-axis (blue)
        cv2.putText(image, 'Z', (z_end[0] + 5, z_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.circle(image, origin, 5, (255, 255, 255), -1)  # Origin (white)
    
    def _transform_image_by_gravity(self, image, vio_pose, camera_matrix, gravity_local=np.array([0, 1, 0])):
        """
        Transform image using VIO pose and gravity direction to make marker frontal.
        
        Args:
            image: Input image (BGR)
            vio_pose: VIO pose as [w, x, y, z, tx, ty, tz] (camera_to_local in SLAM coordinate system)
                     Note: The pose has already been converted from Unreal to SLAM in C++ code
            camera_matrix: 3x3 camera intrinsic matrix
            gravity_local: Gravity direction in local frame (default: [0, 1, 0])
        
        Returns:
            tuple: (transformed_image, H_final) where H_final is the homography matrix
                   Returns (None, None) if transformation fails
        """
        try:
            from scipy.spatial.transform import Rotation as R
            
            # Extract quaternion and translation from vio_pose
            w, x, y, z = vio_pose[0], vio_pose[1], vio_pose[2], vio_pose[3]
            quat_xyzw = np.array([x, y, z, w])
            rot_slam = R.from_quat(quat_xyzw)
            R_slam = rot_slam.as_matrix()
            
            # Transform marker normal from local to camera frame
            R_cam_to_local = R_slam
            R_local_to_cam = R_cam_to_local.T
            marker_normal_camera = R_local_to_cam @ gravity_local
            marker_normal_camera = marker_normal_camera / np.linalg.norm(marker_normal_camera)
            
            # Compute rotation to align marker normal to camera z-axis
            marker_normal_target = np.array([0, 0, 1])
            rotation_axis = np.cross(marker_normal_camera, marker_normal_target)
            axis_norm = np.linalg.norm(rotation_axis)
            
            if axis_norm < 1e-6:
                if np.dot(marker_normal_camera, marker_normal_target) > 0.99:
                    R_rotation = np.eye(3)
                    angle = 0.0
                else:
                    if abs(marker_normal_camera[0]) < 0.9:
                        rotation_axis = np.cross(marker_normal_camera, np.array([1, 0, 0]))
                    else:
                        rotation_axis = np.cross(marker_normal_camera, np.array([0, 1, 0]))
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.pi
            else:
                rotation_axis = rotation_axis / axis_norm
                cos_angle = np.clip(np.dot(marker_normal_camera, marker_normal_target), -1.0, 1.0)
                angle = np.arccos(cos_angle)
            
            # Create rotation matrix using Rodrigues' formula
            if angle < 1e-6:
                R_rotation = np.eye(3)
            else:
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                R_rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            
            # Compute homography H = K * R * K^(-1)
            h, w = image.shape[:2]
            K_inv = np.linalg.inv(camera_matrix)
            H = camera_matrix @ R_rotation @ K_inv
            H = H / H[2, 2]  # Normalize
            
            # Transform image corners to determine output size
            corners_original = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            corners_transformed = cv2.perspectiveTransform(corners_original, H).reshape(-1, 2)
            x_min, x_max = int(np.floor(corners_transformed[:, 0].min())), int(np.ceil(corners_transformed[:, 0].max()))
            y_min, y_max = int(np.floor(corners_transformed[:, 1].min())), int(np.ceil(corners_transformed[:, 1].max()))
            
            # Add margin and adjust homography for translation
            margin = 50
            x_min, y_min = min(x_min, -margin), min(y_min, -margin)
            x_max, y_max = max(x_max, w + margin), max(y_max, h + margin)
            
            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
            H_final = translation @ H
            output_width, output_height = x_max - x_min, y_max - y_min
            
            # Apply transformation
            image_transformed = cv2.warpPerspective(image, H_final, (output_width, output_height), 
                                                    flags=cv2.INTER_LINEAR, 
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=(0, 0, 0))
            
            return image_transformed, H_final
            
        except Exception as e:
            logger.warning(f"Task {self.current_task_id} : Failed to transform image by gravity: {e}")
            return None, None
    
    def _save_failure_visualization(self, image, query_keypoints, matches, save_dir, error_msg, 
                                     use_pose=False, matched_2d_inliers=None, matched_3d_inliers=None,
                                     quat=None, position=None, camera_matrix=None, dist_coeffs=None,
                                     use_transformed_image=False, H_final=None, original_image=None):
        if use_pose and matched_2d_inliers is not None and matched_3d_inliers is not None:
            try:
                self._draw_matches(image, matched_2d_inliers, matched_3d_inliers,
                                 quat, position, camera_matrix, dist_coeffs, save_dir, None,
                                 use_transformed_image=use_transformed_image, H_final=H_final,
                                 original_image=original_image)
            except Exception as e:
                logger.warning(f"Task {self.current_task_id} : Failed to draw with pose: {e}")
        # For simple visualization, use transformed image to show matches on transformed image
        # query_keypoints are extracted from 'image' (which is transformed if use_transformed_image=True)
        self._draw_matches_simple(image, query_keypoints, self.marker_image, self.marker_keypoints,
                                 matches, save_dir, error_msg)
        
        # If using transformed image, also draw matches on original image
        # Transform query_keypoints from transformed image to original image coordinates
        if use_transformed_image and H_final is not None and original_image is not None:
            try:
                query_keypoints_original = self._transform_query_keypoints_to_original(query_keypoints, H_final)
                self._draw_matches_simple(original_image, query_keypoints_original, self.marker_image, 
                                        self.marker_keypoints, matches, save_dir + '_original', error_msg)
            except Exception as e:
                logger.warning(f"Task {self.current_task_id} : Failed to draw matches on original image: {e}", exc_info=True)

    def _estimate_pose_from_image(self, image: np.ndarray, image_prior, intrinsic, save_dir, draw_flag, 
                                   use_transformed_image=False, H_final=None, original_image_shape=None,
                                   original_image=None):
        """
        Estimate pose from image. If use_transformed_image is True, the image is already transformed
        and H_final is the homography matrix to transform points back to original image.
        
        Args:
            image: Input image (may be original or transformed)
            image_prior: Prior information (pose, gravity, vio_pose)
            intrinsic: Camera intrinsics
            save_dir: Directory to save debug images
            draw_flag: Flag for drawing debug images
            use_transformed_image: Whether the image is already transformed
            H_final: Homography matrix to transform points back to original image (if use_transformed_image is True)
            original_image_shape: (height, width) of original image (required if use_transformed_image is True)
            original_image: Original image before transformation (required if use_transformed_image is True and should_draw is True)
        
        Returns:
            tuple: (success, position, rotation, cov_rot, cov_tran, error_msg)
        """
        should_draw = self._should_draw(draw_flag, save_dir)
        start_time = time.time()

        query_keypoints, query_descriptors, query_scores = self._extract_features_sift(image)
        if query_keypoints is None or query_descriptors is None:
            error_msg = f"Task {self.current_task_id} : Failed to extract features"
            logger.error(error_msg)
            if should_draw and self.marker_image is not None:
                try:
                    vis_query = cv2.resize(image, (1920, int(image.shape[0] * 1920 / image.shape[1])))
                    marker_ratio = vis_query.shape[0] / self.marker_image.shape[0]
                    vis_marker = cv2.resize(self.marker_image, (int(self.marker_image.shape[1] * marker_ratio), vis_query.shape[0]))
                    combined = np.hstack([vis_query, vis_marker])
                    cv2.putText(combined, error_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imwrite(save_dir + '_matches.jpg', combined)
                except Exception as e:
                    logger.warning(f"Task {self.current_task_id} : Failed to save visualization: {e}")
            return False, None, None, None, None, error_msg

        logger.info(f"Task {self.current_task_id} : Feature extraction took {time.time() - start_time:.5f} seconds")
        start_time = time.time()

        match_indices, match_scores = self._match_features_sift(
            query_keypoints, query_descriptors, self.marker_keypoints, self.marker_descriptors)
        matches = [(i, match_indices[i], match_scores[i]) for i in range(len(match_indices)) if match_indices[i] >= 0]
        
        logger.info(f"Task {self.current_task_id} : Feature matching took {time.time() - start_time:.5f} seconds, found {len(matches)} matches")
        start_time = time.time()

        if len(matches) == 0:
            error_msg = f"Task {self.current_task_id} : No matches found"
            logger.error(error_msg)
            if should_draw:
                self._save_failure_visualization(image, query_keypoints, [], save_dir, error_msg,
                                                 use_transformed_image=use_transformed_image, H_final=H_final,
                                                 original_image=original_image)
            return False, None, None, None, None, error_msg
        
        # Check minimum matches threshold
        if len(matches) < self.min_matches:
            error_msg = f"Task {self.current_task_id} : Insufficient matches: {len(matches)} < {self.min_matches} (threshold)"
            logger.warning(error_msg)
            if should_draw:
                self._save_failure_visualization(image, query_keypoints, matches, save_dir, error_msg,
                                                 use_transformed_image=use_transformed_image, H_final=H_final,
                                                 original_image=original_image)
            return False, None, None, None, None, error_msg

        num_query_keypoints = len(query_keypoints[0])
        matched_2d_raw = []
        matched_2d_marker = []
        matched_indices_raw = []
        
        for query_idx, marker_idx, match_score in matches:
            if query_idx >= num_query_keypoints or marker_idx >= len(self.marker_keypoints[0]):
                continue
            matched_2d_raw.append(query_keypoints[0][query_idx])
            matched_2d_marker.append(self.marker_keypoints[0][marker_idx])
            matched_indices_raw.append((query_idx, marker_idx, match_score))
        
        if len(matched_2d_raw) == 0:
            error_msg = f"Task {self.current_task_id} : No valid matches after validation"
            logger.error(error_msg)
            if should_draw:
                self._save_failure_visualization(image, query_keypoints, matches, save_dir, error_msg,
                                                 use_transformed_image=use_transformed_image, H_final=H_final,
                                                 original_image=original_image)
            return False, None, None, None, None, error_msg

        matched_2d_array = np.array(matched_2d_raw, dtype=np.float32)
        matched_2d_marker_array = np.array(matched_2d_marker, dtype=np.float32)
        
        # Step 1: Compute homography to get marker corner positions
        homography = None
        homography_mask = None
        if len(matched_2d_raw) >= 8:
            try:
                homography, homography_mask = cv2.findHomography(
                    matched_2d_marker_array, matched_2d_array, cv2.RANSAC,
                    ransacReprojThreshold=8.0, maxIters=3000, confidence=0.99)
                
                if homography is None or (homography_mask is not None and np.sum(homography_mask) < 8):
                    homography, homography_mask = cv2.findHomography(
                        matched_2d_marker_array, matched_2d_array, cv2.LMEDS,
                        maxIters=2000, confidence=0.99)
                
                if homography is not None and homography_mask is not None:
                    filtered_matches = [matched_indices_raw[i] for i in range(len(matched_indices_raw))
                                       if homography_mask.ravel()[i] > 0]
                    if len(filtered_matches) > 0:
                        matches = filtered_matches
                        logger.info(f"Task {self.current_task_id} : Homography filtered: {len(filtered_matches)}/{len(matched_indices_raw)}")
            except Exception as e:
                logger.warning(f"Task {self.current_task_id} : Homography error: {e}")
        
        # Step 2: Use homography to compute marker corner positions in query image
        marker_corners_2d = None
        marker_corners_3d = None
        use_corner_points = False
        
        if homography is not None:
            try:
                # Get marker image dimensions
                marker_height_px, marker_width_px = self.marker_image.shape[:2]
                
                # Define marker corners in marker image (pixel coordinates)
                marker_corners_marker_image = np.array([
                    [0, 0],                          # top-left
                    [marker_width_px, 0],            # top-right
                    [marker_width_px, marker_height_px],  # bottom-right
                    [0, marker_height_px]            # bottom-left
                ], dtype=np.float32).reshape(-1, 1, 2)
                
                # Transform marker corners to query image using homography
                marker_corners_2d = cv2.perspectiveTransform(marker_corners_marker_image, homography)
                marker_corners_2d = marker_corners_2d.reshape(-1, 2)
                
                # Convert marker corners to 3D coordinates (in marker coordinate system)
                # Marker coordinate system: center at (0,0,0), x right, y down
                # Corner order: top-left, top-right, bottom-right, bottom-left
                marker_corners_3d = np.array([
                    self._pixel_to_marker_coords(0, 0),  # top-left: (-half_width, -half_height, 0)
                    self._pixel_to_marker_coords(marker_width_px, 0),  # top-right: (+half_width, -half_height, 0)
                    self._pixel_to_marker_coords(marker_width_px, marker_height_px),  # bottom-right: (+half_width, +half_height, 0)
                    self._pixel_to_marker_coords(0, marker_height_px)  # bottom-left: (-half_width, +half_height, 0)
                ], dtype=np.float32)
                
                # Validate corner points (check if they are within image bounds)
                h, w = image.shape[:2]
                valid_corners = True
                for corner in marker_corners_2d:
                    if corner[0] < -w * 0.1 or corner[0] > w * 1.1 or \
                       corner[1] < -h * 0.1 or corner[1] > h * 1.1:
                        valid_corners = False
                        break
                
                if valid_corners:
                    use_corner_points = True
                    logger.info(f"Task {self.current_task_id} : Using homography-computed marker corners for PnP")
                else:
                    logger.warning(f"Task {self.current_task_id} : Marker corners out of bounds, falling back to feature points")
            except Exception as e:
                logger.warning(f"Task {self.current_task_id} : Failed to compute marker corners from homography: {e}")
        
        # Step 3: Prepare 2D-3D correspondences for PnP
        marker_2d_map = {(q_idx, m_idx): matched_2d_marker[i] 
                        for i, (q_idx, m_idx, _) in enumerate(matched_indices_raw)}
        matched_2d = []
        matched_3d = []
        matched_feature_scores = []
        
        if use_corner_points:
            # Use 4 corner points as primary correspondences
            # Note: marker_corners_2d is in transformed image coordinates if use_transformed_image=True
            matched_2d = marker_corners_2d.tolist()
            matched_3d = marker_corners_3d.tolist()
            matched_feature_scores = [1.0] * 4  # High confidence for corner points
            
            # Optionally add some high-quality feature points for robustness
            # Add top N inlier feature points (sorted by match score)
            if homography_mask is not None:
                inlier_indices = [i for i in range(len(matched_indices_raw)) if homography_mask.ravel()[i] > 0]
                if len(inlier_indices) > 0:
                    # Get match scores for inliers
                    inlier_scores = []
                    for idx in inlier_indices:
                        q_idx, m_idx, match_score = matched_indices_raw[idx]
                        if q_idx < num_query_keypoints:
                            score = query_scores[q_idx] if query_scores is not None and q_idx < len(query_scores) else match_score
                            inlier_scores.append((idx, score))
                    
                    # Sort by score and take top ones
                    inlier_scores.sort(key=lambda x: x[1], reverse=True)
                    num_additional_points = min(10, len(inlier_scores))  # Add up to 10 additional points
                    
                    for idx, _ in inlier_scores[:num_additional_points]:
                        q_idx, m_idx, match_score = matched_indices_raw[idx]
                        if q_idx >= num_query_keypoints or m_idx >= len(self.marker_keypoints[0]):
                            continue
                        marker_pt = marker_2d_map.get((q_idx, m_idx), self.marker_keypoints[0][m_idx])
                        # query_keypoints[0][q_idx] is in transformed image coordinates if use_transformed_image=True
                        matched_2d.append(query_keypoints[0][q_idx])
                        matched_3d.append(self._pixel_to_marker_coords(marker_pt[0], marker_pt[1]))
                        score = query_scores[q_idx] if query_scores is not None and q_idx < len(query_scores) else match_score
                        matched_feature_scores.append(score if score > 0 else 0.5)
                    
                    logger.info(f"Task {self.current_task_id} : Added {num_additional_points} high-quality feature points to corner points")
        else:
            # Fallback: use all feature matches (original method)
            for query_idx, marker_idx, match_score in matches:
                if query_idx >= num_query_keypoints or marker_idx >= len(self.marker_keypoints[0]):
                    continue
                marker_pt = marker_2d_map.get((query_idx, marker_idx), self.marker_keypoints[0][marker_idx])
                # query_keypoints[0][query_idx] is in transformed image coordinates if use_transformed_image=True
                matched_2d.append(query_keypoints[0][query_idx])
                matched_3d.append(self._pixel_to_marker_coords(marker_pt[0], marker_pt[1]))
                matched_feature_scores.append(query_scores[query_idx] if query_scores is not None and query_idx < len(query_scores) 
                                             else (match_score if match_score > 0 else 0.5))
        
        # Convert matched_2d to numpy array for transformation
        matched_2d = np.array(matched_2d, dtype=np.float32)
        matched_3d = np.array(matched_3d, dtype=np.float32)
        matched_feature_scores = np.array(matched_feature_scores)
        
        # If using transformed image, transform ALL matched points (corner points + feature points) back to original image coordinates
        if use_transformed_image and H_final is not None and original_image_shape is not None:
            # Transform matched_2d from transformed image to original image
            matched_2d_original = self._transform_points_from_transformed_to_original(matched_2d, H_final)
            
            # Filter points that are within original image bounds
            # Note: Use a margin to account for:
            # 1. Numerical errors in homography transformation (floating point precision)
            # 2. Points near image boundaries that may have small errors after inverse transformation
            h_orig, w_orig = original_image_shape
            margin_ratio = 0.05  # 5% margin (reduced from 10% for stricter filtering)
            valid_mask = (
                (matched_2d_original[:, 0] >= -w_orig * margin_ratio) & 
                (matched_2d_original[:, 0] < w_orig * (1 + margin_ratio)) &
                (matched_2d_original[:, 1] >= -h_orig * margin_ratio) & 
                (matched_2d_original[:, 1] < h_orig * (1 + margin_ratio))
            )
            matched_2d = matched_2d_original[valid_mask]
            matched_3d = matched_3d[valid_mask]
            matched_feature_scores = matched_feature_scores[valid_mask]
            logger.info(f"Task {self.current_task_id} : After transforming back to original image: {len(matched_2d)} valid matches (including corner points)")
            
            # Use original image shape for reproj_error calculation since matched_2d is now in original image coordinates
            h_orig, w_orig = original_image_shape
            reproj_error = max(1, int(math.sqrt(h_orig ** 2 + w_orig ** 2) * 0.02))
        else:
            # Use current image shape for reproj_error calculation
            reproj_error = max(1, int(math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2) * 0.02))
        
        if len(matched_2d) == 0:
            error_msg = f"Task {self.current_task_id} : No valid matches after homography filtering"
            logger.error(error_msg)
            if should_draw:
                self._save_failure_visualization(image, query_keypoints, matches, save_dir, error_msg,
                                                 use_transformed_image=use_transformed_image, H_final=H_final,
                                                 original_image=original_image)
            return False, None, None, None, None, error_msg

        # matched_2d, matched_3d, and matched_feature_scores are already numpy arrays at this point
        # If using transformed image, they have already been converted to original image coordinates above
        # reproj_error has already been calculated above based on whether we're using transformed image or not
        logger.info(f"Task {self.current_task_id} : Found {len(matched_2d)} matches")
        gravity_camera_expected = None

        # gravity prior is not used for marker pose estimation
        # if image_prior is not None and image_prior[1] is not None and np.linalg.norm(image_prior[1]) > 1e-3:
        #     gravity_camera_expected = image_prior[1]
        #     logger.info(f"Task {self.current_task_id} : use gravity prior {gravity_camera_expected}")

        success, rot_vec, tran_vec, inliers = pnp_ransac_optimization_with_gravity(
            matched_3d,
            matched_2d,
            intrinsic[0],
            intrinsic[1],
            GRAVITY_WORLD,
            gravity_camera_expected,
            GRAVITY_WEIGHT,
            gravity_angle_error_threshold_degree=GRAVITY_ERROR_THRESHOLD_DEGREE,
            ransac_reproj_error=reproj_error,
            ransac_iterations=100,
            ransac_confidence=0.98,
        )

        if not success:
            error_msg = f"Task {self.current_task_id} : PnP failed to find a solution"
            logger.error(error_msg)
            if should_draw:
                self._save_failure_visualization(image, query_keypoints, matches, save_dir, error_msg,
                                                 use_transformed_image=use_transformed_image, H_final=H_final,
                                                 original_image=original_image)
            return False, None, None, None, None, error_msg

        rotation_matrix, _ = cv2.Rodrigues(rot_vec)
        position = tran_vec.flatten()
        quat = rotation_matrix_to_quaternion(rotation_matrix)
        logger.info(f"Task {self.current_task_id} : PnP took {time.time() - start_time:.5f} seconds")

        def compute_cov_scale(num_inliers, min_inliers=5, max_inliers=50, min_scale=0.8, max_scale=1.2):
            if num_inliers <= min_inliers:
                return max_scale
            elif num_inliers >= max_inliers:
                return min_scale
            return max_scale - (num_inliers - min_inliers) / (max_inliers - min_inliers) * (max_scale - min_scale)

        inliers_flat = inliers.squeeze()
        if inliers_flat.ndim == 0:
            inliers_flat = np.array([inliers_flat])
        
        matched_2d_inliers = matched_2d[inliers_flat]
        matched_3d_inliers = matched_3d[inliers_flat]
        projected, jacobian = cv2.projectPoints(
            matched_3d_inliers, rot_vec, tran_vec, intrinsic[0], intrinsic[1]
        )
        projected = projected.reshape(-1, 2)
        reproj_errors = np.linalg.norm(projected - matched_2d_inliers, axis=1)
        
        inlier_ratio = len(inliers_flat) / len(matched_2d) if len(matched_2d) > 0 else 0
        avg_reproj_error = np.mean(reproj_errors)
        max_reproj_error = np.max(reproj_errors) if len(reproj_errors) > 0 else 0.0
        logger.info(
            f"Task {self.current_task_id} : Inliers: {len(inliers_flat)}/{len(matched_2d)} "
            f"(ratio: {inlier_ratio:.3f}), Avg reproj error: {avg_reproj_error:.3f}px, Max reproj error: {max_reproj_error:.3f}px"
        )
        
        # Quality control checks
        def handle_quality_failure(error_msg):
            """Helper function to handle quality check failure."""
            logger.warning(error_msg)
            if should_draw:
                self._save_failure_visualization(image, query_keypoints, matches, save_dir, error_msg,
                                               use_pose=True, matched_2d_inliers=matched_2d_inliers,
                                               matched_3d_inliers=matched_3d_inliers, quat=quat,
                                               position=position, camera_matrix=intrinsic[0],
                                               dist_coeffs=intrinsic[1],
                                               use_transformed_image=use_transformed_image, H_final=H_final,
                                               original_image=original_image)
            return False, None, None, None, None, error_msg
        
        if len(inliers_flat) < self.min_inliers:
            return handle_quality_failure(
                f"Task {self.current_task_id} : Insufficient inliers: {len(inliers_flat)} < {self.min_inliers} (threshold)"
            )
        
        if inlier_ratio < self.min_inlier_ratio:
            return handle_quality_failure(
                f"Task {self.current_task_id} : Inlier ratio too low: {inlier_ratio:.3f} < {self.min_inlier_ratio} (threshold)"
            )
        
        if avg_reproj_error > self.max_avg_reproj_error:
            return handle_quality_failure(
                f"Task {self.current_task_id} : Average reprojection error too high: {avg_reproj_error:.3f}px > {self.max_avg_reproj_error}px (threshold)"
            )
        
        if max_reproj_error > self.max_reproj_error:
            return handle_quality_failure(
                f"Task {self.current_task_id} : Maximum reprojection error too high: {max_reproj_error:.3f}px > {self.max_reproj_error}px (threshold)"
            )

        matched_feature_scores_inliers = matched_feature_scores[inliers_flat]
        feat_scores = matched_feature_scores_inliers / (np.max(matched_feature_scores_inliers) + 1e-6)
        # Get match scores for inliers
        match_scores_array = np.array([match_scores[i] for i, _, _ in matches])
        match_scores_inliers = match_scores_array[inliers_flat]
        match_scores_norm = match_scores_inliers / (np.max(match_scores_inliers) + 1e-6) if len(match_scores_inliers) > 0 else np.ones(len(inliers_flat))
        weights = feat_scores * match_scores_norm + 1e-6
        weights = weights / np.sum(weights)
        sigma2 = np.sum(weights * (reproj_errors**2)) / np.sum(weights)

        J = jacobian[:, :6]
        W = np.diag(np.repeat(weights, 2))
        cov_pose = np.linalg.inv(J.T @ W @ J + 1e-6 * np.eye(6)) * np.abs(sigma2)
        cov_pose *= compute_cov_scale(len(inliers))
        cov_pose = 0.5 * (cov_pose + cov_pose.T)
        cov_rot = cov_pose[:3, :3]
        cov_tran = cov_pose[3:, 3:]

        if should_draw:
            self._draw_matches(image, matched_2d_inliers, matched_3d_inliers, 
                             quat, position, intrinsic[0], intrinsic[1], save_dir, cov_pose,
                             use_transformed_image=use_transformed_image, H_final=H_final,
                             original_image=original_image)
            try:
                # For simple visualization, use transformed image to show matches on transformed image
                # query_keypoints are extracted from 'image' (which is transformed if use_transformed_image=True)
                self._draw_matches_simple(image, query_keypoints, self.marker_image, 
                                        self.marker_keypoints, matches, save_dir, None)
                
                # If using transformed image, also draw matches on original image
                # Transform query_keypoints from transformed image to original image coordinates
                if use_transformed_image and H_final is not None and original_image is not None:
                    try:
                        query_keypoints_original = self._transform_query_keypoints_to_original(query_keypoints, H_final)
                        self._draw_matches_simple(original_image, query_keypoints_original, self.marker_image, 
                                                self.marker_keypoints, matches, save_dir + '_original', None)
                    except Exception as e:
                        logger.warning(f"Task {self.current_task_id} : Failed to draw matches on original image: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"Task {self.current_task_id} : Failed to save simple visualization: {e}")

        error_msg = "No error"
        return True, position.tolist(), quat.tolist(), cov_rot, cov_tran, error_msg

    def _draw_matches_simple(self, query_image, query_keypoints, marker_image, marker_keypoints, 
                            matches, save_dir, error_msg=None):
        try:
            from visualization.draw_map import draw_matches

            size_ratio = 1920 / query_image.shape[1]
            vis_query = cv2.resize(query_image, (1920, int(query_image.shape[0] * size_ratio)))
            marker_ratio = vis_query.shape[0] / marker_image.shape[0]
            vis_marker = cv2.resize(marker_image, (int(marker_image.shape[1] * marker_ratio), vis_query.shape[0]))
            query_kpts_scaled = query_keypoints[0] * size_ratio
            marker_kpts_scaled = marker_keypoints[0] * marker_ratio

            max_query_idx = len(query_keypoints[0])
            max_marker_idx = len(marker_keypoints[0])
            match_indices_array = np.full(max_query_idx, -1, dtype=np.int32)
            match_scores_array = np.zeros(max_query_idx, dtype=np.float32)
            for query_idx, marker_idx, match_score in matches:
                if query_idx < max_query_idx and marker_idx < max_marker_idx:
                    match_indices_array[query_idx] = marker_idx
                    match_scores_array[query_idx] = match_score

            matches_img = draw_matches(vis_query, vis_marker, query_kpts_scaled, marker_kpts_scaled,
                                     match_indices_array, match_scores_array, max_display=500,
                                     score_thresh=0.0, return_img=True)

            if error_msg:
                text_size = cv2.getTextSize(error_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(matches_img, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
                cv2.putText(matches_img, error_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(matches_img, f"Matches: {len(matches)}", (20, matches_img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imwrite(save_dir + '_matches_simple.jpg', matches_img)
            logger.info(f"Task {self.current_task_id} : Saved match visualization to {save_dir}_matches_simple.jpg")
        except Exception as e:
            logger.warning(f"Task {self.current_task_id} : Failed to draw simple matches: {e}", exc_info=True)

    def _draw_matches(self, query_image, matched_2d, matched_3d, quat, position, 
                     camera_matrix, dist_coeffs, save_dir, cov_pose=None,
                     use_transformed_image=False, H_final=None, original_image=None):
        try:
            from visualization.draw_map import draw_2d_3d_matches

            size_ratio = 1920 / query_image.shape[1]
            vis_image = cv2.resize(query_image, (1920, int(query_image.shape[0] * size_ratio)))
            scaled_camera_matrix = size_ratio * camera_matrix.copy()
            scaled_camera_matrix[2, 2] = 1.0
            
            # Convert quaternion to qvec format [x, y, z, w] for draw_2d_3d_matches
            qvec = [quat[1], quat[2], quat[3], quat[0]]
            
            # If using transformed image, we need special handling for projection
            if use_transformed_image and H_final is not None and original_image is not None:
                # For transformed image, we need to:
                # 1. Project 3D points in original image coordinates first
                # 2. Transform the projected points to transformed image coordinates
                # 3. Transform matched_2d to transformed image coordinates
                
                # First, project 3D points in original image coordinates
                from visualization.draw_map import project_points
                from scipy.spatial.transform import Rotation as R
                rotation_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                rvec, _ = cv2.Rodrigues(rotation_matrix)
                # Use original camera matrix for projection (not scaled)
                points_2d_original = project_points(matched_3d, rvec, position, camera_matrix, dist_coeffs)
                
                # Transform projected points from original to transformed image coordinates
                points_2d_transformed = self._transform_points_from_original_to_transformed(points_2d_original, H_final)
                
                # Transform matched_2d from original to transformed image coordinates
                matched_2d_transformed = self._transform_points_from_original_to_transformed(matched_2d, H_final)
                
                # Scale both to visualization size
                matched_2d_scaled = size_ratio * matched_2d_transformed
                points_2d_scaled = size_ratio * points_2d_transformed
                
                # Draw manually instead of using draw_2d_3d_matches
                from visualization.draw_map import draw_points, draw_lines
                matches_img = vis_image.copy()
                draw_points(matches_img, points_2d_scaled, radius=6)
                draw_points(matches_img, matched_2d_scaled, radius=6, color=(255, 0, 0))
                draw_lines(matches_img, matched_2d_scaled, points_2d_scaled)
                cv2.putText(matches_img, "#inlers:" + str(len(matched_2d)), (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            else:
                # Normal case: not using transformed image
                matched_2d_scaled = size_ratio * matched_2d
                matches_img = draw_2d_3d_matches(vis_image, qvec, position, matched_2d_scaled,
                                                 matched_3d, scaled_camera_matrix, dist_coeffs, return_img=True)

            # Draw marker boundary and coordinate axes on transformed image
            # Pass original camera matrix and size_ratio for proper scaling
            self._draw_marker_boundary_and_axes(matches_img, quat, position, camera_matrix, dist_coeffs,
                                               use_transformed_image=use_transformed_image, 
                                               H_final=H_final,
                                               original_image_shape=original_image.shape[:2] if original_image is not None else None,
                                               size_ratio=size_ratio)

            if cov_pose is not None:
                pose_diag = np.diag(cov_pose)
                cv2.putText(matches_img, f"Rot cov: {pose_diag[0:3]}", (20, matches_img.shape[0] - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(matches_img, f"Trans cov: {pose_diag[3:6]}", (20, matches_img.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imwrite(save_dir + '_matches.jpg', matches_img)
            logger.info(f"Task {self.current_task_id} : Saved match visualization to {save_dir}_matches.jpg")
            
            # If using transformed image, also draw on original image
            if use_transformed_image and H_final is not None and original_image is not None:
                try:
                    size_ratio_orig = 1920 / original_image.shape[1]
                    vis_original = cv2.resize(original_image, (1920, int(original_image.shape[0] * size_ratio_orig)))
                    scaled_camera_matrix_orig = size_ratio_orig * camera_matrix.copy()
                    scaled_camera_matrix_orig[2, 2] = 1.0
                    
                    # matched_2d is already in original image coordinates (converted in _estimate_pose_from_image)
                    # So we just need to scale it to visualization size
                    matched_2d_original_scaled = size_ratio_orig * matched_2d
                    
                    matches_img_orig = draw_2d_3d_matches(vis_original, qvec, position, matched_2d_original_scaled,
                                                         matched_3d, scaled_camera_matrix_orig, dist_coeffs, return_img=True)
                    
                    # Draw marker boundary and coordinate axes on original image
                    # Pass original camera matrix (not scaled) and size_ratio
                    self._draw_marker_boundary_and_axes_on_original(matches_img_orig, quat, position, 
                                                                    camera_matrix, dist_coeffs, size_ratio_orig)
                    
                    if cov_pose is not None:
                        pose_diag = np.diag(cov_pose)
                        cv2.putText(matches_img_orig, f"Rot cov: {pose_diag[0:3]}", (20, matches_img_orig.shape[0] - 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.putText(matches_img_orig, f"Trans cov: {pose_diag[3:6]}", (20, matches_img_orig.shape[0] - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    cv2.imwrite(save_dir + '_matches_original.jpg', matches_img_orig)
                    logger.info(f"Task {self.current_task_id} : Saved match visualization on original image to {save_dir}_matches_original.jpg")
                except Exception as e:
                    logger.warning(f"Task {self.current_task_id} : Failed to draw matches on original image: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Task {self.current_task_id} : Failed to draw matches: {e}", exc_info=True)

    def _draw_marker_boundary_and_axes(self, image, quat, position, camera_matrix, dist_coeffs,
                                      use_transformed_image=False, H_final=None, original_image_shape=None,
                                      size_ratio=1.0):
        """
        Draw marker boundary (4 corners) and coordinate axes on image.
        
        Marker coordinate system:
        - Origin (0,0,0) is at the center of the marker
        - X-axis: right (positive x direction in image)
        - Y-axis: down (positive y direction in image)
        - Z-axis: up (perpendicular to marker plane, pointing outward)
        
        Args:
            image: Image to draw on (may be original or transformed, at visualization scale)
            quat: Quaternion [w, x, y, z] (marker_to_camera pose, in original image coordinates)
            position: Translation [x, y, z] (marker_to_camera pose, in original image coordinates)
            camera_matrix: Original camera matrix (not scaled)
            dist_coeffs: Distortion coefficients
            use_transformed_image: Whether the image is transformed
            H_final: Homography matrix from original to transformed image (in original image coordinates)
            original_image_shape: (height, width) of original image (if use_transformed_image is True)
            size_ratio: Scaling ratio from original to visualization size
        """
        try:
            from utils.math_utils import quaternion_to_rotation_matrix

            marker_corners_3d, marker_axes_3d = self._get_marker_corners_and_axes_3d()

            R = quaternion_to_rotation_matrix(quat)
            rvec, _ = cv2.Rodrigues(R)

            # Pose is in original image coordinates.
            # Project 3D points using original camera matrix, then scale to visualization size.
            # If image is transformed, transform points to transformed image coordinates.
            
            # Project in original image coordinates
            corners_2d_original, _ = cv2.projectPoints(marker_corners_3d, rvec, position, camera_matrix, dist_coeffs)
            corners_2d_original = corners_2d_original.reshape(-1, 2)

            axes_2d_original, _ = cv2.projectPoints(marker_axes_3d, rvec, position, camera_matrix, dist_coeffs)
            axes_2d_original = axes_2d_original.reshape(-1, 2)
            
            # If image is transformed, transform points to transformed image coordinates
            if use_transformed_image and H_final is not None and original_image_shape is not None:
                # Transform corners and axes from original to transformed image (at original scale)
                corners_2d_transformed = self._transform_points_from_original_to_transformed(corners_2d_original, H_final)
                axes_2d_transformed = self._transform_points_from_original_to_transformed(axes_2d_original, H_final)
                # Scale to visualization size
                corners_2d = (corners_2d_transformed * size_ratio).astype(np.int32)
                axes_2d = (axes_2d_transformed * size_ratio).astype(np.int32)
            else:
                # Scale to visualization size
                corners_2d = (corners_2d_original * size_ratio).astype(np.int32)
                axes_2d = (axes_2d_original * size_ratio).astype(np.int32)

            self._draw_marker_boundary_and_axes_on_image(image, corners_2d, axes_2d)
        except Exception as e:
            logger.warning(f"Task {self.current_task_id} : Failed to draw marker boundary and axes: {e}")

    def _draw_marker_boundary_and_axes_on_original(self, image, quat, position, 
                                                   camera_matrix, dist_coeffs, size_ratio_orig):
        """
        Draw marker boundary (4 corners) and coordinate axes on original image.
        The pose is estimated in original image coordinates (after transforming matched points back),
        so we can directly project 3D points to original image coordinates.
        
        Args:
            image: Visualization image (resized original image)
            quat: Quaternion [w, x, y, z] (marker_to_camera pose, in original image coordinates)
            position: Translation [x, y, z] (marker_to_camera pose, in original image coordinates)
            camera_matrix: Original camera matrix (not scaled)
            dist_coeffs: Distortion coefficients
            size_ratio_orig: Scaling ratio for original image visualization
        """
        try:
            from utils.math_utils import quaternion_to_rotation_matrix

            marker_corners_3d, marker_axes_3d = self._get_marker_corners_and_axes_3d()

            R = quaternion_to_rotation_matrix(quat)
            rvec, _ = cv2.Rodrigues(R)

            # Pose is in original image coordinates, so directly project to original image coordinates
            corners_2d_original, _ = cv2.projectPoints(marker_corners_3d, rvec, position, 
                                                       camera_matrix, dist_coeffs)
            corners_2d_original = corners_2d_original.reshape(-1, 2)
            corners_2d_scaled = (corners_2d_original * size_ratio_orig).astype(np.int32)
            
            axes_2d_original, _ = cv2.projectPoints(marker_axes_3d, rvec, position, 
                                                    camera_matrix, dist_coeffs)
            axes_2d_original = axes_2d_original.reshape(-1, 2)
            axes_2d_scaled = (axes_2d_original * size_ratio_orig).astype(np.int32)

            self._draw_marker_boundary_and_axes_on_image(image, corners_2d_scaled, axes_2d_scaled)
        except Exception as e:
            logger.warning(f"Task {self.current_task_id} : Failed to draw marker boundary and axes on original image: {e}")

