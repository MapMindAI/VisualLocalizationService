import pycolmap
import sqlite3
import numpy as np
import cv2
import os
import time
import logging
import math
import matplotlib.pyplot as plt
from feature.superpoint import SuperPoint
from feature.superglue import SuperGlue
from retrieval.make_retrieval_db import BowRetireval
from visualization import draw_map
from utils.math_utils import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
from optimization.pnp_optimization import pnp_ransac_optimization_with_gravity

logger = logging.getLogger(__name__)

MESH_NAME = 'output/tsdf_mesh.ply'
DATABASE_NAME = 'database_3d.db'

GRAVITY_WORLD = np.array([0, 0, 1])
GRAVITY_WEIGHT = 1000.0
GRAVITY_ERROR_THRESHOLD_DEGREE = 40.0

USE_POSE_PRIOR_FOR_RETRIEVAL = False
MIN_NUM_OF_MATCHED_FEATURES = 10


class VisualLocalizer:
    """
    Visual Localizer for estimating camera pose using COLMAP database.
    """

    def __init__(self, sp_address: str, top_k=3):
        self.database_path = None
        self.db_images = None  # {image_id: cv Mat} from database
        self.db_descriptors = None  # {image_id: np.array[N,Dim_of_Desc]} from database
        self.db_point3D_depth = None  # {image_id: np.array[N,3]} from database
        self.db_keypoints = None  # {image_id: np.array[N,2]} from database
        self.db_image_poses = None

        self.db_image_shape = None  # shape of database images
        self._initialized = False

        self.superpoint = SuperPoint(sp_address)
        self.superglue = SuperGlue(sp_address)
        self.retrieval = None
        self.top_k = top_k  # retrieval parameter

        self.current_task_id = None

    def load_database(self, model_path: str):
        """
        Load database from the given model path.
        """

        start_time = time.time()

        database_path = os.path.join(model_path, DATABASE_NAME)
        self.database_path = database_path

        # Load database info
        if (
            self.db_descriptors is None
            or self.db_point3D_depth is None
            or self.db_keypoints is None
        ):
            self._load_info_from_db(database_path)
            logger.info(f"Load database info from {database_path}")

        load_db_info_time = time.time() - start_time
        logger.info(f"Database info loaded in {load_db_info_time:.5f} seconds")
        start_time = time.time()

        # Load retrieval BoW model
        self.retrieval = BowRetireval(model_path, self.top_k, self.db_image_poses)

        self._initialized = True
        logger.info("Visual localizer initialization completed!")

    def _load_info_from_db(self, db_path: str):
        """Load descriptors from the SQLite database(db file)"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            logger.info(f"Connected to database at {db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise RuntimeError('Failed to connect to the database.')

        # Load camera parameters, only one camera currently
        camera_id = 1
        cursor.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No camera found with ID: {camera_id}")

        width, height = result[2], result[3]
        self.db_image_shape = np.array([[height, width]], dtype=np.int32)

        # Load image_names and images
        images = {}
        image_poses = {}
        cursor.execute("SELECT image_id, name, qw, qx, qy, qz, tx, ty, tz FROM images")
        for image_id, name, qw, qx, qy, qz, tx, ty, tz in cursor.fetchall():
            image_path = os.path.join(
                os.path.dirname(self.database_path), os.path.join('images', name)
            )
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_info = tuple([name, image])
            images[image_id] = image_info

            # Convert quaternion to rotation matrix
            quat = np.array([qw, qx, qy, qz])
            trans = np.array([tx, ty, tz])
            rotation_matrix = quaternion_to_rotation_matrix(quat).transpose()
            trans = -rotation_matrix @ trans
            image_poses[image_id] = tuple([quat, trans])

        self.images = images
        self.db_image_poses = image_poses
        logger.info(f"Load {len(self.images)} images from {db_path}")

        # Load db_descriptors
        # TODO(wenhao): Add more checks for db info
        descriptors = {}
        cursor.execute("SELECT image_id, rows, cols, data FROM descriptors")
        for image_id, rows, cols, data in cursor.fetchall():
            desc = np.frombuffer(data, dtype=np.uint8)
            descriptors[image_id] = desc.reshape(-1, cols)
        self.db_descriptors = descriptors
        logger.info(f"Loaded descriptors for {len(self.db_descriptors)} images from {db_path}")

        # Load depth point3d
        point3D_depth = {}
        keypoints = {}
        cursor.execute("SELECT image_id, rows, cols, data, coords_3d FROM keypoints")
        for image_id, rows, cols, data, coords_3d_blob in cursor.fetchall():
            if coords_3d_blob is None:
                logger.info(f'image_id: {image_id}, coords_3d_blob is None!')
            coords_3d = np.frombuffer(coords_3d_blob, dtype=np.float64)
            point3D_depth[image_id] = coords_3d.reshape(-1, 3)
            keypoint = np.frombuffer(data, dtype=np.float32)
            keypoints[image_id] = keypoint.reshape(-1, cols)
        self.db_point3D_depth = point3D_depth
        self.db_keypoints = keypoints
        logger.info(
            f"Loaded keypoints and point3D_depth for {len(self.db_descriptors)} images from {db_path}"
        )

        conn.close()

    def _collect_candidate_3D_points_from_depth(self, top_k_ids):
        top_k_ids = set(top_k_ids)  # Use set for faster lookup
        candidate_xy = []
        candidate_xyz = []
        candidate_desc = []

        for image_id in top_k_ids:
            # every point2d has responding point3d,
            keypoints_desc = self.db_descriptors[image_id]
            point3D_depth = self.db_point3D_depth[image_id]
            keypoints = self.db_keypoints[image_id]
            for i in range(len(keypoints_desc)):
                candidate_xyz.append(point3D_depth[i])
                candidate_desc.append(keypoints_desc[i])
                candidate_xy.append(keypoints[i])

        if len(candidate_xyz) == 0:
            logger.error("No candidate 3D points found!")
            return None, None, None
        return (
            np.array(candidate_xy, dtype=np.float32),
            np.array(candidate_xyz, dtype=np.float32),
            np.array(candidate_desc, dtype=np.uint8),
        )

    def set_current_task_id(self, task_id):
        self.current_task_id = task_id

    def _extract_features(self, image):
        if image is None:
            logger.error(f"Task {self.current_task_id} : In extraction_feature: image is None")
            return None, None
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        keypoints, descriptors, scores = self.superpoint.run(image)
        return keypoints, descriptors, scores

    def draw_matches_with_candidates(
        self, query_image, matches_2d_3d_img, top_k_ids, save_dir, cov_pose=None, pose=None
    ):
        def get_image(image_id):
            image_info = self.images[image_id]
            logger.info(f'candidate id: {image_id}, name: {image_info[0]}')
            return image_info[1], image_info[0]

        # Get all candidates image
        max_k = max(3, len(top_k_ids))
        top_k_ids = top_k_ids[:max_k]
        candidate_images = [get_image(i) for i in top_k_ids]

        # Extract features and match
        query_ktps, query_desc, _ = self.superpoint.run(query_image)
        query_img_shape = np.array([[query_image.shape[0], query_image.shape[1]]], dtype=np.int32)

        summary_rows = []
        for img, name in candidate_images:
            # DB images are stored as RGB; query_image is BGR — convert for side-by-side display
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            candidate_img_shape = np.array([[img.shape[0], img.shape[1]]], dtype=np.int32)
            ktps, desc, _ = self.superpoint.run(img)
            match_indices, match_scores = self.superglue.run(
                query_ktps, query_desc, query_img_shape, ktps, desc, candidate_img_shape
            )

            match_vis = draw_map.draw_matches(
                img0=query_image,
                img1=img_bgr,
                kpts0=query_ktps[0],
                kpts1=ktps[0],
                match_indices=match_indices,
                match_scores=match_scores,
                max_display=500,
                score_thresh=0.2,
                return_img=True,
            )

            # Label region
            font_scale = 4
            thickness = 8
            font = cv2.FONT_HERSHEY_SIMPLEX

            (text_w, text_h), baseline = cv2.getTextSize(name, font, font_scale, thickness)
            label_height = text_h + 40  # top margin
            label_img = np.ones((label_height, match_vis.shape[1], 3), dtype=np.uint8) * 255

            text_x = (match_vis.shape[1] - text_w) // 2
            text_y = (label_height + text_h) // 2 - 10  # center with slight upward adjustment

            cv2.putText(label_img, name, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

            # Combine label and match image horizontally
            combined_img = np.vstack([match_vis, label_img])
            summary_rows.append(combined_img)

            spacer = 255 * np.ones((100, combined_img.shape[1], 3), dtype=np.uint8)
            summary_rows.append(spacer)

        # show summary
        summary_image = np.vstack(summary_rows)

        target_height = summary_image.shape[0]
        target_width = int(
            target_height * (matches_2d_3d_img.shape[1] / matches_2d_3d_img.shape[0])
        )
        matches_2d_3d_img = cv2.resize(matches_2d_3d_img, (target_width, target_height))
        summary_image = np.hstack([matches_2d_3d_img, summary_image])

        # resize the image
        output_height = int(summary_image.shape[0] * 1920 / summary_image.shape[1])
        summary_image = cv2.resize(summary_image, (1920, output_height))

        if cov_pose is not None:
            pose_diag = np.diag(cov_pose)
            cv2.putText(
                summary_image,
                f"{pose_diag[0:3]}",
                (20, output_height - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                summary_image,
                f"{pose_diag[3:6]}",
                (20, output_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
        if pose is not None:
            quat = pose[0]
            position = pose[1].flatten()
            text_pos = f"Position: {position}"
            text_quat = f"Quaternion(wxyz): {quat}"
            cv2.putText(
                summary_image,
                text_pos,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                summary_image,
                text_quat,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        cv2.imwrite(save_dir + '_matches.jpg', summary_image)

    def _match_features_with_candidates(self, image, image_prior, intrinsic, save_dir, draw_flag):
        start_time = time.time()

        # Extract features from the image
        keypoints, descriptors, scores = self._extract_features(image)
        extract_time = time.time() - start_time
        logger.info(
            f"Task {self.current_task_id} : Feature extraction took {extract_time:.5f} seconds"
        )
        start_time = time.time()

        if keypoints is None or descriptors is None:
            error_msg = "keypoints/descriptors is None!"
            return False, None, None, None, None, error_msg

        # Find top k ids
        # top_k_ids = self.retrieval.retrieve_general(descriptors[0], image_prior)
        if (
            USE_POSE_PRIOR_FOR_RETRIEVAL
            and (image_prior is not None)
            and (image_prior[0] is not None)
            and self.retrieval.check_prior_pose_valid(image_prior[0])
        ):
            logger.info(f"Task {self.current_task_id} : use pose prior {image_prior[0]}")
            top_k_ids = self.retrieval.retrieve_nearest_pose(image_prior[0])
        else:
            top_k_ids = self.retrieval.retrieve_bow(descriptors[0])
        top_k_time = time.time() - start_time
        logger.info(
            f"Task {self.current_task_id} : Top-k frame search took {top_k_time:.5f} seconds"
        )
        start_time = time.time()

        logger.info(
            f"Task {self.current_task_id} : Top-{len(top_k_ids)} candidate frames: {top_k_ids}"
        )

        # Collect Candidate 3D points
        candidate_xy, candidate_xyz, candidate_desc = self._collect_candidate_3D_points_from_depth(
            top_k_ids
        )
        candidate_time = time.time() - start_time
        logger.info(
            f"Task {self.current_task_id} : Collected candidate 3D points took {candidate_time:.5f} seconds"
        )

        if candidate_xy is None or candidate_xyz is None or candidate_desc is None:
            error_msg = "candidate_xy/candidate_xyz/candidate_desc is None!"
            return False, None, None, None, None, error_msg

        # Match features and get pose
        ok, position, rotation, cov_rot, cov_tran, error_msg = self._match_features(
            image,
            keypoints,
            descriptors,
            scores,
            image_prior,
            intrinsic,
            top_k_ids,
            save_dir,
            draw_flag,
            candidate_xy,
            candidate_xyz,
            candidate_desc,
        )

        return ok, position, rotation, cov_rot, cov_tran, error_msg

    def _match_descriptors_superglue(
        self,
        query_keypoints,
        query_desc,
        point3D_xy,
        point3D_xyz,
        point3D_desc,
        img_shape,
        score_thresh=0.0,
    ):
        # superglue input desc should be (1,N,256), type is UINT8
        # keypoints should be FP32
        # max keypoints size supported by superglue is 4096
        max_num = min(self.superglue.max_point_num, point3D_xyz.shape[0])
        score_thresh = np.float32(score_thresh)
        point3D_xy = np.expand_dims(point3D_xy[:max_num, :], axis=0)
        point3D_xy = np.array(point3D_xy, dtype=np.float32)
        point3D_desc = np.expand_dims(point3D_desc[:max_num, :], axis=0)
        match_indices, match_scores = self.superglue.run(
            query_keypoints, query_desc, img_shape, point3D_xy, point3D_desc, self.db_image_shape
        )
        score_thresh = np.mean(match_scores)
        matches = [
            (i, match_indices[i], match_scores[i])
            for i in range(len(match_indices))
            if match_indices[i] >= 0 and match_scores[i] >= score_thresh
        ]

        matched_2d = []
        matched_3d = []
        for i, j, score in matches:
            matched_2d.append(query_keypoints[0][i])
            matched_3d.append(point3D_xyz[j])

        if len(matched_2d) < 4:
            logger.error(f"Task {self.current_task_id} : Not enough matches for pose estimation")
            return None, None, None, None

        return (
            np.array(matched_2d, dtype=np.float32),
            np.array(matched_3d, dtype=np.float32),
            match_indices,
            match_scores,
        )

    def _match_features(
        self,
        image,
        keypoints,
        descriptors,
        scores,
        image_prior,
        intrinsic,
        top_k_ids,
        save_dir,
        draw_flag,
        point3D_xy=None,
        point3D_xyz=None,
        point3D_desc=None,
    ):
        """Match features between input image and 3D points."""

        start_time = time.time()

        if point3D_xyz is None or point3D_desc is None:
            error_msg = f'Task {self.current_task_id} : point3D_xyz or point3D_desc is None! Return'
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        # 1. Match descriptors between input image and 3d points
        img_shape = np.array([[image.shape[0], image.shape[1]]], dtype=np.int32)
        matched_2d, matched_3d, match_indics, match_scores = self._match_descriptors_superglue(
            keypoints, descriptors, point3D_xy, point3D_xyz, point3D_desc, img_shape
        )

        match_time = time.time() - start_time
        logger.info(f"Task {self.current_task_id} : Feature matching took {match_time:.5f} seconds")
        start_time = time.time()

        if matched_2d is None or matched_3d is None:
            # No pose, draw retrieval result only
            error_msg = "matched_2d/matched_3d is None!"
            if draw_flag == 1 and save_dir != '':
                matches_2d_3d_img = image
                self.draw_matches_with_candidates(image, matches_2d_3d_img, top_k_ids, save_dir)
            return False, None, None, None, None, error_msg

        # 2. SuperPoint scores aligned with matched_2d (same filter as _match_descriptors_superglue)
        sg_thresh = float(np.mean(match_scores))
        matched_feature_scores = np.array(
            [
                scores[0, i]
                for i in range(len(match_indics))
                if match_indics[i] >= 0 and match_scores[i] >= sg_thresh
            ],
            dtype=np.float32,
        )
        matched_sg_scores = np.array(
            [
                match_scores[i]
                for i in range(len(match_indics))
                if match_indics[i] >= 0 and match_scores[i] >= sg_thresh
            ],
            dtype=np.float32,
        )

        # 3. Solve PnP
        reproj_error = max(1, int(math.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2) * 0.02))

        gravity_camera_expected = None
        if (
            (image_prior is not None)
            and (image_prior[1] is not None)
            and np.linalg.norm(image_prior[1]) > 1e-3
        ):
            gravity_camera_expected = image_prior[1]
            logger.info(
                f"Task {self.current_task_id} : use gravity prior {gravity_camera_expected}"
            )

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
            ransac_confidence=0.95,
        )

        if not success:
            error_msg = f"Task {self.current_task_id} : PnP failed to find a solution!!!!"
            logger.error(error_msg)
            # No pose, draw retrieval result only
            if draw_flag == 1 and save_dir != '':
                matches_2d_3d_img = image
                self.draw_matches_with_candidates(image, matches_2d_3d_img, top_k_ids, save_dir)
            return False, None, None, None, None, error_msg

        rotation_matrix, _ = cv2.Rodrigues(rot_vec)
        position = tran_vec
        quat = rotation_matrix_to_quaternion(rotation_matrix)

        pnp_time = time.time() - start_time
        logger.info(f"Task {self.current_task_id} : pnp took {pnp_time:.5f} seconds")

        # 4. Estimate cov by weighted Jocobian
        def compute_cov_scale(
            num_inliers, min_inliers=5, max_inliers=50, min_scale=0.8, max_scale=1.2
        ):
            if num_inliers <= min_inliers:
                return max_scale
            elif num_inliers >= max_inliers:
                return min_scale
            else:
                return max_scale - (num_inliers - min_inliers) / (max_inliers - min_inliers) * (
                    max_scale - min_scale
                )

        '''
            cov = sigma^2 * (J^T @ W @ J).inv, in which:
            - sigma: weighted variance of the reprojection loss
            - W: weight diagonal matrix
            - J: Jocobian in projection, (2N, 6)
        '''
        matched_2d_inliers = matched_2d[inliers.squeeze()]
        matched_3d_inliers = matched_3d[inliers.squeeze()]
        projected, jacobian = cv2.projectPoints(
            matched_3d_inliers, rot_vec, tran_vec, intrinsic[0], intrinsic[1]
        )
        projected = projected.reshape(-1, 2)
        reproj_errors = np.linalg.norm(projected - matched_2d_inliers, axis=1)

        feat_scores = matched_feature_scores[inliers] / np.max(matched_feature_scores[inliers])
        match_scores_norm = matched_sg_scores[inliers] / np.max(matched_sg_scores[inliers])
        weights = feat_scores * match_scores_norm + 1e-6
        weights = weights / np.sum(weights)
        sigma2 = np.sum(weights * (reproj_errors**2)) / (
            np.sum(weights)
        )  # variance of reprojection error

        J = jacobian[:, :6]  # (2N, 6), only use rot and tran part
        W = np.diag(np.repeat(weights, 2))  # (2N, 2N)
        cov_pose = np.linalg.inv(J.T @ W @ J + 1e-6 * np.eye(6)) * np.abs(sigma2)
        cov_scale = compute_cov_scale(len(inliers))
        cov_pose *= cov_scale
        cov_pose = 0.5 * (cov_pose + cov_pose.T)
        cov_rot = cov_pose[:3, :3]
        cov_tran = cov_pose[3:, 3:]

        # 5. Draw retrieval results and project 3d points, rotation:xyzw
        if draw_flag == 1 and save_dir != '':
            size_ratio = 1920 / image.shape[1]
            meshed_image = cv2.resize(image, (1920, int(image.shape[0] * size_ratio)))
            camera_matrix = size_ratio * intrinsic[0]
            camera_matrix[2, 2] = 1.0
            matched_2d_inliers = size_ratio * matched_2d_inliers

            ply_path = os.path.join(os.path.dirname(self.database_path), MESH_NAME)
            draw_map.project_mesh_to_image_opencv(
                meshed_image, rotation_matrix, position, ply_path, camera_matrix, save_dir
            )
            matches_2d_3d_img = draw_map.draw_2d_3d_matches(
                meshed_image,
                rot_vec,
                tran_vec,
                matched_2d_inliers,
                matched_3d_inliers,
                camera_matrix,
                intrinsic[1],
                return_img=True,
            )
            self.draw_matches_with_candidates(
                image, matches_2d_3d_img, top_k_ids, save_dir, cov_pose, [quat, position]
            )

        if len(inliers) < MIN_NUM_OF_MATCHED_FEATURES:
            error_msg = f"Task {self.current_task_id} : failed with too few inliers {len(inliers)}!"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        error_msg = "No error"
        return True, position.flatten().tolist(), quat.tolist(), cov_rot, cov_tran, error_msg

    def estimate_pose_from_byte(
        self, image_bytes: bytes, image_prior, intrinsic, save_dir, draw_flag=0
    ):
        """
        Estimate camera pose from image bytes.

        Args:
          image_bytes (bytes): Image data as bytes

        Returns:
          tuple: (position, rotation), where position is [x, y, z] and rotation is quaternion [x, y, z, w]
        """

        if image_bytes is None:
            error_msg = f"Task {self.current_task_id} : Image bytes are None"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg
        # Check if database is loaded
        if not self._initialized:
            error_msg = f"Task {self.current_task_id} : Database has not been loaded!"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        start_time = time.time()
        np_img = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            error_msg = f"Task {self.current_task_id} : Failed to decode image"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        decode_time = time.time() - start_time
        logger.info(f"Task {self.current_task_id} : Image decoded in {decode_time:.5f} seconds")

        ok, position, rotation, cov_rot, cov_tran, error_msg = self._match_features_with_candidates(
            image, image_prior, intrinsic, save_dir, draw_flag
        )
        return ok, position, rotation, cov_rot, cov_tran, error_msg

    def estimate_pose_from_image(self, image: np.ndarray, intrinsic, save_dir, draw_flag=0):
        """
        Estimate camera pose from image array.

        Args:
          image (np.ndarray): Image as numpy array

        Returns:
          tuple: (position, rotation), where position is [x, y, z] and rotation is quaternion [x, y, z, w]
        """

        if image is None:
            error_msg = f"Task {self.current_task_id} : Image is None"
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        # Check if database is loaded
        if not self._initialized:
            error_msg = f"Task {self.current_task_id} : Database has not been loaded! Call load_database() first."
            logger.error(error_msg)
            return False, None, None, None, None, error_msg

        ok, position, rotation, cov_rot, cov_tran, error_msg = self._match_features_with_candidates(
            image, None, intrinsic, save_dir, draw_flag
        )
        return ok, position, rotation, cov_rot, cov_tran, error_msg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running visual_localizer...")

    model_path = "/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice/"
    image_name = "test2.jpg"
    image_path = model_path + image_name
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    localizer = VisualLocalizer('192.168.19.150:8001', 1)

    # Load the database
    localizer.load_database(model_path)

    # Covert image to bytes
    width, height = image.shape[1], image.shape[0]
    fx, fy, cx, cy = 1000, 1000, width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    print("Intrinsic camera matrix K:")
    print(K)
    dist_coeffs = np.zeros(4)
    intrinsic = tuple([K, dist_coeffs])

    _, buf = cv2.imencode(".jpg", image)
    image_bytes = buf.tobytes()
    pose = localizer.estimate_pose_from_byte(
        image_bytes, None, intrinsic, "/Mobili/python/logs/temp", 1
    )
    print(f"Estimated pose:\n{pose}")
