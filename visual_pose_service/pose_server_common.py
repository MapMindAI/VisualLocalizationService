#!/usr/bin/env python3
"""
Common utilities and classes for pose estimation servers.
"""
import cv2
import os
import numpy as np
import logging
import threading
import time
from datetime import datetime
from typing import Tuple, List, Optional, Any
import queue
import itertools
import multiprocessing as mp

from proto.map import visual_pose_service_pb2


def setup_logging(
    log_flag: int,
    logs_dir: str,
    log_level: str = 'INFO',
    server_name: str = 'pose_server',
    logger_name: str = None,
):
    """
    Setup logging configuration based on log_flag
    log_flag: 0 - console only, 1 - both console and file
    server_name: Name for log directory and file
    logger_name: Name of the logger to configure (e.g., 'visual_localizer', 'marker_pose_estimator')
    """
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level))

    handlers = []
    log_dir = ""

    if log_flag == 0:
        # Console only
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    elif log_flag == 1:
        # Both console and file
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(logs_dir, f"{server_name}_{timestamp}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"{server_name}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

        print(f"Logs will be written to console and file: {log_file}")

    # Add all handlers to root logger
    for handler in handlers:
        logging.getLogger().addHandler(handler)

    # Also configure specific logger if provided
    if logger_name:
        try:
            specific_logger = logging.getLogger(logger_name)
            specific_logger.propagate = False  # Prevent duplicate output
            for handler in handlers:
                specific_logger.addHandler(handler)
            specific_logger.setLevel(getattr(logging, log_level))
        except:
            pass

    return log_dir


class PoseTask:
    _id_counter = itertools.count(0)
    _id_lock = threading.Lock()

    def __init__(self, request, context_peer, time, queue_length):
        with PoseTask._id_lock:
            self.id = next(PoseTask._id_counter)

        self.request = request
        self.context_peer = context_peer
        self.response = None
        self.error = None
        # queue state
        self.add_to_queue_time = time
        self.queue_length = queue_length
        # parse info
        self.image = self._proto_image_to_cv_mat(request.image.image)
        self.intrinsics = self._extract_intrinsics(request.intrinsics)
        self.image_prior = self._extract_prior(request)

    def _proto_image_to_cv_mat(self, image_proto) -> Optional[np.ndarray]:
        """Convert protobuf to OpenCV Mat"""
        try:
            start_time = time.time()
            image_bytes = image_proto.encoded_str
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            convert_time = time.time() - start_time
            logging.info(
                f"Task {self.id} : Converted image to cv::Mat in {convert_time:.5f} seconds"
            )

            if image is None:
                logging.error(f"Task {self.id} : Failed to decode image")
                return None

            logging.info(f"Task {self.id} : Successfully decoded image with shape: {image.shape}")
            return image

        except Exception as e:
            logging.error(f"Task {self.id} : Failed to convert proto image to cv::Mat: {e}")
            return None

    def _extract_prior(self, request):
        if not request.localization_prior:
            return None

        prior_pose = None
        if request.localization_prior.pose:
            rot = request.localization_prior.pose.rotation
            trans = request.localization_prior.pose.translation
            prior_pose = np.array([rot.w, rot.x, rot.y, rot.z, trans.x, trans.y, trans.z])
        prior_gravity = None
        if request.localization_prior.gravity:
            vec = request.localization_prior.gravity
            prior_gravity = np.array([vec.x, vec.y, vec.z])
        vio_pose = None
        if request.localization_prior.vio_pose:
            rot = request.localization_prior.vio_pose.rotation
            trans = request.localization_prior.vio_pose.translation
            vio_pose = np.array([rot.w, rot.x, rot.y, rot.z, trans.x, trans.y, trans.z])
        return (prior_pose, prior_gravity, vio_pose)

    def _extract_intrinsics(self, intrinsics_proto):
        start_time = time.time()

        # Extract camera intrinsics from proto，return tuple (intrinsic, distortion)
        fx, fy, cx, cy = (
            intrinsics_proto.fx,
            intrinsics_proto.fy,
            intrinsics_proto.cx,
            intrinsics_proto.cy,
        )
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if len(intrinsics_proto.distortion_coef) != 4:
            distortion_coef = np.zeros(4)
        else:
            distortion_coef = np.array(intrinsics_proto.distortion_coef)

        get_intrinsics_time = time.time() - start_time
        logging.info(
            f"Task {self.id} : Retrieved camera intrinsics in {get_intrinsics_time:.5f} seconds"
        )
        return (intrinsic, distortion_coef)


class SerializablePoseTask:
    """Serializable PoseTask for process communication"""

    def __init__(self, pose_task):
        self.id = pose_task.id
        self.request = pose_task.request
        self.context_peer = pose_task.context_peer
        self.add_to_queue_time = pose_task.add_to_queue_time
        self.queue_length = pose_task.queue_length
        self.image = pose_task.image
        self.intrinsics = pose_task.intrinsics
        self.image_prior = pose_task.image_prior
        self.response = None
        self.error = None


def _peer_to_client_host(context_peer: str) -> str:
    """Extract host from gRPC context.peer() (e.g. ipv4:127.0.0.1:12345 -> 127.0.0.1)."""
    if not context_peer:
        return "unknown"
    if context_peer.startswith("ipv4:"):
        rest = context_peer[len("ipv4:") :]
        host, _, _ = rest.rpartition(":")
        return host if host else "unknown"
    if context_peer.startswith("ipv6:"):
        return "ipv6"
    # Fallback: host:port without prefix
    host, _, _ = context_peer.rpartition(":")
    return host if host else context_peer


def make_save_dir(context_peer, timestamp, task_id, log_dir):
    client_ip = _peer_to_client_host(context_peer)
    save_dir = os.path.join(log_dir, client_ip, 'request-' + str(timestamp) + f'-task-{task_id}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save_request(image_timestamp, image, intrinsics, image_prior, save_dir):
    image_path = os.path.join(save_dir, 'request_image.jpg')
    cv2.imwrite(image_path, image)

    request_file_path = os.path.join(save_dir, 'request_info.log')
    with open(request_file_path, 'w') as log_file:
        log_file.write(f"Timestamp: {image_timestamp}\n")
        log_file.write(f"Camera Intrinsics:\n")
        log_file.write(f"fx: {intrinsics[0][0, 0]}\n")
        log_file.write(f"fy: {intrinsics[0][1, 1]}\n")
        log_file.write(f"cx: {intrinsics[0][0, 2]}\n")
        log_file.write(f"cy: {intrinsics[0][1, 2]}\n")
        log_file.write(f"Distortion Coefficients: {intrinsics[1]}\n")
        log_file.write(f"Prior: {image_prior}\n")


def estimate_pose(
    estimator,
    image_timestamp,
    image: np.ndarray,
    image_prior,
    camera_intrinsics: tuple,
    save_dir: str,
    log_flag,
    task_id,
    worker_id,
) -> Tuple[bool, Optional[List[float]], Optional[List[float]], Any, Any, Optional[str]]:
    """
    Generic pose estimation function that works with any estimator.
    
    Args:
        estimator: Any estimator object with:
            - _initialized attribute
            - set_current_task_id(task_id) method
            - estimate_pose_from_byte(image_bytes, image_prior, intrinsic, save_dir, log_flag) method
    """
    if not estimator._initialized:
        error_msg = f"Worker {worker_id} - Model not initialized"
        logging.error(error_msg)
        return False, None, None, None, None, error_msg

    start_time = time.time()
    # encode image to JPEG format
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        error_msg = f"Worker {worker_id} - Task {task_id} : Failed to encode image to JPEG"
        logging.error(error_msg)
        return False, None, None, None, None, error_msg

    image_bytes = buffer.tobytes()
    encode_time = time.time() - start_time
    logging.info(
        f"Worker {worker_id} - Task {task_id} : Encoded image time: {encode_time:.5f} seconds"
    )

    estimator.set_current_task_id(task_id)
    process_save_dir = os.path.join(save_dir, f"worker_{worker_id}")

    result = estimator.estimate_pose_from_byte(
        image_bytes, image_prior, camera_intrinsics, process_save_dir, log_flag
    )

    ok, position_array, rotation_array, cov_rot, cov_tran, error_msg = result

    if not ok:
        logging.error(f"Worker {worker_id} - Task {task_id} : Failed to estimate pose: {error_msg}")
        return False, None, None, None, None, error_msg

    # Convert numpy arrays to lists
    position = (
        position_array.tolist() if hasattr(position_array, 'tolist') else list(position_array)
    )
    rotation = (
        rotation_array.tolist() if hasattr(rotation_array, 'tolist') else list(rotation_array)
    )

    logging.info(
        f"Worker {worker_id} - Task {task_id} : Estimated pose - Position: {position}, Rotation: {rotation}"
    )
    return ok, position, rotation, cov_rot, cov_tran, error_msg


def make_failed_response(task_id, error_msg):
    # task_id < 0 means that the request has not been created as a task
    task_id = int(task_id)
    task_id_supplement = ''
    if task_id < 0:
        task_id_supplement = '(unknown, has not been created)'
    response = visual_pose_service_pb2.PoseResponse()
    response.server_log = f"Task_id: {task_id}{task_id_supplement}, err_msg: {error_msg}"
    return response


def process_task(worker_id, task, estimator, log_flag, log_dir):
    """
    Generic task processing function.
    
    Args:
        estimator: Any estimator object compatible with estimate_pose()
    """
    try:
        start_time_original = time.time()

        if task.image is None:
            error_msg = (
                f"Worker {worker_id} - Task {task.id} : Failed to decode image or image is empty"
            )
            logging.error(error_msg)
            task.error = error_msg
            return

        # Per-request debug dirs and request dumps only when log_flag=1 (and log_dir is set for file logging)
        if log_flag and log_dir:
            save_dir = make_save_dir(task.context_peer, task.request.image_timestamp, task.id, log_dir)
            save_request(
                task.request.image_timestamp,
                task.image,
                task.intrinsics,
                task.image_prior,
                save_dir,
            )
        else:
            save_dir = ""

        # Estimate pose
        start_time = time.time()
        ok, position, rotation, cov_rot, cov_tran, error_msg = estimate_pose(
            estimator,
            task.request.image_timestamp,
            task.image,
            task.image_prior,
            task.intrinsics,
            save_dir,
            log_flag,
            task.id,
            worker_id,
        )

        estimate_time = time.time() - start_time
        logging.info(
            f"Worker {worker_id} - Task {task.id} : Pose estimated in {estimate_time:.5f} seconds"
        )

        if not ok:
            error_msg = f"Worker {worker_id} - Task {task.id} : {error_msg}"
            logging.error(error_msg)
            task.error = error_msg
            return

        # Create response
        if cov_rot is None or cov_tran is None:
            cov_rot = np.zeros((3, 3), dtype=np.float64)
            cov_tran = np.zeros((3, 3), dtype=np.float64)
            logging.warning(
                f"Worker {worker_id} - Task {task.id} : Covariance matrices are None, using zero matrices"
            )
        else:
            logging.info(
                f"Worker {worker_id} - Task {task.id} : Covariance {np.diag(cov_rot)} {np.diag(cov_tran)}"
            )

        response = visual_pose_service_pb2.PoseResponse()
        response.status = 1  # success
        response.timestamp = task.request.image_timestamp  # timestamp of the request
        response.pose.translation.x = position[0]
        response.pose.translation.y = position[1]
        response.pose.translation.z = position[2]
        response.pose.rotation.x = rotation[1]
        response.pose.rotation.y = rotation[2]
        response.pose.rotation.z = rotation[3]
        response.pose.rotation.w = rotation[0]

        data_tran = np.array(cov_tran, dtype=np.float64)
        response.translation_covariance.row = 3
        response.translation_covariance.column = 3
        response.translation_covariance.data = data_tran.tobytes()
        data_rot = np.array(cov_rot, dtype=np.float64)
        response.rotation_covariance.row = 3
        response.rotation_covariance.column = 3
        response.rotation_covariance.data = data_rot.tobytes()

        response.server_log = f"Task_id: {task.id}, success!"
        task.response = response

        elapsed_time = time.time() - start_time_original
        logging.info(
            f"Worker {worker_id} - Task {task.id} : Task completed in {elapsed_time:.5f} seconds"
        )
        logging.info(
            f"Worker {worker_id} - Task {task.id} : Successfully processed image and estimated pose"
        )

    except Exception as e:
        error_msg = f"Worker {worker_id} - Task {task.id} : Failed to process task: {e}"
        logging.error(error_msg)
        task.error = error_msg

