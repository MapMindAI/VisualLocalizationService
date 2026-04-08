#!/usr/bin/env python3
import grpc
import logging
import os
from concurrent import futures
import threading
import signal
import time
import queue
import multiprocessing as mp
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
# When marker_pose_server.pyc is in /app/marker_pose_service/, 
# _current_dir will be /app/marker_pose_service
# _visual_pose_service_dir will be /app, which contains proto/ and pose_server_common.py
_visual_pose_service_dir = os.path.dirname(_current_dir)
if _visual_pose_service_dir not in sys.path:
    sys.path.insert(0, _visual_pose_service_dir)

from proto.map import visual_pose_service_pb2_grpc
from pose_server_common import (
    setup_logging,
    PoseTask,
    SerializablePoseTask,
    process_task,
    make_failed_response,
)

try:
    import marker_pose_estimator
except ImportError as e:
    logging.error(f"Failed to import marker_pose_estimator: {e}")
    sys.exit(1)


def worker_process(
    worker_id,
    marker_config,
    log_flag,
    log_dir,
    task_queue,
    result_queue,
    shutdown_event,
):
    try:
        estimator = marker_pose_estimator.MarkerPoseEstimator(marker_config)
        estimator.initialize(marker_config)
        logging.info(f"Worker {worker_id} started successfully")

        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=1)
                if task is None:
                    break
                waiting_time = time.time() - task.add_to_queue_time
                logging.info(f"Worker {worker_id} - Task {task.id} : {task.queue_length} tasks ahead, waiting {waiting_time:.5f}s")
                process_task(worker_id, task, estimator, log_flag, log_dir)
                result_queue.put(task)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                if 'task' in locals():
                    task.error = str(e)
                    result_queue.put(task)

    except Exception as e:
        logging.error(f"Worker {worker_id} initialization failed: {e}")
    finally:
        logging.info(f"Worker {worker_id} shutting down")




class MarkerPoseService(visual_pose_service_pb2_grpc.VisualPoseServiceServicer):
    def __init__(
        self,
        marker_config: dict,
        log_flag: int,
        log_dir: str,
        pool_size: int = 4,
    ):
        self.marker_config = marker_config
        self.log_flag = log_flag
        self.log_dir = log_dir
        self.pool_size = pool_size

        self.task_queue = mp.Queue(maxsize=2000)
        self.result_queue = mp.Queue()
        self.shutdown_event = mp.Event()

        self.pending_tasks = {}
        self.task_results = {}
        self.result_lock = threading.Lock()

        self.workers = []
        for i in range(pool_size):
            worker = mp.Process(target=worker_process, args=(
                i, marker_config, log_flag, log_dir, self.task_queue, self.result_queue, self.shutdown_event))
            worker.start()
            self.workers.append(worker)

        self.result_thread = threading.Thread(target=self._result_handler)
        self.result_thread.start()

    def _result_handler(self):
        while not self.shutdown_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                if result is None:
                    break
                with self.result_lock:
                    self.task_results[result.id] = result
                    if result.id in self.pending_tasks:
                        self.pending_tasks[result.id].set()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Result handler error: {e}")

    def GetPoseFromImage(self, request, context):
        logging.info("Received GetPoseFromImage request")
        send_to_server_time = time.time() - (request.image_timestamp / 1e6)
        logging.info(f"Sending grpc msg took {send_to_server_time:.5f}s")

        try:
            if self.task_queue.qsize() >= 2000:
                error_msg = "Server busy: too many requests"
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details(error_msg)
                return make_failed_response(-1, error_msg)
        except:
            pass

        try:
            if not request.image or not request.image.image:
                error_msg = "No image data received"
                logging.error(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                response = make_failed_response(-1, error_msg)
                return response
            if not request.intrinsics:
                error_msg = "No intrinsic data received or no distortion_coeff data received"
                logging.error(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                response = make_failed_response(-1, error_msg)
                return response

        except Exception as e:
            error_msg = f"Unexpected error in request check: {e}"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            response = make_failed_response(-1, error_msg)
            return response

        # Create task
        task = PoseTask(request, context.peer(), time.time(), self.task_queue.qsize())
        serializable_task = SerializablePoseTask(task)

        # Create event
        done_event = threading.Event()
        with self.result_lock:
            self.pending_tasks[task.id] = done_event

        # Send task to worker
        try:
            self.task_queue.put(serializable_task, timeout=5)
        except queue.Full:
            with self.result_lock:
                if task.id in self.pending_tasks:
                    del self.pending_tasks[task.id]
            error_msg = "Server busy: task queue full"
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(error_msg)
            response = make_failed_response(task.id, error_msg)
            return response

        # Wait for task finishing
        if not done_event.wait(timeout=30):
            with self.result_lock:
                if task.id in self.pending_tasks:
                    del self.pending_tasks[task.id]
                if task.id in self.task_results:
                    del self.task_results[task.id]
            error_msg = f"Task {task.id} : Request processing timeout"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(error_msg)
            response = make_failed_response(task.id, error_msg)
            return response

        # Get result
        result = None
        with self.result_lock:
            if task.id in self.task_results:
                result = self.task_results.pop(task.id)
            if task.id in self.pending_tasks:
                del self.pending_tasks[task.id]

        if result is None:
            error_msg = f"Task {task.id} : No result found"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            response = make_failed_response(task.id, error_msg)
            return response

        if result.error:
            error_msg = f"Task {task.id} : {result.error}"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(result.error)
            response = make_failed_response(task.id, error_msg)
            return response

        return result.response

    def shutdown(self):
        logging.info("Shutting down MarkerPoseService...")
        self.shutdown_event.set()
        for _ in range(self.pool_size):
            try:
                self.task_queue.put(None, timeout=1)
            except:
                pass
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        try:
            self.result_queue.put(None, timeout=1)
        except:
            pass
        self.result_thread.join(timeout=5)


class MarkerPoseServer:
    def __init__(
        self,
        marker_config: dict,
        log_flag: int,
        log_dir: str,
        port: int = 40011,
        max_workers: int = 10,
        pool_size: int = 4,
    ):
        self.marker_config = marker_config
        self.port = port
        self.max_workers = max_workers
        self.grpc_server = None
        self.marker_pose_service = None
        self.running = False
        self.log_flag = log_flag
        self.log_dir = log_dir
        self.pool_size = pool_size
        self._shutdown_event = threading.Event()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if not self._shutdown_event.is_set():
            logging.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
            self.stop()

    def start(self):
        try:
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
            marker_pose_service = MarkerPoseService(self.marker_config, self.log_flag, self.log_dir, self.pool_size)
            visual_pose_service_pb2_grpc.add_VisualPoseServiceServicer_to_server(marker_pose_service, self.grpc_server)
            self.marker_pose_service = marker_pose_service
            listen_addr = f'0.0.0.0:{self.port}'
            self.grpc_server.add_insecure_port(listen_addr)
            self.grpc_server.start()
            self.running = True
            logging.info(f"Marker Pose Server listening on {listen_addr}")
            logging.info(f"Config: {self.marker_config}, Max workers: {self.max_workers}, Pool size: {self.pool_size}")
            self._wait_for_termination()

        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            self.stop()

    def stop(self):
        if self.marker_pose_service:
            self.marker_pose_service.shutdown()

        if self.grpc_server and self.running:
            logging.info("Stopping Marker Pose Server...")
            self.grpc_server.stop(grace=5.0)
            self.running = False
            logging.info("Marker Pose Server stopped")

    def _wait_for_termination(self):
        try:
            while not self._shutdown_event.wait(0.5):
                pass
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
            self._shutdown_event.set()
        finally:
            self.stop()


def load_marker_config(config_path: str = None) -> dict:
    default_config = {
        "marker_image": "/Mobili/python/visual_pose_service/marker_pose_service/marker_image/dm_final.jpg",
        "marker_width": 0.3,
        "enable_vio_retry": True,  # Whether to retry with VIO pose transformation after first attempt fails
    }
    if config_path is None or not config_path:
        logging.info("No marker config file provided, using default")
        return default_config
    if not os.path.exists(config_path):
        logging.warning(f"Marker config file not found: {config_path}, using default")
        return default_config
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        merged_config = default_config.copy()
        merged_config.update(config)
        logging.info(f"Loaded marker configuration from: {config_path}")
        return merged_config
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file {config_path}: {e}, using default")
        return default_config
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {e}, using default")
        return default_config


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Marker Pose gRPC Server')
    parser.add_argument(
        '--marker_config',
        type=str,
        default=None,
        help='Path to marker configuration JSON file. If not provided, uses default configuration.',
    )
    parser.add_argument('--port', type=int, default=40010, help='Server port (default: 40011)')
    parser.add_argument(
        '--max_workers', type=int, default=10, help='Maximum number of worker threads (default: 10)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)',
    )
    parser.add_argument(
        '--log_flag',
        type=int,
        default=0,
        help='Whether to save logs, draw matches and projection images',
    )
    parser.add_argument(
        '--logs_dir',
        type=str,
        default='/Mobili/python/logs',
        help='Logs directory',
    )
    parser.add_argument(
        '--pool_size',
        type=int,
        default=4,
        help='Estimator instance pool size (default: 4)',
    )

    args = parser.parse_args()
    marker_config = load_marker_config(args.marker_config)
    log_dir = setup_logging(args.log_flag, args.logs_dir, args.log_level,
                           server_name='marker_pose_server', logger_name='marker_pose_estimator')
    logging.info("Marker configuration:")
    for key, value in marker_config.items():
        logging.info(f"  {key}: {value}")
    marker_pose_server = MarkerPoseServer(marker_config=marker_config, port=args.port,
                                         max_workers=args.max_workers, log_flag=args.log_flag,
                                         log_dir=log_dir, pool_size=args.pool_size)
    logging.info("Starting Marker Pose Server...")
    marker_pose_server.start()


if __name__ == '__main__':
    main()

