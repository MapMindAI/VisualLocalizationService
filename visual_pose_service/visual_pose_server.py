#!/usr/bin/env python3
import grpc
import logging
from concurrent import futures
import threading
import signal
import time
import queue
import multiprocessing as mp
import os
import sys

_service_root = os.path.dirname(os.path.abspath(__file__))
if _service_root not in sys.path:
    sys.path.insert(0, _service_root)
from proto.map import visual_pose_service_pb2, visual_pose_service_pb2_grpc

# Import common utilities
from pose_server_common import (
    setup_logging,
    PoseTask,
    SerializablePoseTask,
    process_task,
    make_failed_response,
)

try:
    import visual_localizer
except ImportError as e:
    logging.error(f"Failed to import visual_localizer: {e}")
    sys.exit(1)


def worker_process(
    worker_id,
    model_path,
    sp_address,
    top_k,
    sp_model_name,
    sg_model_name,
    log_flag,
    log_dir,
    task_queue,
    result_queue,
    shutdown_event,
):
    try:
        # Create localizer instance
        localizer = visual_localizer.VisualLocalizer(sp_address, top_k, sp_model_name, sg_model_name)
        localizer.load_database(model_path)

        logging.info(f"Worker {worker_id} started successfully")

        while not shutdown_event.is_set():
            try:
                # Get task
                task = task_queue.get(timeout=1)

                # When receive None task, stop the process
                if task is None:
                    break

                waiting_time = time.time() - task.add_to_queue_time
                logging.info(
                    f"Worker {worker_id} - Task {task.id} : {task.queue_length} tasks are ahead of the task, waiting for {waiting_time:.5f}s!"
                )

                # Process task (localizer is passed as estimator)
                process_task(worker_id, task, localizer, log_flag, log_dir)

                # Return result
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


# This class implements the gRPC service for visual pose estimation
class VisualPoseService(visual_pose_service_pb2_grpc.VisualPoseServiceServicer):
    def __init__(
        self,
        model_path: str,
        sp_address: str,
        top_k: int,
        log_flag: int,
        log_dir: str,
        pool_size: int = 4,
        sp_model_name: str = "superpoint_onnx",
        sg_model_name: str = "lightglue_onnx",
    ):
        self.model_path = model_path
        self.log_flag = log_flag
        self.log_dir = log_dir
        self.pool_size = pool_size
        self.sp_address = sp_address
        self.top_k = top_k
        self.sp_model_name = sp_model_name
        self.sg_model_name = sg_model_name

        self.task_queue = mp.Queue(maxsize=2000)
        self.result_queue = mp.Queue()
        self.shutdown_event = mp.Event()

        self.pending_tasks = {}  # events for tasks being process, task_id -> threading.Event
        self.task_results = {}  # task_id -> result
        self.result_lock = threading.Lock()

        # start workers
        self.workers = []
        for i in range(pool_size):
            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    model_path,
                    sp_address,
                    top_k,
                    sp_model_name,
                    sg_model_name,
                    log_flag,
                    log_dir,
                    self.task_queue,
                    self.result_queue,
                    self.shutdown_event,
                ),
            )
            worker.start()
            self.workers.append(worker)

        # start thread for result
        self.result_thread = threading.Thread(target=self._result_handler)
        self.result_thread.start()

    def _result_handler(self):
        """Process results from workers"""
        while not self.shutdown_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                # When receive None task, stop the thread
                if result is None:
                    break
                task_id = result.id
                with self.result_lock:
                    # notify that the result has been put into task_results
                    self.task_results[task_id] = result
                    if task_id in self.pending_tasks:
                        self.pending_tasks[task_id].set()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Result handler error: {e}")

    def GetPoseFromImage(self, request, context):
        logging.info("Received GetPoseFromImage request")
        start_time = time.time()
        start_time_original = start_time

        send_to_server_time = start_time - (request.image_timestamp / 1e6)
        logging.info(f"Sending grpc msg took {send_to_server_time:.5f}s!")

        try:
            # Checking the queue size in a non-blocking way
            if self.task_queue.qsize() >= 2000:
                error_msg = "Server busy: too many requests"
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details()
                response = make_failed_response(-1, error_msg)
                return response
        except:
            pass  # qsize() may not be supported in some systems

        # basic check for request
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
        logging.info("Shutting down VisualPoseService...")
        self.shutdown_event.set()

        # Send None task to all workers in order to stop them
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


class VisualPoseServer:
    def __init__(
        self,
        model_path: str,
        sp_address: str,
        top_k: int,
        log_flag: int,
        log_dir: str,
        port: int = 40010,
        max_workers: int = 10,
        pool_size: int = 4,
        sp_model_name: str = "superpoint_onnx",
        sg_model_name: str = "lightglue_onnx",
    ):
        self.model_path = model_path
        self.port = port
        self.max_workers = max_workers
        self.sp_address = sp_address
        self.grpc_server = None
        self.visual_pose_service = None
        self.running = False
        self.top_k = top_k
        self.log_flag = log_flag
        self.log_dir = log_dir
        self.pool_size = pool_size
        self.sp_model_name = sp_model_name
        self.sg_model_name = sg_model_name
        self._shutdown_event = threading.Event()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if not self._shutdown_event.is_set():
            logging.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
            self.stop()

    def start(self):
        """Start Visual Pose Server"""
        try:
            # Create gRPC server
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ('grpc.max_send_message_length', -1),
                    ('grpc.max_receive_message_length', -1),
                ],
            )

            visual_pose_service = VisualPoseService(
                self.model_path,
                self.sp_address,
                self.top_k,
                self.log_flag,
                self.log_dir,
                self.pool_size,
                self.sp_model_name,
                self.sg_model_name,
            )
            visual_pose_service_pb2_grpc.add_VisualPoseServiceServicer_to_server(
                visual_pose_service, self.grpc_server
            )
            self.visual_pose_service = visual_pose_service

            listen_addr = f'0.0.0.0:{self.port}'
            self.grpc_server.add_insecure_port(listen_addr)

            # Start gRPC server
            self.grpc_server.start()
            self.running = True

            logging.info(f"Visual Pose Server is listening on {listen_addr}")
            logging.info(f"Model path: {self.model_path}")
            logging.info(f"Max workers: {self.max_workers}")
            logging.info(f"Pool size: {self.pool_size}")

            # Wait for termination signal
            self._wait_for_termination()

        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            self.stop()

    def stop(self):
        if self.visual_pose_service:
            self.visual_pose_service.shutdown()

        if self.grpc_server and self.running:
            logging.info("Stopping Visual Pose Server...")
            self.grpc_server.stop(grace=5.0)
            self.running = False
            logging.info("Visual Pose Server stopped")

    def _wait_for_termination(self):
        """Wait for server termination"""
        try:
            while not self._shutdown_event.wait(0.5):
                pass
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
            self._shutdown_event.set()
        finally:
            self.stop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visual Pose gRPC Server')
    parser.add_argument(
        '--model_path',
        type=str,
        # required=True,
        default='',
        help='Path to the COLMAP model directory',
    )
    parser.add_argument('--port', type=int, default=40010, help='Server port (default: 40010)')
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
        '--sp_address',
        type=str,
        default='127.0.0.1:8001',
        help='Address for superpoint service',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number for retrieval images',
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
        default='./logs',
        help='Logs directory',
    )
    parser.add_argument(
        '--pool_size',
        type=int,
        default=4,
        help='Localizer instance pool size (default: 4)',
    )
    parser.add_argument(
        '--sp_model_name',
        type=str,
        default='superpoint_onnx',
        help='Triton model name for SuperPoint (default: superpoint_onnx)',
    )
    parser.add_argument(
        '--sg_model_name',
        type=str,
        default='lightglue_onnx',
        help='Triton model name for SuperGlue/LightGlue (default: lightglue_onnx)',
    )

    args = parser.parse_args()

    # Setup logging based on log_flag
    log_dir = setup_logging(
        args.log_flag, 
        args.logs_dir, 
        args.log_level,
        server_name='visual_pose_server',
        logger_name='visual_localizer'
    )

    # Cerate and start the VisualPoseServer
    visual_pose_server = VisualPoseServer(
        model_path=args.model_path,
        port=args.port,
        max_workers=args.max_workers,
        sp_address=args.sp_address,
        top_k=args.top_k,
        log_flag=args.log_flag,
        log_dir=log_dir,
        pool_size=args.pool_size,
        sp_model_name=args.sp_model_name,
        sg_model_name=args.sg_model_name,
    )

    logging.info("Starting Visual Pose Server...")
    visual_pose_server.start()


if __name__ == '__main__':
    main()
