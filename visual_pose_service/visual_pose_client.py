import grpc
import time
import signal
import os
import cv2
import concurrent.futures
import threading
import numpy as np

from proto.map import visual_pose_service_pb2, visual_pose_service_pb2_grpc
from proto.internal.data import image_data_pb2
from proto.api.vision import image_pb2
from proto.api.vision import sensor_pb2

# MULTI_CLIENTS_ADDRESS = '192.168.19.150:40010'
MULTI_CLIENTS_ADDRESS = '127.0.0.1:40010'

# SINGLE_CLIENT_ADDRESS = '192.168.11.194:40010'
# SINGLE_CLIENT_ADDRESS = '192.168.19.150:40010'
SINGLE_CLIENT_ADDRESS = '127.0.0.1:40010'

TIMEOUT = 30.0 # seconds, gRPC timeout for each request

class VisualPoseClient:
    def __init__(self, server_address='127.0.0.1:40010'):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = visual_pose_service_pb2_grpc.VisualPoseServiceStub(self.channel)
        self.running = True

        # TODO(wenhao): Use stream to send multiple images
        self.folder = '/Mobili/Data/'
        self.image_path = self.folder + 'of13.jpg'

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def load_image(self, image_path=None):
        """Load image file"""
        if image_path is None:
            image_path = self.image_path

        try:
            with open(image_path, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Image file not found - {image_path}")
            return None
        except Exception as e:
            print(f"Error: Failed to read image file - {e}")
            return None

    def create_request(self, image_bytes, timestamp=None):
        """Create gRPC request"""
        if timestamp is None:
            timestamp = int(time.time() * 1000000)  # Microsecond timestamp

        # dm.api.proto.vision.Image
        image_msg = image_pb2.Image(encoded_str=image_bytes)

        # dm.internal.proto.data.ImageData
        image_data_msg = image_data_pb2.ImageData(image=image_msg)

        # dm.api.proto.vision.CameraIntrinsics
        height, width = cv2.imread(self.image_path).shape[:2]
        intrinsics = sensor_pb2.CameraIntrinsics(fx=1302, fy=1302, cx=width / 2, cy=height / 2)
        intrinsics.distortion_coef.extend([0.0, 0.0, 0.0, 0.0])

        # mobili.proto.map.ImageRequest
        request = visual_pose_service_pb2.ImageRequest(
            image=image_data_msg, image_timestamp=timestamp, intrinsics=intrinsics
        )

        return request

    def send_request(self):
        """Send single request"""
        
        def decode_cov(covariance):
          rows = covariance.row
          cols = covariance.column
          data_flat = np.frombuffer(covariance.data, dtype=np.float64)
          
          cov = data_flat.reshape(rows, cols)
          return cov
        
        try:
            image_bytes = self.load_image()
            if image_bytes is None:
                return False

            request_start = time.time()
            request = self.create_request(image_bytes)
            response = self.stub.GetPoseFromImage(request, timeout=TIMEOUT)
            totle_response_time = time.time() - request_start
            send_to_client_time = time.time() - (response.timestamp / 1e6)
            rot_cov = decode_cov(response.rotation_covariance)
            tran_cov = decode_cov(response.translation_covariance)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Response status: {response.status}")
            print(f"[{current_time}] Received pose: \n{response.pose}")
            print(f"[{current_time}] rot cov: \n{rot_cov}")
            print(f"[{current_time}] tran cov: \n{tran_cov}")
            print(f"[{current_time}] Timestamp: {response.timestamp}")
            print(f"[{current_time}] Task: {response.server_log}")
            print(f"[{current_time}] Response time: {totle_response_time:.4f}s, send to client time: {send_to_client_time:.4f}")

            print("-" * 50)

            return True
        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            print(f"Request failed: {e}")
            return False

    def run_multi_clients(self, num_clients=5, interval=1.0, run_time=10.0):
        """Simulate multiple concurrent clients"""
        print(
            f"Starting {num_clients} concurrent clients, each with {interval}s interval, run for {run_time}s"
        )
        print("Press Ctrl+C to stop early...")

        start_time = time.time()
        results = {}

        # Shared event to signal stop, allowing all clients to stop
        stop_event = threading.Event()

        def signal_handler_multi(signum, frame):
            print(f"\n[MAIN] Received signal {signum}, stopping all clients...", flush=True)
            stop_event.set()
            self.running = False

        original_sigint = signal.signal(signal.SIGINT, signal_handler_multi)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler_multi)
        
        shared_channel = grpc.insecure_channel(MULTI_CLIENTS_ADDRESS)
        shared_stub = visual_pose_service_pb2_grpc.VisualPoseServiceStub(shared_channel)

        def client_task(client_id, stub):
            try:
                count = 0
                success_count = 0

                # Load image once per client thread
                image_path = '/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice/test2.jpg'
                image_bytes = None
                try:
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                except Exception as e:
                    print(f"[Client {client_id}] Failed to read image: {e}")
                    return

                # Get image shape for intrinsics
                img = cv2.imread(image_path)
                if img is None:
                    print(f"[Client {client_id}] OpenCV failed to load image.")
                    return
                height, width = img.shape[:2]
                intrinsics = sensor_pb2.CameraIntrinsics(fx=1302, fy=1302, cx=width / 2, cy=height / 2)
                intrinsics.distortion_coef.extend([0.0, 0.0, 0.0, 0.0])

                while (time.time() - start_time) < run_time:
                    count += 1
                    request_start_time = time.time()

                    if stop_event.is_set():
                        break

                    timestamp = int(time.time() * 1e6)
                    image_msg = image_pb2.Image(encoded_str=image_bytes)
                    image_data_msg = image_data_pb2.ImageData(image=image_msg)
                    request = visual_pose_service_pb2.ImageRequest(
                        image=image_data_msg,
                        image_timestamp=timestamp,
                        intrinsics=intrinsics
                    )

                    try:
                        response = stub.GetPoseFromImage(request, timeout=TIMEOUT)
                        totle_response_time = time.time() - request_start_time
                        send_to_client_time = time.time() - (response.timestamp / 1e6)
                        task_id = response.server_log
                        success_count += 1

                        print(f"[Client {client_id}] Request #{count}-task-{task_id} SUCCESS - Totle time: {totle_response_time:.4f}s, Send to client time: {send_to_client_time:.4f}")
                    except grpc.RpcError as e:
                        print(f"[Client {client_id}] gRPC Error: {e.code()} - {e.details()}")
                    except Exception as e:
                        print(f"[Client {client_id}] Request Error: {e}")

                    time.sleep(max(0, interval - (time.time() - request_start_time)))

                results[client_id] = {'total': count, 'success': success_count}
            finally:
                channel.close()
                print(f"[Client {client_id}] Finished: {success_count}/{count} success")

        print("Starting all client threads...", flush=True)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [executor.submit(client_task, i, shared_stub) for i in range(num_clients)]

                # Polling wait, not blocking wait
                remaining_time = run_time
                while remaining_time > 0 and not stop_event.is_set():
                    try:
                        # Wait 0.5 seconds or until a task is completed
                        done, not_done = concurrent.futures.wait(
                            futures, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        if not not_done:
                            break

                        remaining_time = run_time - (time.time() - start_time)

                    except KeyboardInterrupt:
                        print("\n[MAIN] KeyboardInterrupt caught, stopping...", flush=True)
                        stop_event.set()
                        break

                # If there are unfinished tasks, set the stop flag and wait
                if not_done:
                    stop_event.set()
                    print(
                        f"[MAIN] Waiting for {len(not_done)} remaining tasks to finish...",
                        flush=True,
                    )
                    concurrent.futures.wait(not_done, timeout=5)

        except KeyboardInterrupt:
            print("\n[MAIN] KeyboardInterrupt in executor, stopping all clients...", flush=True)
            stop_event.set()

        finally:
            # Recover original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        # Print summary
        total_requests = sum(r.get('total', 0) for r in results.values())
        total_success = sum(r.get('success', 0) for r in results.values())

        print("=" * 50)
        print("MULTI-CLIENT TEST SUMMARY:")
        print(f"Total requests sent: {total_requests}")
        print(f"Total successful: {total_success}")
        print(
            f"Overall success rate: {total_success/total_requests*100:.1f}%"
            if total_requests > 0
            else "No requests sent"
        )
        print("=" * 50)

    def run_continuous(self, interval=1.0, max_num=100):
        """Send request every specified interval"""
        print(f"Starting service with {interval}s interval...")
        print(f"Server address: {self.channel._channel.target()}")
        print("=" * 50)

        request_count = 0
        success_count = 0

        while self.running:
            if request_count >= max_num:
                print(f"Reach the max_num! Stop sending requests")
                break
            start_time = time.time()

            request_count += 1
            print(f"Request #{request_count}:")

            if self.send_request():
                success_count += 1

            # Calculate remaining sleep time
            elapsed_time = time.time() - start_time
            sleep_time = max(0, interval - elapsed_time)

            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"\nService stopped")
        print(f"Total requests: {request_count}")
        print(f"Successful requests: {success_count}")
        print(f"Success rate: {success_count/request_count*100:.1f}%")

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping service...")
        self.running = False

    def close(self):
        if hasattr(self, 'channel'):
            self.channel.close()


def run():
    client = VisualPoseClient(SINGLE_CLIENT_ADDRESS)

    interval = float(input("Enter request interval (default 1.0s): ").strip())
    mode = input("Run mode: (1) single client (2) multi clients? [1/2]: ").strip()
    if mode == '2':
        num_clients = input("Enter number of clients (default 4): ").strip()
        num_clients = int(num_clients) if num_clients.isdigit() else 4
        client.run_multi_clients(num_clients=num_clients, interval=interval, run_time=100.0)
    else:
        try:
            client.run_continuous(interval=interval, max_num=50)
        except KeyboardInterrupt:
            print("\nUser interrupted service")
        except Exception as e:
            print(f"Service error: {e}")
        finally:
            client.close()


if __name__ == '__main__':
    run()
