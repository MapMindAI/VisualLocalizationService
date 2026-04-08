import threading
import time
import cv2
import numpy as np
import signal
import argparse
import random
from superpoint import SuperPoint
from collections import defaultdict

class LoadTest:
    def __init__(self):
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.stats = defaultdict(list)

    def client_loop(self, client_id, img_bytes, triton_url, interval):
        superpoint = SuperPoint(triton_url)
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        while not self.stop_event.is_set():
            try:
                # time.sleep(random.uniform(0.8 * interval, 1.2 * interval))
                # time.sleep(interval)
                
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                start_time = time.time()
                
                kpts, descps, _ = superpoint.run(img)
                
                latency = time.time() - start_time
                with self.lock:
                    self.stats['latency'].append(latency)
                    self.stats['kpts_count'].append(len(kpts[0]))
                
                print(f"[Client {client_id}] {len(kpts[0])} points in {latency:.4f}s")
                
            except Exception as e:
                print(f"[Client {client_id}] ❌ Error: {str(e)}")

    def run(self, args):
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Preload image
        image = cv2.imread(args.image)
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        # Start client threads
        threads = []
        for i in range(args.clients):
            t = threading.Thread(
                target=self.client_loop,
                args=(i, img_bytes, args.triton, args.interval)
            )
            t.start()
            threads.append(t)

        # Periodic stats
        self.print_stats_loop(print_interval=5)

        for t in threads:
            t.join()

    def signal_handler(self, sig, frame):
        print("\n[Main] 🔴 Stopping all clients...")
        self.stop_event.set()

    def print_stats_loop(self, print_interval):
        def stats_loop():
            while not self.stop_event.is_set():
                time.sleep(print_interval)
                with self.lock:
                    if self.stats['latency']:
                        avg_latency = np.mean(self.stats['latency'])
                        avg_kpts = np.mean(self.stats['kpts_count'])
                        print(f"\n📊 STATS: Avg latency={avg_latency:.3f}s | Avg keypoints={avg_kpts:.0f}\n")
        
        threading.Thread(target=stats_loop, daemon=True).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=50)
    parser.add_argument("--image", type=str, default="/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice/test2.jpg")
    parser.add_argument("--triton", type=str, default="192.168.19.150:8001")
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()

    tester = LoadTest()
    tester.run(args)
    print("✅ Test completed.")