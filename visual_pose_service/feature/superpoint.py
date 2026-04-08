import tritonclient.grpc as grpcclient
import numpy as np
import time
import cv2

# TODO(wenhao): Add error handling to prevent direct program exit
class SuperPoint:
    def __init__(
        self,
        triton_url,
        model_name="superpoint_trt",
        model_version="5",
        max_image_shape=1080,
        keypoint_thresh=0.015,
    ):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.max_image_shape = max_image_shape
        self.model_version = model_version
        self.model_name = model_name

        keypoint_thresh_np = np.array([[keypoint_thresh]], dtype=np.float32)
        self.input_thresh = grpcclient.InferInput(
            "keypoint_threshold", keypoint_thresh_np.shape, "FP32"
        )
        self.input_thresh.set_data_from_numpy(keypoint_thresh_np)
        # Define desired outputs
        self.desired_outputs = [
            grpcclient.InferRequestedOutput("kpts"),
            grpcclient.InferRequestedOutput("scores"),
            grpcclient.InferRequestedOutput("descps"),
        ]

    def run(self, image_numpy):
        image = (
            cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)
            if image_numpy.ndim == 3
            else image_numpy.copy()
        )
        image_size = (image.shape[1], image.shape[0])
        # resize the image, if image size too large
        if image.shape[1] > self.max_image_shape:
            new_height = int(self.max_image_shape * image.shape[0] / image.shape[1])
            image_size = (self.max_image_shape, new_height)
            image = cv2.resize(image, image_size).astype(np.uint8)
        if image.shape[0] > self.max_image_shape:
            new_width = int(self.max_image_shape * image.shape[1] / image.shape[0])
            image_size = (new_width, self.max_image_shape)
            image = cv2.resize(image, image_size).astype(np.uint8)

        image = np.expand_dims(image, axis=(0, 3))  # Shape: (1, H, W, 1)
        # Define model inputs
        inputs = []
        input_image = grpcclient.InferInput("image", image.shape, "UINT8")
        input_image.set_data_from_numpy(image)
        inputs.append(input_image)
        inputs.append(self.input_thresh)

        # Run inference
        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=self.desired_outputs,
        )

        # Fetch outputs
        kpts = response.as_numpy("kpts")[0]  # shape: (N, 2)
        scores = response.as_numpy("scores")    # shape: (N,)
        descps = response.as_numpy("descps")  # shape: (N, 256)

        kpts_new = []
        # resize the kpts to original size

        factor_x = image_numpy.shape[1] / image_size[0]
        factor_y = image_numpy.shape[0] / image_size[1]
        for i in range(kpts.shape[0]):
            kpts_new.append([factor_x * kpts[i][0], factor_y * kpts[i][1]])

        return np.array([kpts_new], dtype=np.float32), descps, scores


if __name__ == "__main__":
    superpoint = SuperPoint("192.168.19.150:8001")

    image_1 = cv2.imread("/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice/test2.jpg")

    time_ms_begin = time.time() * 1000
    ktps, descps, _ = superpoint.run(image_1)
    time_ms_end = time.time() * 1000

    print(f"extract {len(ktps[0])} points, used {time_ms_end - time_ms_begin}ms")

    from matplotlib import pyplot as plt

    for i in range(ktps[0].shape[0]):
        pt0 = tuple(map(int, ktps[0][i]))
        cv2.circle(image_1, pt0, 3, (0, 255, 0), -1)

    # image_2 = cv2.imread("/Mobili/Data/2.jpg")
    plt.figure(figsize=(12, 6))
    plt.imshow(image_1[..., ::-1])
    plt.axis("off")
    plt.title("SuperPoint")
    plt.show()
