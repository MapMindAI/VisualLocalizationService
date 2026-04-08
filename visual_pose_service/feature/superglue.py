import tritonclient.grpc as grpcclient
import numpy as np
import time
import cv2


def make_input(name, array, dtype):
    inp = grpcclient.InferInput(name, array.shape, dtype)
    inp.set_data_from_numpy(array)
    return inp

# TODO(wenhao): Add error handling to prevent direct program exit
class SuperGlue:
    def __init__(self, triton_url, model_name="lightglue", model_version="1", match_thresh=0.2):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.model_version = model_version
        self.model_name = model_name
        self.match_thresh_np = np.array([match_thresh], dtype=np.float32)  # [1]
        self.desired_outputs = [
            grpcclient.InferRequestedOutput("match_indices"),
            grpcclient.InferRequestedOutput("match_scores"),
        ]
        self.max_point_num = 4000

    def run(self, kpts0, desc0, img_shape0, kpts1, desc1, img_shape1):
        inputs = [
            make_input("kpts0", kpts0, "FP32"),
            make_input("kpts1", kpts1, "FP32"),
            make_input("desc0", desc0, "UINT8"),
            make_input("desc1", desc1, "UINT8"),
            make_input("img_shape0", img_shape0, "INT32"),
            make_input("img_shape1", img_shape1, "INT32"),
            make_input("match_threshold", self.match_thresh_np, "FP32"),
        ]

        # Run inference
        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=self.desired_outputs,
        )

        # Get results
        match_indices = response.as_numpy("match_indices")[0]  # shape: (N0,)
        match_scores = response.as_numpy("match_scores")[0]  # shape: (N0,)
        return match_indices, match_scores


def draw_matches(
    img0,
    img1,
    kpts0,
    kpts1,
    match_indices,
    match_scores,
    max_display=5000,
    score_thresh=0.0,
):
    """
    Draw matches between two images.

    Args:
        img0, img1: Input grayscale or color images as NumPy arrays.
        kpts0, kpts1: Keypoints in image0 and image1, shape (N0, 2) and (N1, 2).
        match_indices: Array of shape (N0,), giving the index in kpts1 matched to each keypoint in kpts0, or -1.
        match_scores: Array of shape (N0,), confidence score for each match.
        max_display: Max number of matches to draw.
        score_thresh: Minimum confidence threshold to show a match.
    """

    img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

    h0, w0 = img0_color.shape[:2]
    h1, w1 = img1_color.shape[:2]
    out_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    out_img[:h0, :w0] = img0_color
    out_img[:h1, w0:] = img1_color

    # Shift keypoints in img1 for visualization
    kpts1_shifted = kpts1 + np.array([w0, 0], dtype=np.float32)

    # Gather good matches
    matches = [
        (i, match_indices[i], match_scores[i])
        for i in range(len(match_indices))
        if match_indices[i] >= 0 and match_scores[i] >= score_thresh
    ]
    matches = sorted(matches, key=lambda x: -x[2])[:max_display]

    for i, j, score in matches:
        pt0 = tuple(map(int, kpts0[i]))
        pt1 = tuple(map(int, kpts1_shifted[j]))
        color = (0, int(score * 255), 255 - int(score * 255))
        cv2.circle(out_img, pt0, 5, color, -1)
        cv2.circle(out_img, pt1, 5, color, -1)
        cv2.line(out_img, pt0, pt1, color, 3)

    # Show image
    from matplotlib import pyplot as plt

    plt.figure(figsize=(20, 10))
    plt.imshow(out_img[..., ::-1])
    plt.axis("off")
    plt.title("SuperGlue Matches")
    plt.show()


if __name__ == "__main__":
    from superpoint import SuperPoint

    superpoint = SuperPoint("192.168.19.150:8001")
    superglue = SuperGlue("192.168.19.150:8001")

    img_dir = "/mnt/ml-experiment-data/yeliu/gaussian_splatting/GoPro/NanshaOffice/"
    temp_dir = "images/GS010086_0/"
    image_1 = cv2.imread(img_dir + "test.jpg")
    image_2 = cv2.imread(img_dir + temp_dir + "00121.jpg")

    time_ms_begin = time.time() * 1000
    ktps1, descps1, _ = superpoint.run(image_1)
    ktps2, descps2, _ = superpoint.run(image_2)
    time_ms_end = time.time() * 1000

    img_shape1 = np.array([[image_1.shape[0], image_1.shape[1]]], dtype=np.int32)  # [1, 2]
    img_shape2 = np.array([[image_2.shape[0], image_2.shape[1]]], dtype=np.int32)  # [1, 2]
    match_indices, match_scores = superglue.run(
        ktps1, descps1, img_shape1, ktps2, descps2, img_shape2
    )
    time_ms_end_sg = time.time() * 1000

    print("super point time :", time_ms_end - time_ms_begin, "ms")
    print("super glue time :", time_ms_end_sg - time_ms_end, "ms")

    draw_matches(image_1, image_2, ktps1[0], ktps2[0], match_indices, match_scores)
