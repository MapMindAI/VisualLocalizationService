import tritonclient.grpc as grpcclient
import numpy as np
import time
import cv2


class SuperPoint:
    """
    Supports two EasyTensorRT model variants, selected by model_name:

    ONNX  (superpoint_onnx):
        image     : NCHW (1, 1, H, W)  UINT8,  max_batch_size=0
        threshold : FP32 [1, 1]
        outputs   : kpts (-1,-1,2)  scores (-1,-1)  descps (-1,-1,-1) FP32, variable N

    TRT   (superpoint_trt, converted from ONNX by convert_models.sh):
        image     : NCHW (1, 1, H, W)  UINT8,  max_batch_size=0
        threshold : FP32 [1, 1]
        outputs   : kpts (-1,512,2)  scores (-1,512)  descps (-1,512,256) FP32, fixed 512 slots + mask
    """

    def __init__(
        self,
        triton_url,
        model_name="superpoint_onnx",
        model_version="",
        max_image_shape=1080,
        keypoint_thresh=0.015,
    ):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.max_image_shape = max_image_shape
        self.model_version = model_version
        self.model_name = model_name
        # TRT outputs fixed 512 slots + validity mask; ONNX outputs variable N directly
        self.trt_fixed = "trt" in model_name and "onnx" not in model_name

        # both models use threshold shape [1, 1]
        keypoint_thresh_np = np.array([[keypoint_thresh]], dtype=np.float32)
        self.input_thresh = grpcclient.InferInput(
            "keypoint_threshold", keypoint_thresh_np.shape, "FP32"
        )
        self.input_thresh.set_data_from_numpy(keypoint_thresh_np)

        self.desired_outputs = [
            grpcclient.InferRequestedOutput("kpts"),
            grpcclient.InferRequestedOutput("scores"),
            grpcclient.InferRequestedOutput("descps"),
        ]
        if self.trt_fixed:
            self.desired_outputs.append(grpcclient.InferRequestedOutput("mask"))

    def run(self, image_numpy):
        image = (
            cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)
            if image_numpy.ndim == 3
            else image_numpy.copy()
        )
        image_size = (image.shape[1], image.shape[0])
        if image.shape[1] > self.max_image_shape:
            new_height = int(self.max_image_shape * image.shape[0] / image.shape[1])
            image_size = (self.max_image_shape, new_height)
            image = cv2.resize(image, image_size).astype(np.uint8)
        if image.shape[0] > self.max_image_shape:
            new_width = int(self.max_image_shape * image.shape[1] / image.shape[0])
            image_size = (new_width, self.max_image_shape)
            image = cv2.resize(image, image_size).astype(np.uint8)

        # both models use NCHW layout: (1, 1, H, W)
        image_tensor = np.expand_dims(image, axis=(0, 1))
        input_image = grpcclient.InferInput("image", image_tensor.shape, "UINT8")
        input_image.set_data_from_numpy(image_tensor)

        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[input_image, self.input_thresh],
            outputs=self.desired_outputs,
        )

        kpts   = response.as_numpy("kpts")[0]    # (N, 2) or (512, 2)
        scores = response.as_numpy("scores")     # (1, N) or (1, 512)
        descps = response.as_numpy("descps")     # (1, N, 256) or (1, 512, 256)

        if self.trt_fixed:
            # filter out padded slots using the validity mask
            mask  = response.as_numpy("mask")    # (1, 512) bool
            valid = np.where(mask[0])[0]
            kpts   = kpts[valid]
            descps = descps[:, valid, :]
            scores = scores[:, valid]

        factor_x = image_numpy.shape[1] / image_size[0]
        factor_y = image_numpy.shape[0] / image_size[1]
        kpts_scaled = np.array(
            [[factor_x * kpts[i][0], factor_y * kpts[i][1]] for i in range(len(kpts))],
            dtype=np.float32,
        )

        # unified output shapes: (1,N,2), (1,N,256), (1,N)
        kpts_out   = kpts_scaled[np.newaxis]
        descps_out = descps if descps.ndim == 3 else descps[np.newaxis]
        scores_out = scores if scores.ndim == 2 else scores[np.newaxis]
        return kpts_out, descps_out, scores_out
