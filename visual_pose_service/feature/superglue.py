import tritonclient.grpc as grpcclient
import numpy as np
import time
import cv2


def make_input(name, array, dtype):
    dtype_map = {
        "FP32": np.float32, "INT32": np.int32,
        "UINT8": np.uint8,  "BOOL":  np.bool_,
    }
    if dtype in dtype_map:
        array = array.astype(dtype_map[dtype])
    inp = grpcclient.InferInput(name, array.shape, dtype)
    inp.set_data_from_numpy(array)
    return inp


_MAX_KP = 512


class SuperGlue:
    """
    Both EasyTensorRT model variants share the same interface:
        lightglue_onnx : FP32 desc, fixed 512 slots + mask, threshold [1,1], output 'score'
        lightglue_trt  : same as above (converted from ONNX by convert_models.sh)
    """

    def __init__(self, triton_url, model_name="lightglue_onnx", model_version="1", match_thresh=0.2):
        self.grpc_client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.model_version = model_version
        self.model_name = model_name
        self.match_thresh = match_thresh
        self.max_point_num = _MAX_KP

        self.desired_outputs = [
            grpcclient.InferRequestedOutput("match_indices"),
            grpcclient.InferRequestedOutput("score"),
        ]

    def _pad_to_512(self, arr, n):
        """(1, N, d) → (1, 512, d), zero-padded to 512 slots"""
        trunc = arr[:, :n, :].astype(np.float32)
        pad   = np.zeros((1, _MAX_KP - n, arr.shape[2]), dtype=np.float32)
        return np.concatenate([trunc, pad], axis=1)

    def run(self, kpts0, desc0, img_shape0, kpts1, desc1, img_shape1):
        if kpts0.ndim == 2: kpts0 = kpts0[np.newaxis]
        if kpts1.ndim == 2: kpts1 = kpts1[np.newaxis]
        if desc0.ndim == 2: desc0 = desc0[np.newaxis]
        if desc1.ndim == 2: desc1 = desc1[np.newaxis]

        n0 = min(kpts0.shape[1], _MAX_KP)
        n1 = min(kpts1.shape[1], _MAX_KP)

        kpts0_p = self._pad_to_512(kpts0, n0)
        kpts1_p = self._pad_to_512(kpts1, n1)
        desc0_p = self._pad_to_512(desc0, n0)
        desc1_p = self._pad_to_512(desc1, n1)

        mask0 = np.zeros((1, _MAX_KP, 1), dtype=np.bool_); mask0[0, :n0, 0] = True
        mask1 = np.zeros((1, _MAX_KP, 1), dtype=np.bool_); mask1[0, :n1, 0] = True
        threshold = np.array([[self.match_thresh]], dtype=np.float32)

        inputs = [
            make_input("kpts0",      kpts0_p,    "FP32"),
            make_input("kpts1",      kpts1_p,    "FP32"),
            make_input("desc0",      desc0_p,    "FP32"),
            make_input("desc1",      desc1_p,    "FP32"),
            make_input("mask0",      mask0,      "BOOL"),
            make_input("mask1",      mask1,      "BOOL"),
            make_input("img_shape0", img_shape0, "INT32"),
            make_input("img_shape1", img_shape1, "INT32"),
            make_input("threshold",  threshold,  "FP32"),
        ]

        response = self.grpc_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=self.desired_outputs,
        )

        match_indices = response.as_numpy("match_indices")[0, :n0]
        match_scores  = response.as_numpy("score")[0, :n0]
        # mask out matches pointing into the padding region
        match_indices = np.where(match_indices < n1, match_indices, -1)
        return match_indices, match_scores
