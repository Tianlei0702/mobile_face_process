"""Human facial landmark detector based on Convolutional Neural Network."""
import os

import cv2
import numpy as np
#from rknnlite.api import RKNNLite
from rknn.api import RKNN as RKNNLite


class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, model_file,RKNNLite,lite_flag):
        """Initialize a mark detector.

        Args:
            model_file (str): ONNX model path.
        """
        assert os.path.exists(model_file), f"File not found: {model_file}"

        self.rknn_lite = RKNNLite()

        if lite_flag:
            ret = self.rknn_lite.load_rknn(model_file)
            ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        else:
            self.rknn_lite.config( target_platform='rk3588')
            #ret = self.rknn_lite.load_onnx(model="~/playmate_robot/light_face_process/assets/face_landmarks.onnx",inputs=['image_input'], input_size_list = [[1,128,128,3]], outputs= ["dense_1"])
            ret =  self.rknn_lite.load_onnx(model=model_file,inputs=['image_input'], input_size_list = [[1,128,128,3]], outputs= ["dense_1"])
            ret = self.rknn_lite.build(do_quantization=False)
            ret = self.rknn_lite.export_rknn("face_landmarks.rknn")
            ret = self.rknn_lite.init_runtime()

        self.input_size = (128, 128)

    def _preprocess(self, bgrs):
        """Preprocess the inputs to meet the model's needs.

        Args:
            bgrs (np.ndarray): a list of input images in BGR format.

        Returns:
           numpy
        """

        img = cv2.resize(bgrs, (self.input_size[0], self.input_size[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = np.transpose(img, [0, 2, 3, 1]).astype(np.float32)

        return img

    def detect(self, images):
        """Detect facial marks from an face image.

        Args:
            images: a list of face images.

        Returns:
            marks: the facial marks as a numpy array of shape [Batch, 68*2].
        """
        inputs = self._preprocess(images)

        marks = self.rknn_lite.inference(inputs=inputs)

        #marks = self.model.run(["dense_1"], {"image_input": inputs})
        return np.array(marks)

    def visualize(self, image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)
