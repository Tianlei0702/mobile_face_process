"""Human facial landmark detector based on Convolutional Neural Network."""
import os

import cv2
import numpy as np


class FaceEmotion:
    """FaceEmotion detector by Convolutional Neural Network"""

    def __init__(self, model_file, RKNNLite,lite_flag):
        """Initialize a emotion detector.

        Args:
            model_file (str): ONNX model path.
        """
        assert os.path.exists(model_file), f"File not found: {model_file}"
        #self._input_size = 48
        self.rknn_lite = RKNNLite()

        if lite_flag:
            ret = self.rknn_lite.load_rknn(model_file)
            ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        else:
            self.rknn_lite.config( target_platform='rk3588')
            #ret = self.rknn_lite.load_onnx(model="~/playmate_robot/light_face_process/assets/facial_expression.onnx")
            ret =  self.rknn_lite.load_onnx(model=model_file)
            ret = self.rknn_lite.build(do_quantization=False)
            ret = self.rknn_lite.export_rknn("facial_expression.rknn")
            ret = self.rknn_lite.init_runtime()
        
        self.input_size = (48, 48)
        #self.input_size = tuple(input_shape[2:4][::-1])

        self.labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def _preprocess(self, bgrs):
        """Preprocess the inputs to meet the model's needs.

        Args:
            bgrs (np.ndarray): a list of input images in BGR format.

        Returns:
            np.: an array
        """
        # gray = []
        # for img in bgrs:
        #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     img_gray = cv2.resize(img_gray, (48, 48))
        #     gray.append(img_gray)

        img_gray = cv2.cvtColor(bgrs, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, self.input_size)
        img_gray = np.expand_dims(img_gray, axis=0)
        img_gray = np.expand_dims(img_gray, axis=0).astype(np.float32)
        img_gray = np.transpose(img_gray, [0, 2, 3, 1])

        return img_gray

    def detect(self, images):
        """Detect facial emotions from an face image.

        Args:
            images: a list of face images.

        Returns:
            marks: the facial marks as a numpy array of shape [Batch, 68*2].
        """
        inputs = self._preprocess(images)
        emotions = self.rknn_lite.inference(inputs=inputs)[0][0]
        #emotions = self.session.run(self.output_names, {self.input_name: inputs})[0][0]
        #emotion = self.labels[np.argmax(emotions)]
        return self.labels[np.argmax(emotions)]
    
    def visualize(self, image, location, name, score, emotion,  box_color=(0, 0, 255), text_color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text = str(name) +f' {score:.2f} ' + str(emotion)
        location = np.round(location).astype(np.uint16)
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (location[0],location[1]- label_size[1]),(location[2],location[1]+base_line), box_color, cv2.FILLED)
        cv2.putText(image, text, (location[0],location[1]), font, font_scale, text_color, thickness)
