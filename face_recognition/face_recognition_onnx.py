import os

import cv2
import numpy as np
import onnxruntime
from typing import Tuple
from face_recognition.face_db import FaceDB
 
from tqdm import *


class FaceEmbedding:
    def __init__(self, model_file, face_db_file):
        """Initialize a face embedding.

        Args:
            model_file (str): ONNX model file path.
            face_db_file (str): face data db
        """
        assert os.path.exists(model_file), f"File not found: {model_file}"

        self.facedb = FaceDB(face_db_file)

        self.center_cache = {}
        self.nms_threshold = 0.4
        self.input_size = [224,224]
        self.session = onnxruntime.InferenceSession(
            model_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get model configurations from the model file.
        # What is the input like?
        input_cfg = self.session.get_inputs()[0]
        input_name = input_cfg.name
        input_shape = input_cfg.shape
        #self.input_size = tuple(input_shape[2:4][::-1])

        # How about the outputs?
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names


    def _normalize_input(self,img: np.ndarray, normalization:str = "base") -> np.ndarray:
        """Normalize input image.

        Args:
            img (numpy array): the input image.
            normalization (str, optional): the normalization technique. Defaults to "base",
            for no normalization.

        Returns:
            numpy array: the normalized image.
        """

        # issue 131 declares that some normalization techniques improves the accuracy

        if normalization == "base":
            return img

        img *= 255
        if normalization == "VGGFace":
            # mean subtraction based on VGGFace1 training data
            img[..., 0] -= 93.5940
            img[..., 1] -= 104.7624
            img[..., 2] -= 129.1863

        # img = np.transpose(img, [0, 3, 1, 2])
        # print(img.shape)
        return img


    def _resize_image(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize an image to expected size of a ml model with adding black pixels.
        Args:
            img (np.ndarray): pre-loaded image as numpy array
            target_size (np.ndarray): input shape of ml model
        Returns:
            img (np.ndarray): resized input image
        """
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (
            int(img.shape[1] * factor),
            int(img.shape[0] * factor),
        )
        img = cv2.resize(img, dsize)

        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]

        # Put the base image in the middle of the padded image
        img = np.pad(
            img,
            (
                (diff_0 // 2, diff_0 - diff_0 // 2),
                (diff_1 // 2, diff_1 - diff_1 // 2),
                (0, 0),
            ),
            "constant",
        )

        # double check: if target image is not still the same size with target.
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)

        # make it 4-dimensional how ML models expect
        #img = np.array(img)

    
        #img = image.img_to_array(img)-
  
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)

        if img.max() > 1:
            img = (img.astype(np.float32) / 255.0).astype(np.float32)

        return img

    def _l2_normalize(self, x, axis=None, epsilon = 1e-10):
        """
        Normalize input vector with l2
        Args:
            x (np.ndarray or list): given vector
            axis (int): axis along which to normalize
        Returns:
            np.ndarray: l2 normalized vector
        """
        # Convert inputs to numpy arrays if necessary
        x = np.asarray(x)
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + epsilon)


    def embedding(self,img):
        img = self._resize_image(img, (self.input_size[1], self.input_size[0]))
        img = self._normalize_input(img, normalization="VGGFace")
        face_feature = self.session.run(self.output_names, {self.input_name: img})[0][0].tolist()
        face_feature = self._l2_normalize(face_feature)
        return face_feature


    def load_images_from_folder(self,folder_path):
        """
        load image file from folder

        Args:
        folder_path (str): folder path

        Return:
        list: including all image
        """
        names = []
        images = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".jpg", ".png")):
                    file_path = os.path.join(root, file)
                    try:
                        names.append(file[:-4])
                        image = cv2.imread(file_path)
                        images.append(image)
                    except Exception as e:
                        print(f"load image {file_path} error: {e}")
        return names, images


    def build_facedb(self, data_path):
        """
        load face db from folder

        Args:
            data_path: face data folder path
            facedetector: 
        """
        folder_path = data_path
        load_names, loaded_images = self.load_images_from_folder(folder_path)
        
        self.facedb.remove_all()

        for i in tqdm(range(len(load_names)), "build face db ..."):
            name = load_names[i]
            image = loaded_images[i]
            image = self._resize_image(image, (self.input_size[1], self.input_size[0]))
            image = self._normalize_input(image, normalization="VGGFace")
            face_feature = self.session.run(self.output_names, {self.input_name: image})[0][0].tolist()
            face_feature = self._l2_normalize(face_feature)
            self.facedb.add_face(name, face_feature)

        self.facedb.check_face()


    def find(self,img, distance_metric = 'euclidean'):
        current_img = self.embedding(img)

        db_name, db_img = self.facedb.get_face_feature()
        db_img = np.array(db_img)

        recognition_name, recognition_score = 'unknow' , -1

        if distance_metric == "cosine":
            face_threshold = 0.5
            # 计算向量a的模（长度）
            current_norm = np.sqrt(np.sum(current_img ** 2))
            # 计算矩阵B中每个向量的模（长度）
            db_norm = np.sqrt(np.sum(db_img ** 2, axis = 1))
            # 计算向量a与矩阵B中每个向量的点积
            dot_product = np.dot(db_img, current_img)
            # 计算余弦相似度
            scores = dot_product / (current_norm * db_norm)
            
            sorted_indices = np.argsort(scores)[::-1]
            sorted_name = [db_name[i] for i in sorted_indices]
            sorted_scores = scores[sorted_indices]

            if sorted_scores[0] > face_threshold:
                recognition_name, recognition_score = sorted_name[0],sorted_scores[0]
            

        elif distance_metric == "euclidean":
            face_threshold = 1.0

            scores = np.linalg.norm(db_img - current_img, axis =1)

            sorted_indices = np.argsort(scores)
            sorted_name = [db_name[i] for i in sorted_indices]
            sorted_scores = scores[sorted_indices]


            if sorted_scores[0] < face_threshold:
                recognition_name, recognition_score = sorted_name[0],sorted_scores[0]

        return recognition_name, recognition_score


    def visualize(self, image, location, name, score, box_color=(0, 0, 255), text_color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text = str(name) +f' {score:.2f}'
        location = np.round(location).astype(np.uint16)
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (location[0],location[1]- label_size[1]),(location[2],location[1]+base_line), box_color, cv2.FILLED)
        cv2.putText(image, text, (location[0],location[1]), font, font_scale, text_color, thickness)
    
    # def visualize(self, image, results, box_color=(0, 255, 0), text_color=(0, 0, 0)):
    #     """Visualize the detection results.

    #     Args:
    #         image (np.ndarray): image to draw marks on.
    #         results (np.ndarray): face detection results.
    #         box_color (tuple, optional): color of the face box. Defaults to (0, 255, 0).
    #         text_color (tuple, optional): color of the face marks (5 points). Defaults to (0, 0, 255).
    #     """
    #     for det in results:
    #         bbox = det[0:4].astype(np.int32)
    #         conf = det[-1]
    #         cv2.rectangle(image, (bbox[0], bbox[1]),
    #                       (bbox[2], bbox[3]), box_color)
    #         label = f"face: {conf:.2f}"
    #         label_size, base_line = cv2.getTextSize(
    #             label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #         cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1]),
    #                       (bbox[2], bbox[1] + base_line), box_color, cv2.FILLED)
    #         cv2.putText(image, label, (bbox[0], bbox[1]),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

if __name__=="__main__":
    vgg_face = FaceEmbedding(model_file = "assets/vggface.onnx", face_db_file = "face_recognition/facedb.db")
    vgg_face.build_facedb("face_dataset")
