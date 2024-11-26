"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect and crop the human faces in the video frame.
2. Run facial landmark detection on the face image.ls
3. Estimate the pose by solving a PnP problem.

For more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import time

from face_detection import FaceDetector
# from mark_detection import MarkDetector
# from pose_estimation import PoseEstimator
from face_recognition.face_recognition_rknn import FaceEmbedding
# from face_emotion import FaceEmotion
from utils import refine
import numpy as np


build_facedb = True

def run():
    image = cv2.imread("img4.jpg")
    image_height, image_width = image.shape[0], image.shape[1]

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assert_rknn/face_detector.rknn")
    
    face_reco = FaceEmbedding("assert_rknn/facenet128.rknn","face_recognition/facedb.db")

    if build_facedb:
        face_reco.build_facedb("../face_dataset")


    frame = cv2.resize(image, (640, 480))
    # Step 1: Get faces from current frame.
    detected_faces, _ = face_detector.detect(frame, 0.3)
    faces = detected_faces[np.lexsort((detected_faces[:, 1], detected_faces[:, 0]))]

    for i in range(len(faces)):
        # # Step 2: Detect landmarks. Crop and feed the face area into the
        # # mark detector. Note only the first face will be used for
        # # demonstration.
        face = refine(faces, image_width, image_height, 0.15)[i]
        x1, y1, x2, y2 = face[:4].astype(int)
        patch = frame[y1:y2, x1:x2]

        face_name, face_score = face_reco.find(patch,"cosine")
        print(face_name, face_score)


        #     emotion = emotion_detector.detect([patch])
        #     emotion_detector.visualize(frame, face[:4], face_name, face_score, emotion)
        # else:
        #     emotion = emotion_detector.detect([patch])
        #     emotion_detector.visualize(frame, face[:4], face_name_record[i], face_score_record[i], emotion)
        
        # # cv2.imshow("patch", patch)
        # # if cv2.waitKey(1) == 27:
        # #     break

        # # Run the mark detection.
        # marks = mark_detector.detect([patch])[0].reshape([68, 2])

        # # Convert the locations from local face area to the global image.
        # marks *= (x2 - x1)
        # marks[:, 0] += x1
        # marks[:, 1] += y1

        # # Step 3: Try pose estimation with 68 points.
        # pose = pose_estimator.solve(marks)

        # # All done. The best way to show the result would be drawing the
        # # pose on the frame in realtime.

        # # Do you want to see the pose annotation?
        # pose_estimator.visualize(frame, pose, color=(0, 255, 0))

        # # Do you want to see the axes?
        # pose_estimator.draw_axes(frame, pose)

        # # Do you want to see the marks?
        # mark_detector.visualize(frame, marks, color=(0, 255, 0))

        # Do you want to see the face bounding boxes?
        face_detector.visualize(frame, faces)


    # Show preview.
    cv2.imshow("Preview", frame)
    cv2.waitKey(1000)


if __name__ == '__main__':
    run()
