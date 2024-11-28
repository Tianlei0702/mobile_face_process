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

from face_detection_rknn import FaceDetector
from mark_detection_rknn import MarkDetector
from pose_estimation import PoseEstimator
from face_recognition.face_recognition_rknn import FaceEmbedding
from face_emotion_rknn import FaceEmotion
from utils import refine
import numpy as np

lite_flag = False
if lite_flag:
    from rknnlite.api import RKNNLite
else:
    from rknn.api import RKNN as RKNNLite


# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=20,
                    help="The webcam index.")
parser.add_argument("--facedb", action='store_true',
                    help="whether rebuild facedb")
args = parser.parse_args()


def split_imge(image):
    height, width, _ = image.shape
    half_width = width // 2

    left_image = image[:,:half_width]
    right_image = image[:, half_width:]

    return left_image, right_image


def run():
    # Before estimation started, there are some startup works to do.

    # 双目相机图像采集.
    video_src = args.cam if args.video is None else args.video

    cap = cv2.VideoCapture(video_src)
    if args.video is None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    image_width,image_height = 640,480

    if lite_flag:
        # Setup a face detector to detect human faces.
        face_detector = FaceDetector("assert_rknn/face_detector.rknn", RKNNLite,lite_flag)
        face_reco = FaceEmbedding("assert_rknn/facenet128.rknn","face_recognition/facedb.db", RKNNLite,lite_flag)
        emotion_detector = FaceEmotion("assert_rknn/facial_expression.rknn", RKNNLite,lite_flag)
        # # Setup a mark detector to detect landmarks.
        mark_detector = MarkDetector("assert_rknn/face_landmarks.rknn", RKNNLite,lite_flag)
    else:
        face_detector = FaceDetector("../assets/face_detector.onnx", RKNNLite,lite_flag)
        face_reco = FaceEmbedding("../assets/facenet128.onnx","face_recognition/facedb.db", RKNNLite,lite_flag)
        emotion_detector = FaceEmotion("../assets/facial_expression.onnx", RKNNLite,lite_flag)
        mark_detector = MarkDetector("../assets/face_landmarks.onnx", RKNNLite,lite_flag)


    # # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(image_width, image_height)

    if args.facedb:
        face_reco.build_facedb("face_dataset")

    # Measure the performance with a tick meter.
    tm = cv2.TickMeter()
    frame_count = 0

    # Now, let the frames flow.
    while True:

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        if args.video == None:
            left_image, right_image = split_imge(frame)
        else:
            left_image = frame

        tm.start()
        frame_count += 1

        frame = cv2.resize(left_image, (image_width, image_height))

        # Step 1: Get faces from current frame.
        detected_faces, _ = face_detector.detect(frame, 0.7)
        faces = detected_faces[np.lexsort((detected_faces[:, 1], detected_faces[:, 0]))]


            
        # Any valid face found?
        for i in range(len(faces)):
            # # Step 2: Detect landmarks. Crop and feed the face area into the
            # # mark detector. Note only the first face will be used for
            # # demonstration.
            face = refine(faces, image_width, image_height, 0.15)[i]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            face_name, face_score = face_reco.find(patch,"cosine")

            emotion = emotion_detector.detect(patch)
            emotion_detector.visualize(frame, face[:4], face_name, face_score, emotion)


            # Run the mark detection.
            marks = mark_detector.detect(patch)[0].reshape([68, 2])

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Try pose estimation with 68 points.
            pose = pose_estimator.solve(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))

            # Do you want to see the axes?
            #pose_estimator.draw_axes(frame, pose)

            # Do you want to see the marks?
            mark_detector.visualize(frame, marks, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            face_detector.visualize(frame, faces)
        tm.stop()

        fps = frame_count / tm.getTimeSec()
        print("FPS:", fps, frame_count, tm.getTimeSec())
        # Draw the FPS on screen.
        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    run()
