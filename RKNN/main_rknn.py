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
# from face_recognition.face_recognition_onnx import FaceEmbedding
# from face_emotion import FaceEmotion
from utils import refine
import numpy as np

def split_imge(image):
    height, width, _ = image.shape
    half_width = width // 2

    left_image = image[:,:half_width]
    right_image = image[:, half_width:]

    return left_image, right_image



def run():
    # Before estimation started, there are some startup works to do.

    # 双目相机图像采集.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    frame_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    frame_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assert_rknn/face_detector.rknn")

    # # Setup a mark detector to detect landmarks.
    # mark_detector = MarkDetector("assets/face_landmarks.onnx")
    
    # emotion_detector = FaceEmotion("assets/facial_expression.onnx")

    # # Setup a pose estimator to solve pose.
    # pose_estimator = PoseEstimator(frame_width, frame_height)

    # vgg_face = FaceEmbedding("assets/vggface.onnx","face_recognition/facedb.db")
    # if args.facedb:
    #     vgg_face.build_facedb("face_dataset")

    # Measure the performance with a tick meter.
    tm = cv2.TickMeter()
    frame_count = 0

    #人脸数量, 当人脸数量改变的时候，才会进行人脸识别 
    face_num = 0 
    face_recognition_interval = 50 #人脸识别间隔
    face_name_record = []
    face_score_record =[] 
    # Now, let the frames flow.
    while True:

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break
        left_image, right_image = split_imge(frame)


        tm.start()
        frame_count += 1
        frame = cv2.resize(left_image, (640, 480))
        # Step 1: Get faces from current frame.
        detected_faces, _ = face_detector.detect(frame, 0.7)

        faces = detected_faces[np.lexsort((detected_faces[:, 1], detected_faces[:, 0]))]
        if len(faces) == face_num and face_recognition_interval % 50 != 0:
            Recognition_Flag = False
        else:
            print('---------------------------')
            Recognition_Flag = True
            face_name_record = []
            face_score_record =[] 
            face_num = len(faces)
        # Any valid face found?
        for i in range(len(faces)):
            # # Step 2: Detect landmarks. Crop and feed the face area into the
            # # mark detector. Note only the first face will be used for
            # # demonstration.
            # face = refine(faces, frame_width, frame_height, 0.15)[i]
            # x1, y1, x2, y2 = face[:4].astype(int)
            # patch = frame[y1:y2, x1:x2]

            # if Recognition_Flag == True:
            #     face_name, face_score = vgg_face.find(patch,"cosine")
            #     #vgg_face.visualize(frame, face[:4], face_name, face_score)
            #     face_name_record.append(face_name)
            #     face_score_record.append(face_score)

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
