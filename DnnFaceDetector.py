import os

import cv2
import numpy as np


class RectCoordinates:
    def __init__(self, x, y, w, h):
        self.startX = x
        self.startY = y
        self.w = w
        self.h = h
        self.endY = y + h
        self.endX = x + w

    def get_frame(self, frame):
        return frame[self.startY:self.endY, self.startX:self.endX]


class DetectedFace:
    def __init__(self, frame, face_area):
        self.frame = frame
        self.face_area = face_area


class DnnFaceDetector:
    def __init__(self, initial_face_searching_area):
        self.min_face_img_width = 30
        self.min_img_confidence = 0.7
        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
        modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        self.lastDetectedFace = None
        self.lastDetectedFaceArea = initial_face_searching_area
        self.isDetectedFace = False
        self.initial_face_searching_area = initial_face_searching_area
        self.face_searching_frame = None

    def detect_face(self, frame):
        # self.face_searching_frame = self.calculate_face_searching_frame(frame)
        self.face_searching_frame = self.lastDetectedFaceArea.get_frame(frame)
        imageBlob = cv2.dnn.blobFromImage(
            self.face_searching_frame, 1.0, (100, 100),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image

        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
        (h, w) = self.face_searching_frame.shape[:2]
        rect = self.__get_biggest_face_coordinates(detections, w, h, frame[0].size, frame[1].size)
        if rect is not None:
            self.lastDetectedFaceArea = self.calc(rect)
            self.lastDetectedFace = rect.get_frame(self.face_searching_frame)
            self.isDetectedFace = True
        else:
            self.lastDetectedFaceArea = self.initial_face_searching_area
            self.isDetectedFace = False

    def calc(self, rect):
        start_x = self.lastDetectedFaceArea.startX + rect.startX - rect.w / 2
        start_y = self.lastDetectedFaceArea.startY + rect.startY - rect.h / 2
        return RectCoordinates(start_x, start_y, rect.w*2, rect.h*2)

    def __get_biggest_face_coordinates(self, detections, w, h, frame_width, frame_heigth):
        biggest_width = self.min_face_img_width
        biggest_rect = None
        for i in range(0, detections.shape[2]):
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > self.min_img_confidence:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                detection_width = endX - startX
                detection_height = endY - startY
                if detection_width > biggest_width and detection_height > detection_width / 2 \
                        and startY > 0 and startX > 0 and endX > 0 and endY > 0 \
                        and startY < frame_heigth and endY < frame_heigth \
                        and startX < frame_width and endX < frame_width:
                    biggest_width = detection_width
                    biggest_rect = RectCoordinates(startX, startY, detection_width, detection_height)
        return biggest_rect
