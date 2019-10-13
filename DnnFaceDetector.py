import os

import cv2
import numpy as np

from RectCoordinates import RectCoordinates


class DetectedFace:
    def __init__(self, frame, face_area):
        self.frame = frame
        self.face_area = face_area


class DnnFaceDetector:
    def __init__(self):
        self.min_face_img_width = 30
        self.min_img_confidence = 0.7
        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
        modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        self.detected_face_area = None
        self.is_detected_face = False

    def detect_face(self, face_searching_frame):
        imageBlob = cv2.dnn.blobFromImage(
            face_searching_frame, 1.0, (100, 100),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image

        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
        (h, w) = face_searching_frame.shape[:2]
        rect = self.__get_biggest_face_coordinates(detections, w, h)
        if rect is not None:
            self.detected_face_area = rect
            self.is_detected_face = True
        else:
            self.is_detected_face = False

    def __get_biggest_face_coordinates(self, detections, w, h):
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
                        and startY < h and endY < h \
                        and startX < w and endX < w:
                    biggest_width = detection_width
                    biggest_rect = RectCoordinates(startX, startY, detection_width, detection_height)
        return biggest_rect
