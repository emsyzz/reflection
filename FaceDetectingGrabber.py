import threading

import cv2
import dlib
import numpy as np

from DnnFaceDetector import DnnFaceDetector


class FaceDetectingGrabber:

    def __init__(self, source_frame=None):
        self.source_frame = source_frame
        self.frame = None
        self.stopped = False
        self.detected_face_area = None
        self.face_found = False
        self.__fd = DnnFaceDetector()
        self.__predictor = dlib.shape_predictor('face_detection_model/shape_predictor_68_face_landmarks.dat')
        self.__thread = None

    def start(self):
        self.__thread = threading.Thread(target=self.grab, args=())
        self.__thread.start()
        return self

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def grab(self):
        while not self.stopped:
            if self.source_frame is None:
                continue

            self.__fd.detect_face(self.source_frame)

            if self.__fd.is_detected_face:
                detected_face_area = self.__fd.detected_face_area

                face_frame = detected_face_area.get_frame(self.source_frame)

                face = dlib.rectangle(
                    0,
                    0,
                    face_frame.shape[1],
                    face_frame.shape[0]
                )
                landmarks = self.__predictor(face_frame, face)
                self.add_face_landmarks(face_frame, landmarks)

                self.detected_face_area = detected_face_area
                self.face_found = True

                self.frame = face_frame
            else:
                self.detected_face_area = None
                self.face_found = False
                frame = np.zeros((100, 100, 3), np.uint8)

                # Write some Text

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (10, 30)
                font_scale = 1
                font_color = (0, 0, 255)
                line_type = 2

                cv2.putText(frame, 'X',
                            bottom_left_corner_of_text,
                            font,
                            font_scale,
                            font_color,
                            line_type)

                self.frame = frame

    def add_face_landmarks(self, frame, landmarks):
        for n in range(0, landmarks.num_parts):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    def update_source_frame(self, source_frame):
        self.source_frame = source_frame

    def read_frame(self):
        if self.frame is None:
            return None
        return self.frame
