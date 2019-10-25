import threading
import time

import cv2
import dlib
import numpy as np

from DnnFaceDetector import DnnFaceDetector


class FaceDetectingGrabber:

    def __init__(self, source_frame=None):
        self.__source_frame = source_frame
        self.__output_frame = None
        self.stopped = False
        self.__detected_face_area = None
        self.__face_found = False

        self.__fd = DnnFaceDetector()
        self.__predictor = dlib.shape_predictor('face_detection_model/shape_predictor_68_face_landmarks.dat')
        self.__thread = None
        self.__write_lock = threading.Lock()
        self.__read_lock = threading.Lock()

    def start(self):
        self.__thread = threading.Thread(target=self.grab, args=())
        self.__thread.start()
        return self

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def grab(self):
        start_time = time.time()
        x = 1  # displays the frame rate every 1 second
        counter = 0
        while not self.stopped:
            with self.__write_lock:
                if self.__source_frame is None:
                    continue
                source_frame = self.__source_frame.copy()

            self.__fd.detect_face(source_frame)

            if self.__fd.is_detected_face:
                detected_face_area = self.__fd.detected_face_area

                face_frame = detected_face_area.get_frame(source_frame)

                face = dlib.rectangle(
                    0,
                    0,
                    face_frame.shape[1],
                    face_frame.shape[0]
                )
                landmarks = self.__predictor(face_frame, face)
                self.__add_face_landmarks(face_frame, landmarks)

                with self.__read_lock:
                    self.__output_frame = face_frame
                    self.__face_found = True
                    self.__detected_face_area = detected_face_area
            else:
                text_frame = np.zeros((100, 100, 3), np.uint8)

                # Write some Text

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (10, 30)
                font_scale = 1
                font_color = (0, 0, 255)
                line_type = 2

                cv2.putText(text_frame, 'X',
                            bottom_left_corner_of_text,
                            font,
                            font_scale,
                            font_color,
                            line_type)

                with self.__read_lock:
                    self.__output_frame = text_frame
                    self.__face_found = False
                    self.__detected_face_area = None

            counter += 1
            if (time.time() - start_time) > x:
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

    def __add_face_landmarks(self, frame, landmarks):
        for n in range(0, landmarks.num_parts):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    def update_source_frame(self, source_frame):
        with self.__write_lock:
            self.__source_frame = source_frame

    def read(self):
        with self.__read_lock:
            return self.__output_frame, self.__face_found, self.__detected_face_area
