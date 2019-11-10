import threading
import time

import cv2
import numpy as np

from src.detectors import FaceDetector
from src.detectors.DnnFaceDetector import DetectedFace


class DetectedObject:
    detected_face: DetectedFace
    output_frame: np.ndarray

    def __init__(self, face_frame: np.ndarray, detected_face: DetectedFace) -> None:
        self.output_frame = face_frame
        self.detected_face = detected_face


class FaceDetectingGrabber:
    stopped: bool = False
    __source_frame: np.ndarray

    __detected_object: DetectedObject

    __thread: threading.Thread
    __write_lock: threading.Lock = threading.Lock()
    __fd: FaceDetector

    def __init__(self, fd: FaceDetector):
        self.__fd = fd

    def start(self, source_frame: np.ndarray) -> 'FaceDetectingGrabber':
        self.__source_frame = source_frame
        detected_face = DetectedFace(False, None)
        self.__detected_object = self.create_undetected_face_object(detected_face)
        self.__thread = threading.Thread(target=self.grab, args=(), daemon=True)
        self.__thread.start()
        return self

    def stop(self) -> None:
        self.stopped = True
        self.__thread.join()

    def grab(self) -> None:
        start_time = time.time()
        x = 1  # displays the frame rate every 1 second
        counter = 0
        while not self.stopped:
            with self.__write_lock:
                if self.__source_frame is None:
                    continue
                source_frame = self.__source_frame.copy()

            detected_face = self.__fd.detect_face(source_frame)

            if detected_face.is_face_detected:
                detected_face_area = detected_face.detected_face_area
                face_frame = detected_face_area.get_frame(source_frame)
                self.__detected_object = DetectedObject(face_frame, detected_face)
            else:
                if self.__detected_object.detected_face.is_face_detected:
                    self.__detected_object = self.create_undetected_face_object(detected_face)

            counter += 1
            if (time.time() - start_time) > x:
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

    def create_undetected_face_object(self, detected_face) -> DetectedObject:
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
        return DetectedObject(text_frame, detected_face)

    def __add_face_landmarks(self, frame, landmarks) -> None:
        for n in range(0, landmarks.num_parts):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    def update_source_frame(self, source_frame) -> None:
        with self.__write_lock:
            self.__source_frame = source_frame

    def read(self) -> DetectedObject:
        return self.__detected_object
