import dlib
import numpy as np
import prnet

from src.DetectedFace import DetectedFace
from src.RectCoordinates import RectCoordinates
from src.detectors.FaceDetector import AbstractFaceDetector


class PrnFaceDetector(AbstractFaceDetector):

    def __init__(self, prn: prnet.PRN):
        self.__prn = prn

    def detect_face(self, face_searching_frame: np.ndarray) -> 'DetectedFace':
        detected_faces = self.__prn.dlib_detect(face_searching_frame)
        if len(detected_faces) == 0:
            return DetectedFace(False, None)

        detected_face_rect: dlib.rectangle = detected_faces[0].rect

        rect_map = RectCoordinates(
            detected_face_rect.left(),
            detected_face_rect.top(),
            detected_face_rect.width(),
            detected_face_rect.height()
        )

        return DetectedFace(True, rect_map)
