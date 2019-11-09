from typing import Optional

from src.RectCoordinates import RectCoordinates


class DetectedFace:
    is_face_detected: bool
    detected_face_area: RectCoordinates

    def __init__(self, is_face_detected: bool, detected_face_area: Optional[RectCoordinates]):
        self.is_face_detected = is_face_detected
        self.detected_face_area = detected_face_area
