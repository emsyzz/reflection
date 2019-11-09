from src.DetectedFace import DetectedFace


class AbstractFaceDetector:
    def detect_face(self, face_searching_frame) -> 'DetectedFace':
        pass
