import cv2

from SearchingFaceAreaProvider import SearchingFaceAreaProvider
from ThreadedImageGrabber import ThreadedImageGrabber


class FaceSearchingAreaGrabber(ThreadedImageGrabber):

    def __init__(self, src=0):
        super().__init__(src)
        self.sfad = SearchingFaceAreaProvider(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH),
                                              self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.extracted_frame = None

    def grab(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()

                rect = self.sfad.face_searching_area,
                color = (0, 255, 0)
                cv2.rectangle(frame, rect[0].get_start_xy(), rect[0].get_end_xy(), color, 2)
                extracted_frame = rect[0].get_frame(frame)

                self.frame = frame
                self.extracted_frame = extracted_frame

    def read_whole_frame(self):
        if self.frame is None:
            return None
        return self.frame

    def read_extracted_frame(self):
        if self.extracted_frame is None:
            return None
        return self.extracted_frame
