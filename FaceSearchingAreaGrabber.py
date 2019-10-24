import cv2

from SearchingFaceAreaProvider import SearchingFaceAreaProvider
from ThreadedImageGrabber import ThreadedImageGrabber


class FaceSearchingAreaGrabber(ThreadedImageGrabber):

    def __init__(self, src=0):
        super().__init__(src)
        self.sfad = SearchingFaceAreaProvider(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH),
                                              self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def grab(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

                rect = self.sfad.face_searching_area,
                color = (0, 255, 0)
                cv2.rectangle(self.frame, rect[0].get_start_xy(), rect[0].get_end_xy(), color, 2)
