import threading

import cv2

from SearchingFaceAreaProvider import SearchingFaceAreaProvider
from ThreadedImageGrabber import ThreadedImageGrabber


class FaceSearchingAreaGrabber(ThreadedImageGrabber):

    def __init__(self, src=0):
        super().__init__(src)
        self.__sfad = SearchingFaceAreaProvider(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__frame = None
        self.__extracted_frame = None
        self.__write_lock = threading.Lock()
        self.__read_lock = threading.Lock()

    def grab(self):
        color = (0, 255, 0)

        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()

                with self.__write_lock:
                    rect = self.__sfad.face_searching_area
                    cv2.rectangle(frame, rect.get_start_xy(), rect.get_end_xy(), color, 2)
                    extracted_frame = rect.get_frame(frame)

                with self.__read_lock:
                    self.__frame = frame
                    self.__extracted_frame = extracted_frame

    def update_next_searching_frame(self, area):
        with self.__write_lock:
            self.__sfad.update_next_searching_frame(area)

    def update_not_found_face(self):
        with self.__write_lock:
            self.__sfad.update_not_found_face()

    def read(self):
        with self.__read_lock:
            return self.__frame, self.__extracted_frame
