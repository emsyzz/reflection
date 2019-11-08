import threading

import cv2
import numpy as np


class ThreadedImageGrabber:
    stopped: bool = False
    __thread: threading.Thread
    __frame: np.ndarray

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.__frame) = self.stream.read()

    def start(self) -> 'ThreadedImageGrabber':
        self.__thread = threading.Thread(target=self.grab, args=())
        self.__thread.start()
        return self

    def grab(self) -> None:
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()
                self.__frame = frame

    def stop(self) -> None:
        self.stopped = True
        self.__thread.join()

    def read(self) -> np.ndarray:
        return self.__frame
