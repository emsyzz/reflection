import threading
import time

import cv2
import numpy as np


class ThreadedImageGrabber:
    stopped: bool = False

    __stream: cv2.VideoCapture
    __thread: threading.Thread
    __grabbed: bool
    __frame: np.ndarray

    def __init__(self, src: any):
        self.__stream = cv2.VideoCapture(src)

        (self.__grabbed, self.__frame) = self.__stream.read()

    def start(self) -> 'ThreadedImageGrabber':
        self.__thread = threading.Thread(target=self.grab, args=(), daemon=True)
        self.__thread.start()
        return self

    def grab(self) -> None:
        start_time = time.time()
        x = 1  # displays the frame rate every 1 second
        counter = 0
        while not self.stopped:
            if not self.__grabbed:
                self.stop()
            else:
                (self.__grabbed, self.__frame) = self.__stream.read()
            counter += 1
            if (time.time() - start_time) > x:
                print("Image grabber FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

    def stop(self) -> None:
        self.stopped = True
        self.__thread.join()

    def read(self) -> np.ndarray:
        return self.__frame
