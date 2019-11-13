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

    __frame_id: int

    def __init__(self, src: any, rotation: any, height: int, width: int):
        self.__stream = cv2.VideoCapture(src)
        self.__rotation = rotation
        self.__height = height
        self.__width = width

        self.calc_frame()

    def calc_frame(self):
        (self.__grabbed, frame) = self.__stream.read()
        if self.__height is not None:
            frame = cv2.resize(frame, (self.__height, self.__width))
        if self.__rotation is not None:
            frame = cv2.rotate(frame, self.__rotation)
        self.__frame = frame

    def start(self) -> 'ThreadedImageGrabber':
        self.__frame_id = 0
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
                self.calc_frame()
                self.__frame_id += 1
            counter += 1
            if (time.time() - start_time) > x:
                print("Image grabber FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

    def stop(self) -> None:
        self.stopped = True
        self.__thread.join()

    def read_frame_id(self) -> int:
        return self.__frame_id

    def read(self) -> np.ndarray:
        return self.__frame
