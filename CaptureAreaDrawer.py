import copy
import threading

import cv2
import numpy
import numpy as np

import RectCoordinates


class CaptureAreaDrawer:
    OUTLINE_COLOR = (0, 255, 0)

    stopped: bool = False

    __source_frame: numpy.ndarray
    __frame: numpy.ndarray
    __rect: RectCoordinates

    __thread: threading.Thread
    __event_handler: threading.Event = threading.Event()
    __write_lock: threading.Lock = threading.Lock()

    def __init__(self, source_frame: np.ndarray, rect: RectCoordinates):
        self.__source_frame = source_frame
        self.__rect = rect

    def start(self) -> 'CaptureAreaDrawer':
        self.__frame = self.create_retangled_frame(self.OUTLINE_COLOR, self.__rect, self.__source_frame.copy())
        self.__thread = threading.Thread(target=self.draw, args=())
        self.__thread.start()
        return self

    def draw(self):
        while not self.stopped and self.__event_handler.wait(100):
            with self.__write_lock:
                rectangled_frame = self.__source_frame.copy()
                rect = copy.copy(self.__rect)

            self.__frame = self.create_retangled_frame(self.OUTLINE_COLOR, rect, rectangled_frame)

    def create_retangled_frame(self, color, rect, rectangled_frame):
        cv2.rectangle(rectangled_frame, rect.get_start_xy(), rect.get_end_xy(), color, 2)
        return rectangled_frame

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def update_source_frame(self, source_frame):
        with self.__write_lock:
            self.__source_frame = source_frame
            self.__event_handler.set()

    def update_rectangle(self, rect):
        with self.__write_lock:
            self.__rect = rect
            self.__event_handler.set()

    def read(self):
        return self.__frame
