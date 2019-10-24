import threading

import cv2


class ThreadedImageGrabber:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.__thread = None

    def start(self):
        self.__thread = threading.Thread(target=self.grab, args=())
        self.__thread.start()
        return self

    def grab(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        self.__thread.join()
