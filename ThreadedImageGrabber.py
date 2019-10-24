import threading

import cv2


class ThreadedImageGrabber:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.grab, args=()).start()
        return self

    def grab(self):
        while not self.stopped:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.stop()
                break

            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
