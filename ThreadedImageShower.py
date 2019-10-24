import threading
import cv2


class ThreadedImageShower:

    def __init__(self, frame=None, frame_name="Preview"):
        self.frame = frame
        self.frame_name = frame_name
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.stop()
                break

            cv2.imshow(self.frame_name, self.frame)

    def stop(self):
        self.stopped = True
