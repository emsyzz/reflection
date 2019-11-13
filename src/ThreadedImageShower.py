import threading
import time

import cv2


class ThreadedImageShower:
    __event: threading.Event
    __thread: threading.Thread
    stopped: bool = False

    def __init__(self, windows):
        self.windows = windows

    def start(self):
        self.__event = threading.Event()
        self.__thread = threading.Thread(target=self.show, args=(), daemon=True)
        self.__thread.start()
        return self

    def show(self):
        while not self.stopped:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.stopped = True
                break

            if not self.__event.isSet():
                continue
            self.__event.clear()

            for window_name in list(self.windows):
                frame = self.windows[window_name]
                if frame is None:
                    continue

                cv2.imshow(window_name, frame)

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()
        self.__thread.join()

    def update_window(self, window, frame):
        self.windows[window] = frame
        self.__event.set()
