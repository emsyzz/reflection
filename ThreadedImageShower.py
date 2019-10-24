import threading
import cv2


class ThreadedImageShower:

    def __init__(self, windows):
        self.windows = windows
        self.stopped = False
        self.__thread = None

    def start(self):
        self.__thread = threading.Thread(target=self.show, args=())
        self.__thread.start()
        return self

    def show(self):
        while not self.stopped:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

            for window_name, frame in self.windows.items():
                if frame is None:
                    continue

                cv2.imshow(window_name, frame)

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def update_window(self, window, frame):
        self.windows[window] = frame
