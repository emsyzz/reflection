import threading
import cv2


class ThreadedImageShower:

    __thread: threading.Thread
    stopped: bool = False

    def __init__(self, windows):
        self.windows = windows

    def start(self):
        self.__thread = threading.Thread(target=self.show, args=())
        self.__thread.start()
        return self

    def show(self):
        while not self.stopped:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.stopped = True
                break

            for window_name, frame in self.windows.items():
                if frame is None:
                    continue

                cv2.imshow(window_name, frame)

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()
        self.__thread.join()

    def update_window(self, window, frame):
        self.windows[window] = frame
