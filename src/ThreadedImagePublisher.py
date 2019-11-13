import subprocess
import threading
import time
from multiprocessing import Queue
from queue import Empty

import cv2


class ThreadedImagePublisher:
    __thread: threading.Thread
    stopped: bool = False
    __event: threading.Event

    def __init__(self, frame, output_device):
        self.__frame = frame
        self.__stream_output = subprocess.Popen(
            [f'ffmpeg -i - -vcodec rawvideo -pix_fmt bgr24 -threads 0 -f v4l2 {output_device}'],
            stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True,
            stderr=subprocess.STDOUT, bufsize=1)

    def start(self):
        self.q = Queue()
        t = threading.Thread(target=self.enqueue_output, args=(self.__stream_output.stdout, self.q))
        t.daemon = True  # thread dies with the program
        t.start()
        self.__event = threading.Event()

        self.__thread = threading.Thread(target=self.publish, args=(), daemon=True)
        self.__thread.start()
        return self

    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def publish(self):
        start_time = time.time()
        x = 1  # displays the frame rate every 1 second
        counter = 0
        while not self.stopped:
            try:
                line = self.q.get_nowait()  # or q.get(timeout=.1)
                print(line)
            except Empty:
                pass

            encoded_image = cv2.imencode('.bmp', self.__frame)[1].tobytes()
            self.__stream_output.stdin.write(encoded_image)

            counter += 1
            if (time.time() - start_time) > x:
                print("Image publisher FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

            self.__event.wait()
            self.__event.clear()

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def update_frame(self, frame):
        self.__frame = frame
        self.__event.set()
