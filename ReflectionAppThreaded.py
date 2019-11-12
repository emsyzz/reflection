import os
import subprocess
from multiprocessing import Queue
from queue import Empty
from threading import Thread

import cv2
import numpy as np
import prnet

from src import RectCoordinates
from src.CaptureAreaDrawer import CaptureAreaDrawer
from src.FaceDetectingGrabber import FaceDetectingGrabber, DetectedObject
from src.SearchingFaceAreaProvider import SearchingFaceAreaProvider
from src.ThreadedImageGrabber import ThreadedImageGrabber
from src.ThreadedImageShower import ThreadedImageShower
from src.ThreadedPRNet import ThreadedPRNet, PRNResult
from src.detectors import FaceDetector
from src.detectors.DnnFaceDetector import DnnFaceDetector


class ReflectionAppThreaded:
    DETECTED_FACE_WINDOW = "Detected face"
    FACE_SEARCHING_AREA_WINDOW = "Face searching area"
    PRNET_WINDOW = "PRNET_WINDOW"

    __stream_width: int
    __stream_height: int
    __sfad: SearchingFaceAreaProvider
    __windows_shower: ThreadedImageShower
    __capture_area_drawer: CaptureAreaDrawer
    __image_grabber: ThreadedImageGrabber
    __face_detecting_grabber: FaceDetectingGrabber
    __threaded_prnet: ThreadedPRNet
    __prn: prnet.PRN

    def __init__(self, source_device="/dev/video0", output_device="/dev/video1"):
        self.__prn = prnet.PRN(is_dlib=False)
        self.__stream_width, self.__stream_height = self.get_video_capturer_dimensions(source_device)
        self.__sfad = SearchingFaceAreaProvider(self.__stream_width, self.__stream_height)

        if os.environ['DEBUG'] == "1":
            self.__windows_shower = ThreadedImageShower({
                self.FACE_SEARCHING_AREA_WINDOW: None,
                self.DETECTED_FACE_WINDOW: None,
                self.PRNET_WINDOW: None
            }).start()
        self.__image_grabber = ThreadedImageGrabber(source_device).start()
        self.__capture_area_drawer = CaptureAreaDrawer(
            self.__image_grabber.read().copy(),
            self.__sfad.face_searching_area
        ).start()

        face_detector: FaceDetector = DnnFaceDetector(True)

        self.__face_detecting_grabber = FaceDetectingGrabber(
            face_detector,
            self.__image_grabber.read()
        )

        self.__threaded_prnet = ThreadedPRNet(
            self.__prn
        )

        self.__face_detecting_grabber.start(self.__image_grabber.read())
        self.__threaded_prnet.start()

        self.__stream_output = subprocess.Popen(
            [f'ffmpeg -i - -vcodec rawvideo -pix_fmt bgr24 -threads 0 -f v4l2 {output_device}'],
            stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True,
            stderr=subprocess.STDOUT, bufsize=1)

        self.q = Queue()
        t = Thread(target=self.enqueue_output, args=(self.__stream_output.stdout, self.q))
        t.daemon = True  # thread dies with the program
        t.start()

    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def loop(self):
        camera_frame = self.__image_grabber.read()

        self.__capture_area_drawer.update_source_frame(camera_frame.copy())
        self.__face_detecting_grabber.update_source_frame(camera_frame)

        rectangled_frame: RectCoordinates = self.__capture_area_drawer.read()
        detected_object: DetectedObject = self.__face_detecting_grabber.read()
        prn_result: PRNResult = self.__threaded_prnet.read()

        if detected_object.detected_face.is_face_detected:
            self.__sfad.update_next_searching_frame(detected_object.detected_face.detected_face_area)
            self.__threaded_prnet.update_source_frame(self.__face_detecting_grabber.read().output_frame.copy())
        else:
            self.__sfad.update_not_found_face()

        self.__capture_area_drawer.update_rectangle(self.__sfad.face_searching_area)

        if os.environ['DEBUG'] == "1":
            self.__windows_shower.update_window(self.FACE_SEARCHING_AREA_WINDOW, rectangled_frame)

        # self.__write_to_v4l2_loopback(cv2.resize(rectangled_frame, (640, 480), interpolation=cv2.INTER_AREA))

        if prn_result is not None:
            self.__write_to_v4l2_loopback(
                cv2.cvtColor(self.__scale_cropped_face_image(prn_result.face_texture, 992, 992),
                             cv2.COLOR_BGR2GRAY))
            if os.environ['DEBUG'] == "1":
                self.__windows_shower.update_window(
                    "face_texture",
                    self.__scale_cropped_face_image(prn_result.face_texture.copy(), self.CROPPED_FACE_BASE_HEIGHT,
                                                    self.CROPPED_FACE_BASE_WIDTH))
                self.__windows_shower.update_window(
                    "face_with_pose",
                    self.__scale_cropped_face_image(prn_result.face_with_pose, self.CROPPED_FACE_BASE_HEIGHT,
                                                    self.CROPPED_FACE_BASE_WIDTH))
                self.__windows_shower.update_window(
                    "face_with_landmarks",
                    self.__scale_cropped_face_image(prn_result.face_with_landmarks, self.CROPPED_FACE_BASE_HEIGHT,
                                                    self.CROPPED_FACE_BASE_WIDTH))

        if os.environ['DEBUG'] == "1":
            self.__windows_shower.update_window(self.DETECTED_FACE_WINDOW,
                                            self.__scale_cropped_face_image(detected_object.output_frame))

    def __write_to_v4l2_loopback(self, source_frame: np.ndarray):
        try:
            line = self.q.get_nowait()  # or q.get(timeout=.1)
            print(line)
        except Empty:
            pass

        encoded_image = cv2.imencode('.jpg', source_frame)[1].tobytes()
        self.__stream_output.stdin.write(encoded_image)

    CROPPED_FACE_BASE_HEIGHT = 480
    CROPPED_FACE_BASE_WIDTH = 480

    def __scale_cropped_face_image(self, source_frame: np.ndarray, height=CROPPED_FACE_BASE_HEIGHT,
                                   width=CROPPED_FACE_BASE_WIDTH):
        original_height, original_width, colors = source_frame.shape
        if original_height > original_width:
            height_percent = (height / float(original_height))
            new_width_size = int(float(original_width) * float(height_percent))

            width_diff = (width - new_width_size) / 2

            rounding_fix_fill = np.zeros((width, int(not width_diff.is_integer()), colors), np.uint8)
            width_diff = int(width_diff)
            width_fill = np.zeros((width, width_diff, colors), np.uint8)

            new_image = cv2.resize(
                source_frame,
                (new_width_size, height),
                interpolation=cv2.INTER_AREA
            )
            new_image = np.concatenate((width_fill, new_image, width_fill, rounding_fix_fill), axis=1)
        else:
            width_percent = (width / float(original_width))
            new_height_size = int((float(original_height) * float(width_percent)))

            height_diff = (height - new_height_size) / 2

            rounding_fix_fill = np.zeros((int(not height_diff.is_integer()), height, colors), np.uint8)
            height_diff = int(height_diff)
            height_fill = np.zeros((height_diff, height, colors), np.uint8)

            new_image = cv2.resize(
                source_frame,
                (width, new_height_size),
                interpolation=cv2.INTER_AREA
            )
            new_image = np.concatenate((height_fill, new_image, height_fill, rounding_fix_fill), axis=0)

        return new_image

    def start(self):
        while True:
            if self.is_stopped():
                self.stop()
                break

            self.loop()

    def get_video_capturer_dimensions(self, source):
        # Init input video stuff
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            print("Video not opened. Exiting.")
            exit()
        stream_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        stream_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        del video_capture

        return stream_width, stream_height

    def stop(self):
        self.__image_grabber.stop()
        self.__capture_area_drawer.stop()
        if os.environ['DEBUG'] == "1":
            self.__windows_shower.stop()

    def is_stopped(self):
        return self.__image_grabber.stopped \
               or self.__capture_area_drawer.stopped \
               or (os.environ['DEBUG'] == "1" and self.__windows_shower.stopped)


os.environ['DEBUG'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ReflectionAppThreaded("/dev/video0", "/dev/video1").start()
