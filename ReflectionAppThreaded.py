import cv2

import RectCoordinates
from CaptureAreaDrawer import CaptureAreaDrawer
from FaceDetectingGrabber import FaceDetectingGrabber, DetectedObject
from SearchingFaceAreaProvider import SearchingFaceAreaProvider
from ThreadedImageGrabber import ThreadedImageGrabber
from ThreadedImageShower import ThreadedImageShower


class ReflectionAppThreaded:
    DETECTED_FACE_WINDOW = "Detected face"
    FACE_SEARCHING_AREA_WINDOW = "Face searching area"

    __stream_width: int
    __stream_height: int
    __sfad: SearchingFaceAreaProvider
    __windows_shower: ThreadedImageShower
    __capture_area_drawer: CaptureAreaDrawer
    __image_grabber: ThreadedImageGrabber
    __face_detecting_grabber: FaceDetectingGrabber

    def __init__(self, source=0):
        self.__stream_width, self.__stream_height = self.get_video_capturer_dimensions(source)
        self.__sfad = SearchingFaceAreaProvider(self.__stream_width, self.__stream_height)

        self.__windows_shower = ThreadedImageShower({
            self.FACE_SEARCHING_AREA_WINDOW: None,
            self.DETECTED_FACE_WINDOW: None
        }).start()
        self.__image_grabber = ThreadedImageGrabber(source).start()
        self.__capture_area_drawer = CaptureAreaDrawer(
            self.__image_grabber.read().copy(),
            self.__sfad.face_searching_area
        ).start()
        self.__face_detecting_grabber = FaceDetectingGrabber(
            self.__image_grabber.read().copy()
        ).start()

    def loop(self):
        camera_frame = self.__image_grabber.read()

        self.__capture_area_drawer.update_source_frame(camera_frame.copy())
        self.__face_detecting_grabber.update_source_frame(camera_frame.copy())

        rectangled_frame: RectCoordinates = self.__capture_area_drawer.read()
        detected_object: DetectedObject = self.__face_detecting_grabber.read()

        self.__windows_shower.update_window(self.FACE_SEARCHING_AREA_WINDOW, rectangled_frame)
        self.__windows_shower.update_window(self.DETECTED_FACE_WINDOW, detected_object.output_frame)

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
        self.__windows_shower.stop()

    def is_stopped(self):
        return self.__image_grabber.stopped \
               or self.__capture_area_drawer.stopped \
               or self.__windows_shower.stopped


ReflectionAppThreaded(0).start()
