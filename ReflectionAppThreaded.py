import cv2
import numpy as np

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

        if detected_object.detected_face.is_face_detected:
            self.__sfad.update_next_searching_frame(detected_object.detected_face.detected_face_area)
        else:
            self.__sfad.update_not_found_face()

        self.__capture_area_drawer.update_rectangle(self.__sfad.face_searching_area)

        self.__windows_shower.update_window(self.FACE_SEARCHING_AREA_WINDOW, rectangled_frame)
        self.__windows_shower.update_window(self.DETECTED_FACE_WINDOW, self.__scale_cropped_face_image(detected_object))

    CROPPED_FACE_BASE_HEIGHT = 480
    CROPPED_FACE_BASE_WIDTH = 480

    def __scale_cropped_face_image(self, detected_object: DetectedObject):
        original_height, original_width = detected_object.output_frame.shape[:2]
        if original_height > original_width:
            height_percent = (self.CROPPED_FACE_BASE_HEIGHT / float(original_height))
            new_width_size = int(float(original_width) * float(height_percent))

            width_diff = (self.CROPPED_FACE_BASE_WIDTH - new_width_size) / 2

            rounding_fix_fill = np.zeros((self.CROPPED_FACE_BASE_WIDTH, int(not width_diff.is_integer()), 3), np.uint8)
            width_diff = int(width_diff)
            width_fill = np.zeros((self.CROPPED_FACE_BASE_WIDTH, width_diff, 3), np.uint8)

            new_image = cv2.resize(
                detected_object.output_frame,
                (new_width_size, self.CROPPED_FACE_BASE_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            new_image = np.concatenate((width_fill, new_image, width_fill, rounding_fix_fill), axis=1)
        else:
            width_percent = (self.CROPPED_FACE_BASE_WIDTH / float(original_width))
            new_height_size = int((float(original_height) * float(width_percent)))

            height_diff = (self.CROPPED_FACE_BASE_HEIGHT - new_height_size) / 2

            rounding_fix_fill = np.zeros((int(not height_diff.is_integer()), self.CROPPED_FACE_BASE_HEIGHT, 3), np.uint8)
            height_diff = int(height_diff)
            height_fill = np.zeros((height_diff, self.CROPPED_FACE_BASE_HEIGHT, 3), np.uint8)

            new_image = cv2.resize(
                detected_object.output_frame,
                (self.CROPPED_FACE_BASE_WIDTH, new_height_size),
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
        self.__windows_shower.stop()

    def is_stopped(self):
        return self.__image_grabber.stopped \
               or self.__capture_area_drawer.stopped \
               or self.__windows_shower.stopped


ReflectionAppThreaded(0).start()
