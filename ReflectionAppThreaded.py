import cv2

from FaceDetectingGrabber import FaceDetectingGrabber
from FaceSearchingAreaGrabber import FaceSearchingAreaGrabber
from ThreadedImageShower import ThreadedImageShower

DETECTED_FACE_WINDOW = "Detected face"
FACE_SEARCHING_AREA_WINDOW = "Face searching area"


def threadVideoGet(source=0):
    windows_shower = ThreadedImageShower({FACE_SEARCHING_AREA_WINDOW: None, DETECTED_FACE_WINDOW: None}).start()

    image_grabber = FaceSearchingAreaGrabber(source).start()

    processed_image_grabber = None

    image_processor_enabled = True
    if image_processor_enabled:
        processed_image_grabber = FaceDetectingGrabber().start()

    while True:
        if image_processor_enabled:
            processed_image_grabber_stopped = processed_image_grabber.stopped
        else:
            processed_image_grabber_stopped = False

        if image_grabber.stopped \
                or windows_shower.stopped \
                or processed_image_grabber_stopped:
            image_grabber.stop()
            windows_shower.stop()

            if image_processor_enabled:
                processed_image_grabber.stop()

            break

        camera_frame, cropped_camera_frame = image_grabber.read()
        windows_shower.update_window(FACE_SEARCHING_AREA_WINDOW, camera_frame)

        if image_processor_enabled:
            processed_face_frame, face_found, detected_face_area = processed_image_grabber.read()
            windows_shower.update_window(DETECTED_FACE_WINDOW, processed_face_frame)

            if cropped_camera_frame is not None:
                processed_image_grabber.update_source_frame(cropped_camera_frame.copy())

            if face_found:
                pass
                image_grabber.update_next_searching_frame(detected_face_area)
            else:
                image_grabber.update_not_found_face()

threadVideoGet(0)
cv2.destroyAllWindows()