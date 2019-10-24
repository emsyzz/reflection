from FaceDetectingGrabber import FaceDetectingGrabber
from FaceSearchingAreaGrabber import FaceSearchingAreaGrabber
from ThreadedImageShower import ThreadedImageShower


def threadVideoGet(source=0):
    windows_shower = ThreadedImageShower({"Face searching area": None, "Detected face": None}).start()

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

        windows_shower.update_window("Face searching area", image_grabber.read_whole_frame())

        if image_processor_enabled:
            windows_shower.update_window("Detected face", processed_image_grabber.read_frame())
            if image_grabber.read_extracted_frame() is not None:
                processed_image_grabber.update_source_frame(image_grabber.read_extracted_frame().copy())


threadVideoGet(0)
