from FaceSearchingAreaGrabber import FaceSearchingAreaGrabber
from ThreadedImageShower import ThreadedImageShower


def threadVideoGet(source=0):
    image_getter = FaceSearchingAreaGrabber(source).start()
    image_shower = ThreadedImageShower(image_getter.frame, "Face searching area").start()

    while True:
        if image_getter.stopped or image_shower.stopped:
            image_getter.stop()
            image_shower.stop()
            break

        image_shower.frame = image_getter.frame


threadVideoGet(0)
