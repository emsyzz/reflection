import cv2
from imutils.video import FPS

from DnnFaceDetector import DnnFaceDetector, RectCoordinates
from VideoFrameProcessor import VideoFrameProcessor


class ReflectionApp:
    def __init__(self, videoCapture):
        self.__video_capture = videoCapture

    def showProcessedVideo(self, maxWidth):
        fps = FPS().start()

        # Check if camera opened successfully
        if (self.__video_capture.isOpened() == False):
            print("Error opening video stream or file")

        self.width = self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        img_fps = self.__video_capture.get(cv2.CAP_PROP_FPS)

        starting_face_search_area = self.__face_searching_area()
        self.__fd = DnnFaceDetector(starting_face_search_area)

        # Read until video is completed
        while (self.__video_capture.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.__video_capture.read()
            if ret == True:

                self.__fd.detect_face(frame)

                vfp = VideoFrameProcessor(self.__fd.face_searching_frame)
                # vfp.show_processed_frame("face_searching_area")
                # self.__fd.detectFace(frame)

                # cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 3)
                # self.__debug_full_img(frame)

                if self.__fd.isDetectedFace:
                    vfp = VideoFrameProcessor(self.__fd.lastDetectedFace)
                    vfp.show_processed_frame("Detected face")

                # update the FPS counter
                fps.update()

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # When everything done, release the video capture object
        self.__video_capture.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def __face_searching_area(self):
        h = int(self.height / 3)
        y = int(self.height / 3)
        w = int(self.width / 3)
        x = int(self.width / 3)
        return RectCoordinates(x, y, w, h)

    def __debug_full_img(self, frame, wanted_frame_width=800):
        vfp = VideoFrameProcessor(frame)
        vfp.resize_to_width(wanted_frame_width)
        vfp.show_processed_frame("Main frame")


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_source = cv2.VideoCapture('testvideo/couple.mp4')

c1 = ReflectionApp(video_source)
c1.showProcessedVideo(1024)
