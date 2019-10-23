import cv2
from imutils.video import FPS
import dlib

from DnnFaceDetector import DnnFaceDetector
from SearchingFaceAreaProvider import SearchingFaceAreaProvider
from VideoFrameProcessor import VideoFrameProcessor


class ReflectionApp:
    def __init__(self, videoCapture):
        self.__video_capture = videoCapture
        self.__fd = DnnFaceDetector()
        self.__predictor = dlib.shape_predictor('face_detection_model/shape_predictor_68_face_landmarks.dat')

    def showProcessedVideo(self, max_width):
        fps = FPS().start()

        # Check if camera opened successfully
        if (self.__video_capture.isOpened() == False):
            print("Error opening video stream or file")

        self.width = self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        img_fps = self.__video_capture.get(cv2.CAP_PROP_FPS)

        sfad = SearchingFaceAreaProvider(self.width, self.height)

        # Read until video is completed
        while (self.__video_capture.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.__video_capture.read()
            if ret == True:

                face_searching_frame = sfad.face_searching_frame(frame)
                vfp = VideoFrameProcessor(frame)
                vfp.show_processed_frame("Face searching area", (sfad.face_searching_area, (0, 255, 0)))

                self.__fd.detect_face(face_searching_frame)

                if self.__fd.is_detected_face:
                    detected_face_area = self.__fd.detected_face_area

                    face_frame = detected_face_area.get_frame(face_searching_frame)

                    face = dlib.rectangle(
                        0,
                        0,
                        face_frame.shape[1],
                        face_frame.shape[0]
                    )
                    landmarks = self.__predictor(face_frame, face)

                    vfp = VideoFrameProcessor(face_frame)
                    vfp.add_face_landmarks(landmarks)
                    vfp.show_processed_frame("Detected face")
                    sfad.update_next_searching_frame(detected_face_area)
                else:
                    sfad.update_not_found_face()

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

    def __debug_full_img(self, frame, wanted_frame_width=800):
        vfp = VideoFrameProcessor(frame)
        vfp.resize_to_width(wanted_frame_width)
        vfp.show_processed_frame("Main frame")


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_source = cv2.VideoCapture(0)
# video_source = cv2.VideoCapture('samples/MVI_3617.MP4')
# video_source = cv2.VideoCapture('samples/MVI_3618.MP4')
# video_source = cv2.VideoCapture('samples/MVI_3619.MP4')
# video_source = cv2.VideoCapture('samples/IMG_7870.MOV')
# video_source = cv2.VideoCapture('samples/IMG_7869.MOV')

c1 = ReflectionApp(video_source)
c1.showProcessedVideo(1024)
