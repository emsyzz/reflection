import sys
import cv2


class VideoFrameProcessor:
    def __init__(self, frame):
        self.originalFrame = frame
        self.processedFrame = frame

    def scale(self, scale_percent=50):
        width = int(self.originalFrame.shape[1] * scale_percent / 100)
        height = int(self.originalFrame.shape[0] * scale_percent / 100)
        self.__resize(width, height)

    def resize_to_width(self, wanted_frame_width):
        width = self.originalFrame.shape[1]
        height = self.originalFrame.shape[0]
        wanted_frame_height = int(height * wanted_frame_width / width)

        self.__resize(wanted_frame_height, wanted_frame_width)

    def __resize(self, wanted_frame_height, wanted_frame_width):
        dim = (wanted_frame_width, wanted_frame_height)
        self.dim = dim
        # INTER_LINEAR is faster than INTER_AREA
        self.processedFrame = cv2.resize(self.originalFrame, dim, interpolation=cv2.INTER_LINEAR)
        print('Resized img to width : ', self.processedFrame.shape)

    def add_face_landmarks(self, landmarks):
        for n in range(0, landmarks.num_parts):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(self.processedFrame, (x, y), 2, (255, 0, 0), -1)

    def add_dot_in_meridian(self):
        row = int(self.originalFrame.shape[1] / 2)
        col = int(self.originalFrame.shape[0] / 2)
        cv2.circle(self.processedFrame, (row, col), 5, (0, 255, 0), -1)

    def show_processed_frame(self, frame_name, outline=None):
        try:
            if outline is not None:
                rect, color = outline
                cv2.rectangle(self.originalFrame, rect.get_start_xy(), rect.get_end_xy(), color, 2)
                cv2.imshow(frame_name, self.originalFrame)
            else:
                cv2.imshow(frame_name, self.processedFrame)
        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
