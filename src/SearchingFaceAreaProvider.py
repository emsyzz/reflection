import copy

from src.RectCoordinates import RectCoordinates


class SearchingFaceAreaProvider:
    def __init__(self, full_frame_width, full_frame_height):
        self.full_frame_width = full_frame_width
        self.full_frame_height = full_frame_height
        self.initial_area_scale = 1.5
        self.initial_face_searching_area = self.__initial_face_searching_area()
        self.face_searching_area = copy.copy(self.initial_face_searching_area)
        self.not_found_faces_in_a_row = 0
        self.max_unfound_faces_before_area_reset = 15

    def __initial_face_searching_area(self):
        h = int(self.full_frame_height / self.initial_area_scale)
        y = int((self.full_frame_height - h) / 2)
        w = int(self.full_frame_width / self.initial_area_scale)
        x = int((self.full_frame_width - w) / 2)
        return RectCoordinates(x, y, w, h)

    def face_searching_frame(self, frame):
        return self.face_searching_area.get_frame(frame)

    def update_next_searching_frame(self, detected_face_area):
        self.face_searching_area = self.__calc_next_searching_area(detected_face_area)
        self.not_found_faces_in_a_row = 0

    def update_not_found_face(self):
        self.not_found_faces_in_a_row = self.not_found_faces_in_a_row + 1
        if self.not_found_faces_in_a_row > self.max_unfound_faces_before_area_reset:
            self.face_searching_area = copy.copy(self.initial_face_searching_area)
            self.not_found_faces_in_a_row = 0

    PADDING_PERCENTAGE_X = .3
    PADDING_PERCENTAGE_Y = .3

    def __calc_next_searching_area(self, rect: RectCoordinates):
        start_origin_x = rect.startX
        start_origin_y = rect.startY
        end_origin_x = rect.endX
        end_origin_y = rect.endY

        padding_x = (rect.w * self.PADDING_PERCENTAGE_X)
        padding_y = (rect.h * self.PADDING_PERCENTAGE_Y)

        start_x = start_origin_x - padding_x
        start_y = start_origin_y - padding_y

        end_x = end_origin_x + padding_x
        end_y = end_origin_y + padding_y

        if start_y < 0:
            start_y = 0
        if start_x < 0:
            start_x = 0
        if end_x > self.full_frame_width:
            end_x = self.full_frame_width
        if end_y > self.full_frame_height:
            end_y = self.full_frame_height

        return RectCoordinates(start_x, start_y, end_x - start_x, end_y - start_y)
