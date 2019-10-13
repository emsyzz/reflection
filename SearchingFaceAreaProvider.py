from RectCoordinates import RectCoordinates


class SearchingFaceAreaProvider:
    def __init__(self, full_frame_width, full_frame_height):
        self.full_frame_width = full_frame_width
        self.full_frame_height = full_frame_height
        self.initial_face_searching_area = self.__initial_face_searching_area()
        self.face_searching_area = self.initial_face_searching_area
        self.not_found_faces_in_a_row = 0
        self.max_unfound_faces_before_area_reset = 15

    def __initial_face_searching_area(self):
        h = int(self.full_frame_height / 3)
        y = int(self.full_frame_height / 3)
        w = int(self.full_frame_width / 3)
        x = int(self.full_frame_width / 3)
        return RectCoordinates(x, y, w, h)

    def face_searching_frame(self, frame):
        return self.face_searching_area.get_frame(frame)

    def update_next_searching_frame(self, detected_face_area):
        self.face_searching_area = self.__calc_next_searching_area(detected_face_area)
        self.not_found_faces_in_a_row = 0

    def update_not_found_face(self):
        self.not_found_faces_in_a_row = self.not_found_faces_in_a_row + 1
        if self.not_found_faces_in_a_row > self.max_unfound_faces_before_area_reset:
            self.face_searching_area = self.initial_face_searching_area
            self.not_found_faces_in_a_row = 0

    def __calc_next_searching_area(self, rect):
        start_x = self.face_searching_area.startX + rect.startX - rect.w
        if start_x < 0:
            start_x = 0
        start_y = self.face_searching_area.startY + rect.startY - rect.h / 2
        if start_y < 0:
            start_y = 0
        end_x = rect.w * 3
        if end_x > self.full_frame_width:
            end_x = self.full_frame_width
        end_y = rect.h * 2
        if end_y > self.full_frame_height:
            end_y = self.full_frame_height
        return RectCoordinates(start_x, start_y, end_x, end_y)
