class RectCoordinates:
    def __init__(self, x, y, w, h):
        self.startX = int(x)
        self.startY = int(y)
        self.w = int(w)
        self.h = int(h)
        self.endY = int(y + h)
        self.endX = int(x + w)

    def get_start_xy(self):
        return self.startX, self.startY

    def get_end_xy(self):
        return self.endX, self.endY

    def get_frame(self, frame):
        return frame[self.startY:self.endY, self.startX:self.endX]
