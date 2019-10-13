class RectCoordinates:
    def __init__(self, x, y, w, h):
        self.startX = x
        self.startY = y
        self.w = w
        self.h = h
        self.endY = y + h
        self.endX = x + w

    def get_frame(self, frame):
        return frame[self.startY:self.endY, self.startX:self.endX]
