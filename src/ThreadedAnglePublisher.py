import threading
import time

import serial

from src.GCodeSender import GCodeSender
import requests

from src.SplashSender import SplashSender


class ThreadedAnglePublisher:
    __thread: threading.Thread
    stopped: bool = False
    __event: threading.Event

    original_splash_eye: list
    original_splash_target: list

    def __init__(self, output_serial_device: str, angle_limit: tuple,
                 angle_mapping: tuple, enable_projection_angle: bool):
        self.enable_projection_angle = enable_projection_angle
        self.__output_serial_device = output_serial_device
        self.__face_angle = 0.0
        self.__angle_limit = angle_limit
        self.__angle_mapping = angle_mapping

    def start(self):
        self.__event = threading.Event()

        self.__thread = threading.Thread(target=self.publish, args=(), daemon=True)
        self.__thread.start()
        return self

    def publish(self):
        if self.enable_projection_angle:
            splash_sender = SplashSender()
            original_eye = splash_sender.send_immediate("getObjectAttribute?Camera&eye")
            original_target = splash_sender.send_immediate("getObjectAttribute?Camera&target")

            original_eye_str = "&".join(map(str, original_eye))
            original_target_str = "&".join(map(str, original_target))

        # Open grbl serial port
        with serial.Serial(self.__output_serial_device, 115200) as s:
            gcode_sender = GCodeSender(s).start()

            # Wake up grbl
            s.write(bytes("\r\n\r\n", 'ascii'))
            time.sleep(2)  # Wait for grbl to initialize
            s.flushInput()  # Flush startup text in serial input

            gcode_sender.send_immediate('$X')
            gcode_sender.send_immediate('$H')
            gcode_sender.send_immediate('X0')

            # Stream g-code to grbl
            while not self.stopped:
                face_angle = self.__face_angle
                print('Input: %s' % face_angle)
                g = convert_angle(-face_angle, self.__angle_limit, self.__angle_mapping,
                                  offset=-100)
                print('Gcode: %s' % g)
                gcode_sender.send_immediate(g)

                face_angles = "&".join(map(str, [str(face_angle), 0, 0]))
                around_point = "&".join(map(str, [0, 0, 0]))

                if self.enable_projection_angle:
                    splash_sender.send_immediate(
                        f"rotateAroundPointFixed?Camera&null&{face_angles}&{around_point}&{original_eye_str}&{original_target_str}"
                    )

                if not self.__event.wait(6):
                    self.__face_angle = 0
                self.__event.clear()

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def update_angle(self, face_angle: float):
        if abs(face_angle - self.__face_angle) > 0.1:
            self.__face_angle = face_angle
            self.__event.set()


def convert_angle(angle, limits, mapping, offset=0):
    (minimum, maximum) = limits
    (map_min, map_max) = mapping
    map_max += 1  # allow maximum value
    if angle < minimum:
        angle = minimum
    elif angle > maximum:
        angle = maximum
    out_range = map_max - map_min
    in_range = maximum - minimum
    in_val = angle - minimum
    val = (float(in_val) / in_range) * out_range
    out_val = minimum + val
    out_val += offset
    return 'X%s' % int(out_val)
