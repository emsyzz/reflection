import threading
import time
import math

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

    def __init__(self, output_serial_device: str, enable_projection_angle: bool, angle_limit: tuple,
                 angle_mapping: tuple, max_speed: float, acceleration: float, angle_tolerance: float,
                 animation_delay: float):
        self.enable_projection_angle = enable_projection_angle
        self.__output_serial_device = output_serial_device
        self.__face_angle = 0.0
        self.__old_angle = 0.0
        self.__angle_limit = angle_limit
        self.__angle_mapping = angle_mapping
        self.__max_velocity = max_speed  # units per second
        self.__acceleration = acceleration  # units per second^2
        self.__angle_tolerance = angle_tolerance
        self.__animation_delay = animation_delay
        self.__direction = None
        self.__swap_direction = False

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
                old_angle = self.__old_angle
                face_angle = self.__face_angle
                print('TAP: Input: %s' % face_angle)
                g = convert_angle(-face_angle, self.__angle_limit, self.__angle_mapping, offset=-100)

                if self.__swap_direction:
                    # add halt to Gcode if direction is changed
                    print("TAP: Swap movement direction")
                    g = "!%s" % g
                    self.__swap_direction = False

                print('TAP: Gcode: %s' % g)
                gcode_sender.send_immediate(g)

                if self.enable_projection_angle:
                    start_time = time.time()
                    distance = abs(face_angle - old_angle)

                    if distance > 0.0:
                        print("TAP: Moving from %.3f to %.3f" % (old_angle, face_angle))
                        print("TAP: Movement distance: %.3f" % distance)

                        traveled_distance = 0.0
                        while traveled_distance < distance:
                            current_time = time.time() - start_time
                            traveled_distance = calculate_distance(
                                current_time,
                                distance,
                                self.__max_velocity,
                                0,
                                self.__acceleration
                            )

                            print("TAP: Offset at %.3f: %.3f" % (current_time, traveled_distance))

                            if self.__face_angle > self.__old_angle:
                                angle_at_time = self.__old_angle + traveled_distance
                            elif self.__face_angle < self.__old_angle:
                                angle_at_time = self.__old_angle - traveled_distance

                            print("TAP: Angle at %.3f: %.3f" % (current_time, angle_at_time))
                            face_angles = "&".join(map(str, [str(angle_at_time), 0, 0]))
                            around_point = "&".join(map(str, [0, 0, 0]))
                            splash_sender.send_immediate(
                                f"rotateAroundPointFixed?Camera&null&{face_angles}&{around_point}&{original_eye_str}&{original_target_str}"
                            )
                            time.sleep(self.__animation_delay)  # 100 ms delay
                    else:
                        face_angles = "&".join(map(str, [str(face_angle), 0, 0]))
                        around_point = "&".join(map(str, [0, 0, 0]))
                        splash_sender.send_immediate(
                            f"rotateAroundPointFixed?Camera&null&{face_angles}&{around_point}&{original_eye_str}&{original_target_str}"
                        )

                if not self.__event.wait(6):
                    self.__face_angle = 0.0
                    self.__old_angle = 0.0
                self.__event.clear()

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def update_angle(self, face_angle: float):
        distance = abs(face_angle - self.__face_angle)
        if distance > self.__angle_tolerance:
            direction = face_angle > self.__face_angle
            self.__swap_direction = self.__direction is not None and direction != self.__direction
            self.__direction = direction
            self.__old_angle = self.__face_angle
            self.__face_angle = face_angle
            self.__event.set()


def dis_acc(vel_ini, acc, tim_acc):
    return vel_ini * tim_acc + 0.5 * acc * math.pow(tim_acc, 2)


def tim_acc(vel_max, vel_ini, acc):
    return (vel_max - vel_ini) / acc  # time at which max speed is reached


def calculate_distance(t, d_end, v_max, v_ini, a):
    # calculate traveled distance at given time

    t_acc = tim_acc(v_max, v_ini, a)  # time at which max speed is reached
    d_acc = dis_acc(v_ini, a, t_acc)  # distance at which max speed is reached

    # if distance to end is smaller than acceleration distance
    # then return only how far it could have gotten
    if d_acc > d_end:
        d_cur = dis_acc(v_ini, a, t)
    else:
        d_dec = d_end - d_acc  # distance at which deceleration starts
        d_crs = d_dec - d_acc  # cruise distance

        t_crs = d_crs * v_max  # time spent in cruise
        t_dec = t_crs + t_acc  # time at witch deceleration starts

        if t < t_acc:
            # we are currently accelerating
            d_cur = dis_acc(v_ini, a, t)
        elif t > t_dec:
            # we are currently decelerating
            d_cur = d_dec + (d_acc - dis_acc(v_max, a, t))
        else:
            # we are at max speed
            d_cur = d_acc + (v_max * (t - t_acc))

    if d_end > d_cur > 0.0:
        # not at end yet
        d_cur = d_cur
    else:
        # reached end
        d_cur = d_end

    return d_cur


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
