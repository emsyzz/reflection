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

    def __init__(self, output_serial_device: str, angle_limit: tuple,
                 angle_mapping: tuple, enable_projection_angle: bool):
        self.enable_projection_angle = enable_projection_angle
        self.__output_serial_device = output_serial_device
        self.__face_angle = 0.0
        self.__old_angle = 0.0
        self.__max_velocity = 1.0  # units per second
        self.__acceleration = 1.0  # units per second^2
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
                old_angle = self.__old_angle
                face_angle = self.__face_angle
                print('Input: %s' % face_angle)
                g = convert_angle(-face_angle, self.__angle_limit, self.__angle_mapping,
                                  offset=-100)
                print('Gcode: %s' % g)
                gcode_sender.send_immediate(g)

                if self.enable_projection_angle:
                    start_time = time.time()
                    duration = movement_duration(
                        abs(face_angle - old_angle),
                        self.__max_velocity,
                        0,
                        self.__acceleration
                    )
                    print("Movement duration: %0.3f" % duration)

                    current_time = time.time() - start_time
                    while current_time < duration:
                        current_time = time.time() - start_time
                        angle_at_time = calculate_position(
                            current_time,
                            old_angle,
                            face_angle,
                            self.__max_velocity,
                            0,
                            self.__acceleration
                        )
                        print("Angle at %.3f: %.3f" % (current_time, angle_at_time))
                        face_angles = "&".join(map(str, [str(angle_at_time), 0, 0]))
                        around_point = "&".join(map(str, [0, 0, 0]))
                        splash_sender.send_immediate(
                            f"rotateAroundPointFixed?Camera&null&{face_angles}&{around_point}&{original_eye_str}&{original_target_str}"
                        )
                        time.sleep(0.01)  # 10 ms delay

                if not self.__event.wait(6):
                    self.__face_angle = 0.0
                    self.__old_angle = 0.0
                self.__event.clear()

    def stop(self):
        self.stopped = True
        self.__thread.join()

    def update_angle(self, face_angle: float):
        if abs(face_angle - self.__face_angle) > 0.1:
            self.__old_angle = self.__face_angle
            self.__face_angle = face_angle
            self.__event.set()


def dis_acc(vel_ini, acc, tim_acc):
    return vel_ini * tim_acc + 0.5 * acc * math.pow(tim_acc, 2)


def tim_acc(vel_max, vel_ini, acc):
    return (vel_max - vel_ini) / acc  # time at which max speed is reached


def movement_duration(dis, vel_max, vel_ini, acc):
    return (tim_acc(vel_max, vel_ini, acc) * 2) + (dis * vel_max)


def calculate_position(t, d_ini, d_end, v_max, v_ini, a):
    # calculate traveled distance at given time

    print("Traveling from %.2f to %.2f" % (d_ini, d_end))

    t_acc = tim_acc(v_max, v_ini, a)  # time at which max speed is reached

    d_acc = dis_acc(v_ini, a, t_acc)  # distance at which max speed is reached
    d_dec = d_end - d_acc  # distance at which deceleration starts
    d_crs = d_dec - d_acc  # cruise distance
    #print("Distance at which decelerates - %f" % d_dec)
    #print("Cruise distance - %f" % d_crs)

    t_crs = d_crs * v_max  # time spent in cruise
    t_dec = t_crs + t_acc  # time at witch deceleration starts

    #print("Distance after acceleration - %f" % d_acc)
    #print("Time it takes to accelerate - %f" % t_acc)

    if t < t_acc:
        # we are currently accelerating
        d_cur = dis_acc(v_ini, a, t)
    elif t > t_dec:
        # we are currently decelerating
        d_cur = d_dec + (d_acc - dis_acc(v_max, a, t))
    else:
        # we are at max speed
        d_cur = d_acc + (v_max * (t - t_acc))

    if d_cur < d_end:
        # not at end yet
        d_cur = d_cur
    else:
        # reached end
        d_cur = d_end

    #print("Distance traveled at given time - %f" % d_cur)

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
