#!/usr/bin/env python
import sys
import threading
import time

import serial

angle_limit = 1


def convert(angle, limits, mapping, offset=0):
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


class GCodeCommand:
    command: str
    finished: bool

    def __init__(self, command: str, finished: bool = False) -> None:
        self.command = command
        self.finished = finished


class GCodeSender:
    __serial_device: serial.Serial
    __thread: threading.Thread
    __command_received: threading.Event

    __command: GCodeCommand = None
    __current_command: GCodeCommand = None

    __write_lock: threading.Lock

    stopped: bool = False

    def __init__(self, serial_output: serial.Serial) -> None:
        self.__serial_device = serial_output

    def start(self) -> 'GCodeSender':
        self.__thread = threading.Thread(target=self.__send_commands, args=())
        self.__command_received = threading.Event()
        self.__write_lock = threading.Lock()
        self.__thread.start()
        self.stopped = False
        return self

    def stop(self) -> None:
        self.stopped = True
        self.__thread.join()

    def __send_commands(self) -> None:
        while not self.stopped:
            self.__command_received.clear()
            if self.__command and not self.__command.finished:
                self.__current_command = self.__command
                retries = 15
                while not self.__current_command.finished:
                    try:
                        self.send_immediate(self.__current_command.command)
                        self.__current_command.finished = True
                    except Exception as e:
                        retries -= 1
                        if retries <= 0:
                            print("Retry limit exceeded. breaking")
                            break
                        else:
                            print("exception caught while sending. " + str(e) + ". Retrying")

            self.__command_received.wait(1)

    def send_immediate(self, command) -> None:
        with self.__write_lock:
            print("Send: %s" % command)
            self.__serial_device.write(bytes('%s\n' % command, 'ascii'))
            inp = self.__serial_device.readline()
            print("Receive: %s" % inp)

    def send(self, command):
        self.__command = GCodeCommand(command, False)
        self.__command_received.set()


# Open grbl serial port
with serial.Serial('/dev/pts/9', 115200) as s:
    gcode_sender = GCodeSender(s).start()

    # Wake up grbl
    s.write(bytes("\r\n\r\n", 'ascii'))
    time.sleep(2)  # Wait for grbl to initialize
    s.flushInput()  # Flush startup text in serial input

    gcode_sender.send_immediate('$X')

    # Stream g-code to grbl
    while True:
        line = sys.stdin.readline()
        if not line:
            continue
        line = float(line.strip())  # Strip all EOL characters for consistency
        print('Input: %s' % line)
        g = convert(line, (-angle_limit, angle_limit), (0, 60), offset=-30)
        print('Gcode: %s' % g)
        gcode_sender.send(g)

gcode_sender.stop()
