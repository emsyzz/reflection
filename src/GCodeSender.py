#!/usr/bin/env python
import threading

import serial

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

            self.__command_received.wait()
            self.__command_received.clear()

    def send_immediate(self, command) -> None:
        with self.__write_lock:
            print("Send: %s" % command)
            self.__serial_device.write(bytes('%s\n' % command, 'ascii'))
            inp = self.__serial_device.readline()
            print("Receive: %s" % inp)

    def send(self, command):
        self.__command = GCodeCommand(command, False)
        self.__command_received.set()
