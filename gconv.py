#!/usr/bin/env python
import sys
import time

import serial

from src.GCodeSender import GCodeSender
from src.ThreadedAnglePublisher import convert_angle

angle_limit = 0.6

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
        g = convert_angle(line, (-angle_limit, angle_limit), (-100, 100),
                          offset=0)
        print('Gcode: %s' % g)
        gcode_sender.send(g)

gcode_sender.stop()
