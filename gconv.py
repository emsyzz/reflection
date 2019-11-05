#!/usr/bin/env python
import sys
import serial
import time
import math

angle_limit = 1

def convert(angle, limits, mapping, offset=0):
    (minimum, maximum) = limits
    (map_min, map_max) = mapping
    map_max += 1 # allow maximum value
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

def send_gcode(ser, code):
    print("Send: %s" % code)
    ser.write(bytes('%s\n' % code, 'ascii'))
    inp = ser.readline()
    print("Receive: %s" % inp)
    time.sleep(1)


# Open grbl serial port
with serial.Serial(sys.argv[1], 115200) as s:

    # Wake up grbl
    s.write(bytes("\r\n\r\n", 'ascii'))
    time.sleep(2)   # Wait for grbl to initialize
    s.flushInput()  # Flush startup text in serial input

    send_gcode(s, '$X')

    # Stream g-code to grbl
    while True:
        line = sys.stdin.readline()
        if not line:
            continue
        l = float(line.strip()) # Strip all EOL characters for consistency
        print('Input: %s' % l)
        g = convert(l, (-angle_limit, angle_limit), (0, 60), offset=-30)
        print('Gcode: %s' % g)
        send_gcode(s, g)

