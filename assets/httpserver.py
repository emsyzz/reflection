# This script should be loaded inside a python object of Splash.
# To do so, add the following to one of a scene configuration:
#
# "python" : {
#   "type" : "python",
#   "file" : "./httpserver.py"
# }
#
# For this to work,  the configuration file should be in the same
# directory as this script. Otherwise, modify the path accordingly

import splash
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

hostName = "localhost"
hostPort = 10000
reg = re.compile('/(\w*)\??(\w*)?\&?(\w*)?\&?(.*)?')

class SplashServer(BaseHTTPRequestHandler):
    def sendResult(self, result):
        if len(result) != 0:
            strResult = str(result)
            self.send_response(200)
            self.send_header("Content-Length", len(strResult))
            self.send_header("Content-type", "text")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(bytes(strResult, "utf-8"))
        else:
            self.send_response(204)

    def do_GET(self):
        match = reg.findall(self.path)[0]
        command = match[0]

        if match[0] == "getObjectList":
            result = splash.get_object_list()
            self.sendResult(result)
        elif match[0] == "getObjectTypes":
            result = splash.get_object_types()
            self.sendResult(result)
        elif match[0] == "getObjectAttributeDescription":
            obj = match[1]
            attr = match[2]
            result = splash.get_object_attribute_description(obj, attr)
            self.sendResult(result)
        elif match[0] == "getObjectAttribute":
            obj = match[1]
            attr = match[2]
            result = splash.get_object_attribute(obj, attr)
            self.sendResult(result)
        elif match[0] == "getObjectAttributes":
            obj = match[1]
            attr = match[2]
            result = splash.get_object_attributes(obj)
            self.sendResult(result)
        elif match[0] == "rotateAroundObject":
            rotatable_object = match[1]  # Camera
            target_object = match[2]  # RMeshObject
            object_position = splash.get_object_attribute(target_object, "position")

            (rotation_x, rotation_y, rotation_z) = match[3].split('&')

            result = splash.set_object_attribute(
                rotatable_object,
                "rotateAroundPoint",
                [
                    float(rotation_x), float(rotation_y), float(rotation_z),
                    object_position[0], object_position[1], object_position[2],
                ]
            )
            if result:
                self.sendResult("\"OK\"")
            else:
                self.sendResult("null")
        elif match[0] == "rotateAroundPoint":
            rotatable_object = match[1]  # Camera
            (rotation_x, rotation_y, rotation_z, point_x, point_y, point_z) = match[3].split('&')

            result = splash.set_object_attribute(
                rotatable_object,
                "rotateAroundPoint",
                [
                    float(rotation_x), float(rotation_y), float(rotation_z),
                    float(point_x), float(point_y), float(point_z),
                ]
            )
            if result:
                self.sendResult("\"OK\"")
            else:
                self.sendResult("null")
        elif match[0] == "rotateAroundPointFixed":
            rotatable_object = match[1]  # Camera
            (
                rotation_x, rotation_y, rotation_z,
                point_x, point_y, point_z,
                original_eye_x, original_eye_y, original_eye_z,
                original_target_x, original_target_y, original_target_z
            ) = match[3].split('&')

            splash.set_object_attribute(
                rotatable_object,
                "eye",
                [float(original_eye_x), float(original_eye_y), float(original_eye_z)]
            )
            splash.set_object_attribute(
                rotatable_object,
                "target",
                [float(original_target_x), float(original_target_y), float(original_target_z)]
            )

            result = splash.set_object_attribute(
                rotatable_object,
                "rotateAroundPoint",
                [
                    float(rotation_x), float(rotation_y), float(rotation_z),
                    float(point_x), float(point_y), float(point_z),
                ]
            )
            if result:
                self.sendResult("\"OK\"")
            else:
                self.sendResult("null")
        elif match[0] == "getObjectLinks":
            result = splash.get_object_links()
            self.sendResult(result)
        elif match[0] == "setObjectAttribute":
            obj = match[1]
            attr = match[2]
            value = match[3].split('&')
            for i in range(len(value)):
                converted = False
                try:
                    value[i] = int(value[i])
                    converted = True
                except:
                    pass
                
                if not converted:
                    try:
                        value[i] = float(value[i])
                    except:
                        pass

            if splash.set_object_attribute(obj, attr, value):
                self.sendResult("\"OK\"")
            else:
                self.sendResult("null")

def splash_init():
    global splashServer
    print("Starting HTTP server")
    splashServer = HTTPServer((hostName, hostPort), SplashServer)
    splashServer.timeout = 0.1

def splash_loop():
    global splashServer
    splashServer.handle_request()

def splash_stop():
    print("Shutting down HTTP server")
