import threading
import time

import cv2
import numpy as np
import prnet
from prnet.utils import estimate_pose, plot_pose_box

TEXTURE_SIZE = 256


class PRNResult:
    pose: np.ndarray

    def __init__(self, pose: np.ndarray) -> None:
        self.pose = pose


class ThreadedPRNet:
    stopped: bool = False
    __source_frame: np.ndarray

    __prn_result: PRNResult = None

    __thread: threading.Thread
    __write_lock: threading.Lock = threading.Lock()
    __prn: prnet.PRN

    __original_face_floated: np.ndarray
    __original_face_pos: any
    __original_face_vertices: any

    __prn_pos: any

    def __init__(self, prn: prnet.PRN, source_frame: np.ndarray):
        self.__prn = prn
        self.__source_frame = source_frame

        self.__init()

    def start(self, source_frame: np.ndarray) -> 'ThreadedPRNet':
        self.__source_frame = source_frame
        self.__init()

        self.__thread = threading.Thread(target=self.grab, args=(), daemon=True)
        self.__thread.start()

        return self

    def __init(self):
        box = np.array(
            [0, self.__source_frame.shape[1] - 1, 0, self.__source_frame.shape[0] - 1])  # cropped with bounding box
        self.__prn_pos = self.__prn.process(self.__source_frame, box)

    def stop(self) -> None:
        self.stopped = True
        self.__thread.join()

    def grab(self) -> None:
        start_time = time.time()
        x = 1  # displays the frame rate every 1 second
        counter = 0
        while not self.stopped:
            with self.__write_lock:
                if self.__source_frame is None:
                    continue
                source_frame = self.__source_frame.copy()
                [h, w, c] = source_frame.shape

            box = np.array(
                [0, source_frame.shape[1] - 1, 0, source_frame.shape[0] - 1])  # cropped with bounding box
            prn_pos = self.__prn.process(source_frame, box)

            source_frame = source_frame / 255.
            if prn_pos is None:
                continue

            # 3D vertices
            vertices = self.__prn.get_vertices(prn_pos)
            save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

            # # get landmarks
            kpt = self.__prn.get_landmarks(prn_pos)
            #
            # # estimate pose
            camera_matrix, pose = estimate_pose(vertices)

            pose = plot_pose_box((source_frame * 255).astype(np.uint8), camera_matrix, kpt)

            try:
                self.__prn_result = PRNResult(self.__extract_texture(source_frame, prn_pos))
            except Exception as err:
                print("errrrrrror: " + str(err))

            counter += 1
            if (time.time() - start_time) > x:
                print("PRNET FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

    def update_source_frame(self, source_frame) -> None:
        with self.__write_lock:
            self.__source_frame = source_frame

    def read(self) -> PRNResult:
        return self.__prn_result

    def __extract_texture(self, new_face: np.ndarray, prn_pos: any):
        ref_pos = prn_pos.astype(np.uint8)

        new_texture = cv2.remap(new_face, ref_pos[:, :, :2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

        return (new_texture * 256).astype(np.uint8)
