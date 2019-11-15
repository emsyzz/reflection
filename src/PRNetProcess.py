import ctypes
import time
from multiprocessing import Event, Lock
from multiprocessing.sharedctypes import RawArray, Value
from typing import Optional

import cv2
import numpy as np
import prnet
from prnet.utils import estimate_pose, plot_kpt, plot_pose_box


def grab(
        ### INPUT
        input_source_frame: RawArray,
        source_frame_shape: RawArray,
        input_stopped: Value,
        input_event: Event,
        ### Locks
        write_lock: Lock,
        read_lock: Lock,
        ### OUTPUT REQUIRED
        output_result_id: Value,
        output_face_angle: Value,
        output_face_texture: RawArray,
        output_face_texture_shape: RawArray,
        ### OUTPUT OPTIONAL
        output_face_with_landmarks: Optional[RawArray],
        output_face_with_pose: Optional[RawArray],
        output_face_with_shape: Optional[RawArray],
        # End Shared memory
        is_debug: bool
) -> None:
    prn = prnet.PRN(is_dlib=False)
    start_time = time.time()
    x = 1  # displays the frame rate every 1 second
    counter = 0
    while not input_stopped.value:
        input_event.wait()
        input_event.clear()
        with write_lock:
            __source_frame_shape: np.ndarray = np.ctypeslib.as_array(source_frame_shape).copy()
            __source_frame: np.ndarray = np.ctypeslib.as_array(
                input_source_frame[:__source_frame_shape[0] * __source_frame_shape[1] * __source_frame_shape[2]]
            ).copy()

        source_frame: np.ndarray = __source_frame.reshape(
            __source_frame_shape
        ).astype(np.uint8)

        [h, w, c] = source_frame.shape

        box = np.array(
            [0, source_frame.shape[1] - 1, 0, source_frame.shape[0] - 1])  # cropped with bounding box
        prn_pos = prn.process(source_frame, box)

        if prn_pos is None:
            continue

        # 3D vertices
        vertices = prn.get_vertices(prn_pos)
        save_vertices = vertices.copy()
        save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        # # get landmarks
        kpt = prn.get_landmarks(prn_pos)
        #
        # # estimate pose
        camera_matrix, pose = estimate_pose(vertices)

        face_angle_x, face_angle_y, face_angle_z = pose

        if is_debug:
            # 0 < face_angle_x : Face positioned to the right
            print(f"Face angle: {face_angle_x}")

        try:
            __face_angle = face_angle_x
            __textured_face = extract_texture((source_frame / 255).astype(np.float32), prn_pos)
            __face_texture = np.ctypeslib.as_ctypes(__textured_face.flatten())

            if is_debug:
                __landmarked_face = plot_kpt(source_frame, kpt)
                __face_with_landmarks = np.ctypeslib.as_ctypes(__landmarked_face.flatten())
                __face_with_pose = np.ctypeslib.as_ctypes(plot_pose_box(source_frame.flatten(), camera_matrix, kpt))
        except Exception as err:
            print("PRNET ERROR: " + str(err))
            raise err

        with read_lock:
            output_face_angle.value = __face_angle
            move_ndarray_to_raw_array(output_face_texture, __face_texture)
            output_face_texture_shape[:] = __textured_face.shape
            if is_debug:
                move_ndarray_to_raw_array(output_face_with_landmarks, __face_with_landmarks)
                move_ndarray_to_raw_array(output_face_with_pose, __face_with_pose)
                output_face_with_shape[:] = __landmarked_face.shape
            output_result_id.value += 1

        counter += 1
        if (time.time() - start_time) > x:
            print("PRNET FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()


def extract_texture(new_face: np.ndarray, prn_pos: any):
    new_texture = cv2.remap(new_face, prn_pos[:, :, :2].astype(np.float32), None, interpolation=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    return (new_texture * 255).astype(np.uint8)


def move_ndarray_to_raw_array(dest: RawArray, src: ctypes.Array):
    dest[:len(src)] = src[:]
