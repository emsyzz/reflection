import ctypes
import multiprocessing
import os
from multiprocessing import Event, Lock
from multiprocessing.sharedctypes import Value, RawArray
from typing import Optional

import numpy as np
from numpy.ctypeslib import as_array

from src.PRNetProcess import grab

TEXTURE_SIZE = 256


class PRNResult:
    id: int
    face_angle: float
    face_texture: np.ndarray
    face_with_pose: np.ndarray
    face_with_landmarks: np.ndarray

    def __init__(
            self,
            id: int,
            face_angle: float = None,
            face_texture: np.ndarray = None,
            face_with_landmarks: np.ndarray = None,
            face_with_pose: np.ndarray = None
    ) -> None:
        self.id = id
        self.face_angle = face_angle
        self.face_texture = face_texture
        self.face_with_landmarks = face_with_landmarks
        self.face_with_pose = face_with_pose


class ThreadedPRNet:
    # Shared memory
    ### INPUT
    _source_frame: RawArray
    _stopped: Value
    _event: Event
    ### Locks
    write_lock: Lock
    read_lock: Lock
    ### OUTPUT
    _result_id: Value
    _face_angle: Value
    _face_texture: RawArray
    _face_texture_shape: RawArray
    _face_with_landmarks: RawArray
    _face_with_pose: RawArray
    _face_with_shape: RawArray
    # End Shared memory
    is_debug: bool

    _process: multiprocessing.Process
    _write_lock: Lock = Lock()
    _read_lock: Lock

    _original_face_floated: np.ndarray
    _original_face_pos: any
    _original_face_vertices: any

    _prn_pos: any
    _last_prn_result: Optional[PRNResult]

    def start(self) -> 'ThreadedPRNet':
        self.is_debug = os.environ['DEBUG'] == "1"
        self.__result_id = 0

        self.__last_prn_result = None

        # Shared memory
        ### INPUT
        self.__source_frame = RawArray(ctypes.c_uint8, 2000 * 2000 * 3)
        self.__source_frame_shape = RawArray(ctypes.c_int, 3)
        self.__stopped = Value(ctypes.c_bool, False)
        self.__event = Event()
        # LOCKS
        self.__write_lock = Lock()
        self.__read_lock = Lock()
        ### OUTPUT
        self.__result_id = Value(ctypes.c_int, 0)
        self.__face_angle = Value(ctypes.c_float, 0.0)
        self.__face_texture = RawArray(ctypes.c_uint8, 2000 * 2000 * 3)
        self.__face_texture_shape = RawArray(ctypes.c_int, 3)
        if self.is_debug:
            self.__face_with_landmarks = RawArray(ctypes.c_uint8, 2000 * 2000 * 3)
            self.__face_with_pose = RawArray(ctypes.c_uint8, 2000 * 2000 * 3)
            self.__face_with_shape = RawArray(ctypes.c_int, 3)
        else:
            self.__face_with_landmarks = None
            self.__face_with_pose = None
            self.__face_with_shape = None
        # End Shared memory

        self.__process = multiprocessing.Process(target=grab, args=(
            ### INPUT
            self.__source_frame,
            self.__source_frame_shape,
            self.__stopped,
            self.__event,
            ### Locks
            self.__write_lock,
            self.__read_lock,
            ### OUTPUT REQUIRED
            self.__result_id,
            self.__face_angle,
            self.__face_texture,
            self.__face_texture_shape,
            ### OUTPUT OPTIONAL
            self.__face_with_landmarks,
            self.__face_with_pose,
            self.__face_with_shape,
            self.is_debug
        ), daemon=True)
        self.__process.start()

        return self

    def stop(self) -> None:
        self.__stopped.value = True
        self.__process.join()

    def update_source_frame(self, source_frame: np.ndarray) -> None:
        with self.__write_lock:
            self.__source_frame_shape[:] = source_frame.shape
            _ctypes_source_frame = np.ctypeslib.as_ctypes(source_frame.flatten())
            self.__source_frame[:len(_ctypes_source_frame)] = _ctypes_source_frame[:]
        self.__event.set()

    def read(self) -> Optional[PRNResult]:
        with self.__read_lock:
            if self.__result_id.value == 0:
                return None

            if self.__last_prn_result and self.__last_prn_result.id == self.__result_id.value:
                return self.__last_prn_result

            _result_id = self.__result_id.value
            _face_angle = self.__face_angle.value
            _face_texture = as_array(self.__face_texture).copy().astype(np.uint8)
            _face_texture_shape = as_array(self.__face_texture_shape).copy()
            if self.is_debug:
                _face_with_pose = as_array(self.__face_with_pose).copy().astype(np.uint8)
                _face_with_landmarks = as_array(self.__face_with_landmarks).copy().astype(np.uint8)
                _face_with_shape = as_array(self.__face_with_shape).copy()

        prn_result = PRNResult(
            _result_id,
            _face_angle,
            raw_array_with_shape_to_ndarray(_face_texture_shape, _face_texture)
        )

        if self.is_debug:
            prn_result.face_with_pose = raw_array_with_shape_to_ndarray(_face_with_shape, _face_with_pose)
            prn_result.face_with_landmarks = raw_array_with_shape_to_ndarray(_face_with_shape, _face_with_landmarks)

        self.__last_prn_result = prn_result
        return self.__last_prn_result


def raw_array_with_shape_to_ndarray(shape: np.ndarray, raw_array: np.ndarray):
    return raw_array[:shape[0] * shape[1] * shape[2]].reshape(shape)
