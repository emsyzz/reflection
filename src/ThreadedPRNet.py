import ctypes
import itertools
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

    def _init_(
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
        self._result_id = 0

        self._last_prn_result = None

        # Shared memory
        ### INPUT
        self._source_frame = RawArray(ctypes.c_int, 2000 * 2000 * 3)
        self._source_frame_shape = RawArray(ctypes.c_int, 3)
        self._stopped = Value(ctypes.c_bool, False)
        self._event = Event()
        # LOCKS
        self._write_lock = Lock()
        self._read_lock = Lock()
        ### OUTPUT
        self._result_id = Value(ctypes.c_int, 0)
        self._face_angle = Value(ctypes.c_float, 0.0)
        self._face_texture = RawArray(ctypes.c_int, 2000 * 2000 * 3)
        self._face_texture_shape = RawArray(ctypes.c_int, 3)
        if self.is_debug:
            self._face_with_landmarks = RawArray(ctypes.c_int, 2000 * 2000 * 3)
            self._face_with_pose = RawArray(ctypes.c_int, 2000 * 2000 * 3)
            self._face_with_shape = RawArray(ctypes.c_int, 3)
        else:
            self._face_with_landmarks = None
            self._face_with_pose = None
            self._face_with_shape = None
        # End Shared memory

        self._process = multiprocessing.Process(target=grab, args=(
            ### INPUT
            self._source_frame,
            self._source_frame_shape,
            self._stopped,
            self._event,
            ### Locks
            self._write_lock,
            self._read_lock,
            ### OUTPUT REQUIRED
            self._result_id,
            self._face_angle,
            self._face_texture,
            self._face_texture_shape,
            ### OUTPUT OPTIONAL
            self._face_with_landmarks,
            self._face_with_pose,
            self._face_with_shape,
            self.is_debug
        ))
        self._process.start()

        return self

    def stop(self) -> None:
        self._stopped.value = True
        self._process.join()

    def update_source_frame(self, source_frame: np.ndarray) -> None:
        with self._write_lock:
            self._source_frame_shape[:] = source_frame.shape
            _ctypes_source_frame = np.ctypeslib.as_ctypes(source_frame.reshape(-1))
            self._source_frame[:len(_ctypes_source_frame)] = _ctypes_source_frame[:]
        self._event.set()

    def read(self) -> Optional[PRNResult]:
        with self._read_lock:
            if self._last_prn_result is None:
                return None

            if self._last_prn_result.id != self._result_id.value:
                _result_id = self._result_id.value
                _face_angle = self._face_angle.value
                _face_texture = as_array(self._face_texture.value).copy()
                _face_texture_shape = as_array(self._face_texture_shape.value).copy()
                if self.is_debug:
                    _face_with_pose = as_array(self._face_with_pose.value).copy()
                    _face_with_landmarks = as_array(self._face_with_landmarks.value).copy()
                    _face_with_shape = as_array(self._face_with_shape.value).copy()

        prn_result = PRNResult(
            _result_id,
            _face_angle,
            _face_texture.reshape(_face_texture_shape)
        )

        if self.is_debug:
            prn_result.face_with_pose = _face_with_pose.reshape(_face_with_shape)
            prn_result.face_with_landmarks = _face_with_landmarks.reshape(_face_with_shape)

        self._last_prn_result = prn_result
        return self._last_prn_result
