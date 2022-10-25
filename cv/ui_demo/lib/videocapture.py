# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: videocapture.py
#
#                       Author: kewuaa
#                      Created: 2022-06-22 14:38:29
#                last modified: 2022-06-27 17:03:09
#******************************************************************#
import asyncio
import threading

import cv2


class Camera:
    """摄像头."""

    __lock = threading.Lock()
    __instance = None

    def __new__(cls, capture=None, file: str = None):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, capture=None, file: str = None) -> None:
        """初始化.

        Args:
        capture:
            选择摄像头

        Returns:
            None
        """

        self._capture = capture or 0
        self._file = file
        self._c = None

    def __enter__(self):
        self._c = cv2.VideoCapture(self._file or self._capture + cv2.CAP_DSHOW)
        return self._c

    def __exit__(self, exc_type, exc_val, traceback):
        self._c.release()

    async def __aenter__(self):
        self._c = await asyncio.get_running_loop().run_in_executor(None, cv2.VideoCapture, self._file or self._capture + cv2.CAP_DSHOW)
        return self._c

    async def __aexit__(self, exc_type, exc_val, traceback):
        await asyncio.get_running_loop().run_in_executor(None, self._c.release)

