from tkinter import filedialog
from functools import partial
from pathlib import Path
import asyncio

from PIL import ImageTk
from PIL import Image
import numpy as np
import cv2

from .lib import asynctk
from .lib import videocapture
from .ui.base_app import MainApp


async def array2image(image_array: np.ndarray, size: tuple) -> Image.Image:
    """numpy数组转换为特定大小的图片.

    Args:
    image_array:
        numpy数组
    size:
        图片大小

    Returns:
        Image.Image
    """

    img = Image.fromarray(image_array)
    return img.convert('RGBA').resize(size, Image.ANTIALIAS)


async def load_image(image_path: str, size: tuple) -> Image.Image:
    """根据图片路径加载图片, 并调整为特定大小.

    Args:
    image_path:
        图片路径
    size:
        图片大小

    Returns:
        Image.Image
    """

    img = await asyncio.get_running_loop().run_in_executor(None, Image.open, image_path)
    return img.convert('RGBA').resize(size, Image.ANTIALIAS)


def set_image(label, img):
    pyimg = ImageTk.PhotoImage(img)
    label.config(image=pyimg)
    label.img = pyimg


def clear_image(*labels):
    for label in labels:
        if hasattr(label, 'img'):
            del label.img


class App(MainApp):
    """应用程序."""

    def __init__(self) -> None:
        super().__init__()
        self.__root = self.main_toplevel
        self.__root.title('face detector')
        self._run_detector = False
        self._face_detector = self.init_face_detector()

    def init_face_detector(self):
        current_path = Path(__file__).parent
        return cv2.CascadeClassifier(str(current_path / './haarcascade_frontalface_default.xml'))

    def _state_change(self):
        for button in self.open_button, self.close_button:
            state = 'disabled' if str(button['state']) == 'normal' else 'normal'
            button.config(state=state)

    async def _process_frame(self, frame: np.ndarray) -> Image.Image:
        """帧处理.

        Args:
        frame:
            视频帧

        Returns:
            None
        """

        frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self._face_detector.detectMultiScale(gray)
        if len(face):
            x, y, w, h = face[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Person', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        label_size = self.output_label.winfo_width(), self.output_label.winfo_height()
        img = await array2image(frame, label_size)
        return img

    @asynctk.asynctk_wrapper('display')
    async def _display(self, file: str = None) -> None:
        """播放画面."""

        root = self.__root
        root.after(0, self._state_change)
        input_label = self.input_label
        output_label = self.output_label
        if file is not None:
            flip = partial(np.flip, axis=2)
        else:
            flip = partial(np.flip, axis=(1, 2))
        callback = lambda future: root.after(0, set_image, output_label, future.result())
        loop = asyncio.events.get_running_loop()
        try:
            async with videocapture.Camera(file=file) as camera:
                if not camera.isOpened():
                    camera.open()
                while 1:
                    ret, frame = await loop.run_in_executor(None, camera.read)
                    if not ret:
                        break
                    frame = flip(frame)
                    label_size = input_label.winfo_width(), input_label.winfo_height()
                    img = await array2image(frame, label_size)
                    if self._run_detector:
                        future = loop.create_task(self._process_frame(frame))
                        future.add_done_callback(callback)
                    root.after(0, set_image, input_label, img)
                    await asyncio.sleep(0.0333)
        except asyncio.CancelledError:
            return
        finally:
            if locals().get('future') is not None:
                future.remove_done_callback(callback)
                future.cancel()
            root.after(0, clear_image, input_label, output_label)
            root.after(0, self._state_change)

    def switch(self) -> None:
        """开关人脸识别器."""

        self._run_detector = not self._run_detector
        if not self._run_detector:
            self.__root.after(1000, clear_image, self.output_label)

    def open_camera(self) -> None:
        """打开摄像头."""

        self._display()

    def open_from_file(self) -> None:
        """打开视频文件."""

        file_path = filedialog.askopenfilename(title='please choose video file', filetypes=[('video', '*.avi *.mp4')])
        if file_path is None:
            return
        self.do_after_close(self._display, file_path)

    @asynctk.asynctk_wrapper()
    async def do_after_close(self, callback=None, *args) -> None:
        """等待摄像头关闭.

        Args:
        callback:
            关闭后的回调函数

        Returns:
            None
        """

        if not callable(callback):
            raise TypeError('A callable object is required')
        task = self.__root._callback_tasks.pop('display', None)
        if task is None:
            self.__root.after(0, callback, *args)
        if task is not None:
            def fut_callback(fut):
                if fut.cancelled():
                    future.set_result(None)

            future = asynctk._callback_loop.create_future()
            future.add_done_callback(lambda fut: self.__root.after(0, callback, *args))
            task.add_done_callback(fut_callback)
            task.cancel()
            await future

    def close_camera(self):
        """关闭摄像头."""

        task = self.__root._callback_tasks.pop('display', None)
        if task is not None:
            task.cancel()


if __name__ == '__main__':
    app = App()
    app.run()

