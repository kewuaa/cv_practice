# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: asynctk.py
#
#                       Author: kewuaa
#                      Created: 2022-06-20 15:16:07
#                last modified: 2022-07-02 10:42:56
#******************************************************************#
from functools import partial
from functools import wraps
import asyncio
import threading
import tkinter as tk


# 创建tkinter事件循环
_callback_loop = asyncio.new_event_loop()


def asynctk_wrapper(name: str = None):
    """将协程函数包装为普通函数.

    包装后的函数将把该协程作为任务调度, 并以name为名注册到AsyncTk对象上
    """

    def wrapper(coro_func):
        if not asyncio.iscoroutinefunction(coro_func):
            raise TypeError('A coroutine function is required')
        @wraps(coro_func)
        def new_func(*args, **kwargs):
            future = create_task(coro_func(*args, **kwargs))
            if name is not None:
                AsyncTk().register_task(name, future)
            return future

        new_func.__doc__ = coro_func.__doc__
        return new_func

    return wrapper


def create_task(coro) -> asyncio.futures.Future:
    """创建任务.

    Args:
    coro:
        协程

    Returns:
        future对象
    """

    if not asyncio.iscoroutine(coro):
        raise TypeError('A coroutine object is required')
    future = _callback_loop.create_future()
    def callback():
        asyncio.futures._chain_future(asyncio.tasks.ensure_future(coro), future)

    _callback_loop.call_soon_threadsafe(callback)
    return future


def call_soon(callback, *args, context=None) -> asyncio.events.Handle:
    """注册回调函数, 下一次循环运行.

    Args:
    func:
        需要注册的函数

    Returns:
        handle
    """

    return _callback_loop.call_soon_threadsafe(callback, *args, context=context)


def call_at(when: float, callback, *args, context=None) -> asyncio.events.Handle:
    """注册回调函数, 在指定时间调用.

    Args:
    when:
        指定时间
    callback:
        回调函数

    Returns:
        handle
    """

    _callback = partial(_callback_loop.call_at, when, callback, *args, context=context)
    return _callback_loop.call_soon_threadsafe(_callback)


def call_later(delay: float, callback, *args, context=None) -> asyncio.events.Handle:
    """注册回调函数, delay秒后运行.

    Args:
    delay:
        注册函数运行时间
    func:
        需要注册的函数

    Returns:
        future对象
    """

    _callback = partial(_callback_loop.call_later, delay, callback, *args, context=context)
    return _callback_loop.call_soon_threadsafe(_callback)


def _star_loop(loop: asyncio.base_events.BaseEventLoop, callback=None) -> None:
    """启动事件循环.

    Args:
    loop:
        需要启动的事件循环

    Returns:
        None
    """

    if callback is not None and not callable(callback):
        raise TypeError('The callable object is required')
    try:
        loop.run_forever()
    finally:
        pending_tasks = asyncio.tasks.all_tasks(loop=loop)
        for task in pending_tasks:
            task.cancel()
        loop.run_until_complete(_shutdown(loop))
        loop.close()
        if callback is not None:
            callback()


async def _shutdown(loop: asyncio.base_events.BaseEventLoop) -> None:
    """关闭所有待完成的任务."""

    shutdown_tasks = loop.create_task(loop.shutdown_asyncgens()), loop.create_task(_shutdown_executor(loop))
    for task in shutdown_tasks:
        await task


async def _shutdown_executor(loop: asyncio.base_events.BaseEventLoop) -> None:
    """关闭事件循环执行器."""

    if (executor := loop._default_executor) is None:
        return
    future = loop.create_future()

    def do_shutdown(future: asyncio.futures.Future) -> None:
        try:
            executor.shutdown(wait=True)
            loop.call_soon_threadsafe(future.set_result, None)
        except Exception as exc:
            loop.call_soon_threadsafe(future.set_exception, exc)

    t = threading.Thread(target=do_shutdown, args=(future,))
    t.start()
    try:
        await future
    finally:
        t.join()



class AsyncTk(tk.Tk):
    """异步tkinter."""

    __instance = None
    __lock = threading.Lock()
    __first_init = True

    def __new__(cls, exit_all: bool = False):
        if cls.__instance is None:
            with cls.__lock:
                if cls.__instance is None:
                    cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, exit_all: bool = False) -> None:
        """初始化.

        Args:
        exit_all:
            退出时是否退出全部

        Returns:
            None
        """

        if self.__first_init:
            with self.__lock:
                if self.__first_init:
                    super().__init__()
                    self._exit_all = exit_all
                    # 绑定退出时触发事件
                    self.protocol('WM_DELETE_WINDOW', self.quit_app)
                    self._callback_tasks = {}
                    self.__first_init = False

    def register_task(self, name: str, future_or_task: asyncio.futures.Future) -> None:
        """注册任务."""

        if not asyncio.futures.isfuture(future_or_task):
            raise TypeError('A future or a task object is required')
        self._callback_tasks[name] = future_or_task

    def mainloop(self, n: int = 0) -> None:
        """事件循环."""

        def start_callbackloop_in_another_thread() -> None:
            """开启处理回调的事件循环."""

            asyncio.events.set_event_loop(_callback_loop)
            _star_loop(_callback_loop, callback=partial(self.after, 0, self.destroy))

        t = threading.Thread(target=start_callbackloop_in_another_thread, args=())
        t.start()
        try:
            super().mainloop(n=n)
        finally:
            t.join()

    def quit_app(self) -> None:
        """退出."""

        _callback_loop.call_soon_threadsafe(_callback_loop.stop)


if __name__ == '__main__':
    root = AsyncTk()
    root.mainloop()
