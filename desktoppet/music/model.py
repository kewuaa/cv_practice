# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-02-11 15:15:54
# @Last Modified by:   None
# @Last Modified time: 2022-02-11 18:38:45
from collections import namedtuple
import os
import base64
import asyncio

from hzy.aiofile import aiofile
from aiohttp import ClientSession

SongInfo = namedtuple('SongInfo', ['text', 'id', 'pic', 'pic_url'])


class BaseMusicer(object):
    """basic of all musicer."""

    def __init__(self, js, current_path, name):
        super(BaseMusicer, self).__init__()
        self.sess = ClientSession()
        self.current_path = current_path
        asyncio.create_task(self.load_js(js, name))

    @property
    def session(self):
        if self.sess.closed:
            self.sess = ClientSession()
        return self.sess

    async def load_js(self, js, name):
        if not os.path.exists(
                path := os.path.join(self.current_path, f'{name.split(".")[-1]}.js')):
            async with aiofile.open_async(
                    path, 'w') as f:
                b64content = js.encode()
                content = base64.b64decode(b64content)
                await f.write(content.decode())

    async def _get_song_info(self, song):
        pass

    async def _get_song_url(self, _id):
        pass

    async def reset_cookie(self, cookie: str) -> None:
        async with aiofile.open_async(
                os.path.join(self.current_path, 'cookie.py'), 'w') as f:
            await f.write(f'cookie = "{cookie}"')

    async def close(self):
        await self.session.close()
