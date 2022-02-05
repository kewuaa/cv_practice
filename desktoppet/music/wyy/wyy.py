# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-02-04 13:30:14
# @Last Modified by:   None
# @Last Modified time: 2022-02-05 14:29:53
from collections import namedtuple
import os
import asyncio
import base64

from hzy import fake_ua
from hzy.aiofile import aiofile

try:
    from .js_code import wyy_js
    from .cookie import cookie
except ImportError:
    from js_code import wyy_js
    from cookie import cookie


current_path, _ = os.path.split(__file__)
ua = fake_ua.UserAgent()
SongInfo = namedtuple('SongInfo', ['text', 'id', 'pic', 'pic_url'])


class Musicer(object):
    """docstring for Musicer."""

    URL = 'https://music.163.com/weapi/song/enhance/player/url/v1?csrf_token='
    SEARCH_URL = 'https://music.163.com/weapi/cloudsearch/get/web?csrf_token='
    HEADERS = {
        'user-agent': '',
        'cookie': cookie,
        'refer': 'https://music.163.com/',
    }
    ENCSECKEY = 'ddb9e95ecba455a303a46b36f291368947d49531f824f5c4adbea2ff7ce22a2e0615a837d727ced55fdbfa85b3590466a39b85749ee5845d29786a7727fd8f154f953ca755d533fe84aa0f100c767f6dbc8441a5ad35711706cb9cf662018025a4519405aa738af496cd3d01594d62821ed0f39b4af97dee184b26e655dd4737'
    JS_FILE_PATH = os.path.join(current_path, 'wyy.js')
    CMD = "node {path} {request_str}"
    INFO_REQUEST_STR = [r'{\"hlpretag\":\"<span class="s-fc7">\",\"hlposttag\":\"</span>\",\"s\":\"',
                        '',
                        r'\",\"type\":\"1\",\"offset\":\"0\",\"total\":\"true\",\"limit\":\"30\",\"csrf_token\":\"\"}']
    URL_REQUEST_STR = [r'{\"ids\":\"[',
                       '',
                       r']\",\"level\":\"standard\",\"encodeType\":\"aac\",\"csrf_token\":\"\"}']
    # '{"ids": "[1293886117]","level":"standard","encodeType":"aac","csrf_token":""}'

    def __init__(self):
        super(Musicer, self).__init__()
        asyncio.create_task(self.load_js())

    async def load_js(self):
        if not os.path.exists(self.JS_FILE_PATH):
            async with aiofile.open_async(self.JS_FILE_PATH, 'w') as f:
                b64content = wyy_js.encode()
                content = base64.b64decode(b64content)
                await f.write(content.decode())

    async def _get_song_info(self, session, song):
        self.HEADERS['user-agent'] = ua.get_ua()
        self.INFO_REQUEST_STR[1] = song
        request_str = ''.join(self.INFO_REQUEST_STR)
        data = {
            'params': await self._get_params(request_str),
            'encSecKey': self.ENCSECKEY,
        }
        res = await session.post(self.SEARCH_URL, headers=self.HEADERS, data=data)
        try:
            result_dict = await res.json(content_type=None)
        except Exception as e:
            print('error:', e)
            raise e
        else:
            assert (songs := result_dict.get('result')) is not None, '出现未知错误'
            songs = songs['songs']
        return [SongInfo(
            f"网易云: {song['name']}-->{song['ar'][0]['name']}-->《{song['al']['name']}》",
            (str(song['id']), 'wyy'),
            os.path.split(pic_url := song['al']['picUrl'])[1],
            pic_url)
            for song in songs]

    async def _get_song_url(self, session, _id):
        self.HEADERS['user-agent'] = ua.get_ua()
        self.URL_REQUEST_STR[1] = str(_id)
        request_str = ''.join(self.URL_REQUEST_STR)
        data = {
            'params': await self._get_params(request_str),
            'encSecKey': self.ENCSECKEY,
        }
        res = await session.post(self.URL, headers=self.HEADERS, data=data)
        try:
            result_dict = await res.json(content_type=None)
        except Exception as e:
            print('error:', e)
            raise e
        else:
            url = result_dict['data'][0]['url']
            assert url is not None, 'VIP或无版权歌曲，无法播放与下载'
        return url

    async def _get_params(self, request_str: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            self.CMD.format(path=self.JS_FILE_PATH, request_str=request_str),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        assert not stderr, f'subprocess err: {stderr.decode("gbk")}'
        result = stdout.decode('utf-8').strip()
        return result

    @staticmethod
    async def reset_cookie(cookie: str) -> None:
        async with aiofile.open_async(
                os.path.join(current_path, 'cookie.py'), 'w') as f:
            await f.write(f'cookie = "{cookie}"')
