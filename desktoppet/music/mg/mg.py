# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-02-07 00:40:21
# @Last Modified by:   None
# @Last Modified time: 2022-02-12 07:27:58
from urllib.parse import quote
from yarl import URL
import os
import json
import time
import hashlib
import asyncio

from lxml.html import fromstring

from hzy import fake_ua
from hzy.aiofile.aiofile import AsyncFuncWrapper
from model import BaseMusicer
from model import SongInfo
from .crypt import encrypt
from .cookie import cookie


current_path, _ = os.path.split(os.path.realpath(__file__))
ua = fake_ua.UserAgent('internetexplorer')
fromstring = AsyncFuncWrapper(fromstring)


class Musicer(BaseMusicer):
    """docstring for Musicer."""

    SEARCH_URL = 'https://music.migu.cn/v3/search?page=1&type=song&i={i}&f=html&s={s}&c=001002A&keyword={keyword}&v=3.22.4'
    KEY_STR = 'c001002Afhtmlk3915f4a1-4229-4cf9-86a3-08d888cbf524-n41644147336766keyword{keyword}s{s}u{user_agent}/220001v3.22.4'
    SONG_URL = 'https://music.migu.cn/v3/api/music/audioPlayer/getPlayInfo?dataType=2&data={}&secKey=ElP1Za4xkAwkmEnBhaswmP%2FcK91dJQEYRVjJSVvQ9PKXL1CrvdcQVQ2MbjtSfy1JMU8o%2FzkTJY2ypU3NWk%2BXf7aYAv93IdJQAJZKmC%2Fe%2B48V2s52iOeCUcFYc9piXHT%2FMlawqSS4bwaqX%2BucR9J1A3XE21rQSkhjPKLXOAhRESc%3D'
    PERSONAL_KEY = '4ea5c508a6566e76240543f8feb06fd457777be39549c4016436afda65d2330e'
    TO_ENCRYP = '{"copyrightId":"%s","type":1,"auditionsFlag":11}'
    HEADERS = {
        'user-agent': '',
        'cookie': cookie,
        'referer': 'https://music.migu.cn/v3',
    }

    def __init__(self):
        super(Musicer, self).__init__('', current_path, __name__)
        self.encode = lambda string: quote(string).replace('/', '%2F').replace('%28', '(').replace('%29', ')')

    async def load_js(self, js, name):
        return

    async def _get_song_info(self, song):
        keyword = quote(song)
        time_stamp = int(time.time())
        user_agent = self.HEADERS['user-agent'] = ua.get_ua()
        sha1 = hashlib.sha1(self.encode(self.KEY_STR.format(
                                        keyword=keyword,
                                        s=time_stamp,
                                        user_agent=user_agent
                                        )).encode())
        self.HEADERS['referer'] = 'https://music.migu.cn/v3'
        res = await self.session.get(
            self.SEARCH_URL.format(i=sha1.hexdigest(),
                                   s=time_stamp,
                                   keyword=keyword),
            headers=self.HEADERS,
            allow_redirects=False)
        assert (status := res.status) == 200, f'response: {status}'
        text = await res.text()
        tree = await fromstring(text)
        song_list = tree.xpath('//div[@class="songlist-body"]/div')
        return [self._parse_info(song) for song in song_list]

    @staticmethod
    def _parse_info(lxml_tree):
        info = lxml_tree.xpath(
            './div[@class="song-actions single-column"]//@data-share')[0]
        info_dict = json.loads(info)
        return SongInfo(f'咪咕: {info_dict["title"]}-->{info_dict["singer"]}--><{info_dict["album"]}>',
                        (os.path.split(info_dict['linkUrl'])[1], 'mg'),
                        os.path.split(pic_url := info_dict['imgUrl'])[1],
                        ':'.join(['https', pic_url]))

    async def _get_song_url(self, _id):
        to_encryp = self.TO_ENCRYP % _id
        data = encrypt(to_encryp, self.PERSONAL_KEY).decode()
        data = self.encode(data)
        self.HEADERS['user-agent'] = ua.get_ua()
        self.HEADERS['referer'] = 'https://music.migu.cn/v3/music/player/audio'
        res = await self.session.get(URL(self.SONG_URL.format(data), encoded=True),
                                     headers=self.HEADERS,
                                     allow_redirects=False)
        assert (status := res.status) == 200, f'response: {status}'
        result = await res.json(content_type=None)
        assert result['returnCode'] == '000000', result['msg']
        url = result['data']['playUrl']
        return ':'.join(['https', url])
