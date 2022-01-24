# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-21 18:36:13
# @Last Modified by:   None
# @Last Modified time: 2022-01-23 15:01:31
import os
import json
import asyncio

import requests as req
import fake_ua
from aiohttp import ClientSession


def write(message, path, mode='r', encoding=None):
    with open(path, mode=mode, encoding=encoding) as f:
        f.write(message)


current_path, _ = os.path.split(__file__)
ua = fake_ua.UserAgent()
cookie = '_ntes_nuid=c6b62f4920356fba00bb02b9ab8ffb6e; _ntes_nnid=4fb10a59e820fa78798fdffdfa1dcbcc,1640399690494; hb_MA-BFF5-63705950A31C_source=www.google.com.hk; _iuqxldmzr_=32; NMTID=00OggMHqS2StC6LaUBPh-O98krfkUoAAAF-ex-VNw; WNMCID=jdtbkl.1642743174682.01.0; WEVNSM=1.0.0; WM_TID=uWRfK7Me84ZEFBBBRUJq6C7aA7%2BSeboU; WM_NI=%2FV4N5Jo6ZG3%2Bv6UpcZc6GBokJ3fGI31ctIO85DwBFe8aql29Cgj%2B6C8RTxQtzi4JxuKk1r0MYn8cNmzTsNpcQDbpoHE3FgwRJGll5ZqpwtEcLRGdij7Vuf233QYfsei%2Fc0U%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6ee84c63bf5b49ea6b66ba1968eb7d14f939f9abaf47b81f5e1acb64d8796bc82fc2af0fea7c3b92aed8afcafb842adea96a8b221b6ea9b91e252a1ea89a3ee618aedaaa8cc39f3bdbcaace34f79cb9a3c56ffcbfbf8db13381b9a6ccb549fc8e9c99d480bab7fcabfb7f89b5ff93b3479bb2f884cd67ba8cbdaff369939cae91ee4487ed8a88f65a8a99a48db852b3b0b995c869908e8985ec62f58abb87b139f18f9787f26fabf1add1d837e2a3; JSESSIONID-WYYY=HW13SDPOfx%2FFqo97Onz0zaiUoV351m%2BAJoZlZl4G41KTcn3h2oVK9ZkTEd40r%2FTdh%2FzjuRhxykV2pcCWIi8gjuKmVlc2Mag0ZAV9pgwYh4HQZRYQyD4qS3bP8uM1CMy5Dm7cKvql964nmGPnOkPA4OoB5s4HHYV69a4xzgMh3rWKIe0u%3A1642822349624; playerid=16777386'


class MusicPlayer(object):
    """docstring for MusicPlayer."""

    URL = 'https://music.163.com/weapi/song/enhance/player/url/v1?csrf_token='
    HEADERS = {
        'user-agent': '',
        'cookie': cookie,
        'refer': 'https://music.163.com/',
    }
    ENCSECKEY = 'ddb9e95ecba455a303a46b36f291368947d49531f824f5c4adbea2ff7ce22a2e0615a837d727ced55fdbfa85b3590466a39b85749ee5845d29786a7727fd8f154f953ca755d533fe84aa0f100c767f6dbc8441a5ad35711706cb9cf662018025a4519405aa738af496cd3d01594d62821ed0f39b4af97dee184b26e655dd4737'
    CMD = 'node {path} {request_str}'
    REQUEST_DICT = {
        'ids': '[1293886117]',
        'level': 'standard',
        'encodeType': 'aac',
        'csrf_token': '',
    }   # '{"ids": "[1293886117]","level":"standard","encodeType":"aac","csrf_token":""}'

    def __init__(self):
        super(MusicPlayer, self).__init__()

    def _get_url(self):
        self.HEADERS['user-agent'] = ua.get_ua()
        data = {
            'params': self._get_params(self.REQUEST_DICT),
            'encSecKey': self.ENCSECKEY,
        }
        # print(data)
        res = req.post(self.URL, headers=self.HEADERS, data=data)
        try:
            result_dict = res.json()
        except Exception as e:
            print('error:', e)
            raise e
        else:
            url = result_dict['data'][0]['url']
        return url

    def _download(self, url, path):
        res = req.get(url)
        with open(path, 'wb') as f:
            f.write(res.content)

    def _get_params(self, request_dict: dict) -> str:
        request_str = json.dumps(request_dict)
        request_str = json.dumps(request_str)
        # print(request_str)
        res = os.popen(self.CMD.format(
            path=f'{current_path}/music.js', request_str=request_str))
        result = res.read().strip()
        # print(result := res.read().strip())
        return result


if __name__ == '__main__':
    music_player = MusicPlayer()
    print(music_player._get_url())
