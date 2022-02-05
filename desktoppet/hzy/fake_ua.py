# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-23 08:28:50
# @Last Modified by:   None
# @Last Modified time: 2022-01-23 08:50:35
import random

import requests as req


class UserAgent(object):
    """faker headers."""

    URL = 'http://fake-useragent.herokuapp.com/browsers/0.1.5'

    def __init__(self):
        super(UserAgent, self).__init__()
        self.uas = req.get(self.URL).json()['browsers']

    def get_ua(self):
        uas = self.uas[random.choice(list(self.uas.keys()))]
        return random.choice(uas)


if __name__ == '__main__':
    Ua = UserAgent()
    print(Ua.get_ua())
