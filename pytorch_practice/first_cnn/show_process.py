# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2021-12-26 16:50:31
# @Last Modified by:   None
# @Last Modified time: 2021-12-26 19:51:03
import visdom
import numpy as np


class Shower:

    def __init__(self, env, win):
        self.vis = visdom.Visdom(env=env)
        self.win = win
        self.counter = 0

    def __call__(self, Y):
        self.vis.line(X=np.array([self.counter]), Y=Y, win=self.win,
                      update='append' if self.counter else None)
        self.counter += 1

    def clear(self):
        self.counter = 0
