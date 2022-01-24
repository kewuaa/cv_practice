# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-22 09:54:40
# @Last Modified by:   None
# @Last Modified time: 2022-01-22 12:56:51
import os


current_path, _ = os.path.split(__file__)
cmd = f'node {current_path}/music.js'
res = os.popen(cmd)
print(res.read())
