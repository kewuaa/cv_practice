# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-02-05 20:27:59
# @Last Modified by:   None
# @Last Modified time: 2022-02-05 20:33:28
import os
import time


cmd = 'pip install -r requirements.txt'
res = os.popen(cmd)
print(res.read())
time.sleep(3)
os.remove(__file__)
