# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-30 14:25:57
# @Last Modified by:   None
# @Last Modified time: 2022-02-04 07:56:18
import base64
import os


# with open('./pictures.py', 'a') as fp:
#     for i in range(1, 5):
#         fp.write(f'pet_{i} = [\n')
#         for j in range(1, 47):
#             with open(f'./resources/pet_{i}/shime{j}.png', 'rb') as f:
#                 content = f.read()
#                 b64content = base64.b64encode(content)
#                 b64str = b64content.decode()
#             fp.write(f'\t"{b64str}",\n')
#         fp.write('\t]\n')
from pictures import *
print(eval('pet_1'))
