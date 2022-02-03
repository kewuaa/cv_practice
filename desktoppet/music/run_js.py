# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-22 09:54:40
# @Last Modified by:   None
# @Last Modified time: 2022-01-26 08:21:52
import os
import json
import asyncio


URL_REQUEST_DICT = {
    'ids': '[1293886117]',
    'level': 'standard',
    'encodeType': 'aac',
    'csrf_token': '',
}
# string = json.dumps(URL_REQUEST_DICT)
# string = json.dumps(string)
string = r'{\"ids\":\"[1293886117]\",\"level\":\"standard\",\"encodeType\":\"aac\",\"csrf_token\":\"\"}'
current_path, _ = os.path.split(__file__)
cmd = f'node {current_path}/music.js {string}'

print(string)
print('===========================================================')
print(cmd)
print('===========================================================')
# async def test():
#     proc = await asyncio.create_subprocess_shell(
#         cmd,
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE)
#     stdout, stderr = await proc.communicate()
#     print(stderr)
#     print(stdout.decode('utf-8'))
res = os.popen(cmd)
print(res.read())
print('==================================')
# asyncio.run(test())
