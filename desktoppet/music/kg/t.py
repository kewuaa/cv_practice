# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-02-05 14:25:03
# @Last Modified by:   None
# @Last Modified time: 2022-02-05 14:30:18
import base64

with open('js_code.py', 'w') as fp:
    with open('kg.js', 'r') as f:
        content = f.read().encode()
        b64content = base64.b64encode(content)
        b64str = b64content.decode()
    fp.write(f'kg_js = "{b64str}"')