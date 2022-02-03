# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-30 14:25:57
# @Last Modified by:   None
# @Last Modified time: 2022-02-03 20:14:46
import base64
import os


def transform(file_to_trans: str, save_path: str) -> None:
    with open(save_path, 'w') as fp:
        pass
    with open(save_path, 'a') as fp:
        for img in os.listdir(file_to_trans):
            name, _ = os.path.splitext(img)
            with open(os.path.join(file_to_trans, img), 'rb') as f:
                b64str = base64.b64encode(f.read()).decode()
            fp.write(f'{name}_py = "{b64str}"\n')
        else:
            print('transform finished')


def transform_js(to_trans: str, save_path: str) -> None:
    with open(save_path, 'w') as fp:
        _, name = os.path.split(to_trans)
        name, _ = os.path.splitext(name)
        with open(to_trans, 'r') as f:
            byte = f.read()
            print(byte)
            byte = byte.encode()
            b64str = base64.b64encode(byte).decode()
        fp.write(f'{name}_js = "{b64str}"\n')


if __name__ == '__main__':
    # transform(r'C:\Users\Lenovo\Desktop\images', './music/pictures.py')
    transform_js('./translate/sign.js', './translate/js_code.py')
