# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-15 08:58:38
# @Last Modified by:   None
# @Last Modified time: 2022-01-23 08:59:50
from typing import NoReturn, Text, Dict, Tuple, Callable

# from pprint import pprint
from collections import deque
from sys import argv
from sys import exit
import json
import os
import re
import asyncio

from lxml.html import fromstring
from aiohttp import ClientSession
from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QMainWindow
from PySide2.QtCore import QThread
from PySide2.QtCore import Signal
from PySide2.QtCore import Slot
from PySide2.QtCore import Qt
import requests as req
import fake_ua
# import execjs

try:
    from .ui_translate import Ui_MainWindow
except ImportError:
    from ui_translate import Ui_MainWindow


current_path, _ = os.path.split(__file__)
ua = fake_ua.UserAgent()
with open(f'{current_path}/cookies.setting', 'r') as f:
    cookie = f.read()


class BaiduTranslater(object):
    """docstring for BaiduTranslater."""

    BASE_URL = 'https://fanyi.baidu.com'
    POST_URL = 'https://fanyi.baidu.com/v2transapi?'
    HEADERS = {
        'user-agent': '',
        'cookie': cookie,
    }
    CMD = 'node {path} "{query}"'
    FROM = '中文'
    TO = '英语'
    DOMAIN = '通用'

    def __init__(self, _from=None, to=None, domain=None):
        super(BaiduTranslater, self).__init__()
        if _from is not None:
            self.FROM = _from
        if to is not None:
            self.TO = to
        if domain is not None:
            self.DOMAIN = domain
        # with open('sign.js', 'r') as f:
        #     js_code = f.read()
        # self._get_sign_func = execjs.compile(js_code)
        res = req.get(self.BASE_URL, headers=self.HEADERS).text
        token, self._lang_map = self._get_token_and_map(res)
        self._domain_map = self._get_domains_map(res)
        self._map_dict = {**self._lang_map, **self._domain_map}
        self.data = {
            'from': self._lang_map[self.FROM],
            'to': self._lang_map[self.TO],
            'domain': self._domain_map[self.DOMAIN],
            'token': token,
        }

    def __call__(self, query: Text) -> Text:
        """获取翻译结果."""
        sign = self._get_sign(query)
        self.HEADERS['user-agent'] = ua.get_ua()
        self.data.update({'sign': sign, 'query': query})
        result = req.post(
            self.POST_URL, headers=self.HEADERS, data=self.data).json()
        assert (trans_result := result.get('trans_result')) is not None, \
            '出现未知错误'
        trans_result = trans_result['data'][0]['dst']
        return trans_result

    async def async_trans(self, query: Text) -> Text:
        """异步获取翻译结果."""
        coro = asyncio.to_thread(self._get_sign, query)
        sign = await coro
        self.HEADERS['user-agent'] = ua.get_ua()
        self.data.update({'sign': sign, 'query': query})
        async with ClientSession() as session:
            async with session.post(
                    self.POST_URL, headers=self.HEADERS, data=self.data) as response:
                result = await response.json()
                assert (trans_result := result.get('trans_result')) is not None, \
                    '出现未知错误'
                trans_result = trans_result['data'][0]['dst']
                return trans_result

    def _get_token_and_map(self, res: 'request response') -> Tuple[Text, Dict]:
        """获得token参数."""
        token = re.search(r"token: '(.*)',", res).group(1)
        lang_map = re.search(r'langList: ({[\s\S]*?})', res).group(1)
        # lang_map = re.sub(r"\s", '', lang_map)
        lang_map = lang_map.replace("'", '"')
        lang_map = json.loads(lang_map)
        lang_map = {lang: abbre for abbre, lang in lang_map.items()}
        return token, lang_map

    def _get_domains_map(self, res: 'request response') -> Dict:
        """获得domain."""
        tree = fromstring(res)
        domain_trans_iter_wrappers = tree.xpath(
            '//div[@class="domain-trans domain-trans-small"]/div[2]/div')
        domain_map = {
            '通用': 'common',
        }
        for domain_trans_iter_wrapper in domain_trans_iter_wrappers:
            value = domain_trans_iter_wrapper.xpath(
                './div[1]/@data-domain-value')[0]
            key = domain_trans_iter_wrapper.xpath(
                './div[1]/span[1]/text()')[0]
            domain_map[key] = value
        return domain_map

    def _get_sign(self, query: Text) -> Text:
        """获得sign参数."""
        popline = os.popen(self.CMD.format(
            path=f'{current_path}/sign.js', query=query))
        return popline.read().strip()


class TransThread(QThread):
    """docstring for TransThread."""

    signal = Signal(str)
    clear_signal = Signal()

    def __init__(self, translater, queue):
        super(TransThread, self).__init__()
        self.translater = translater
        self.queue = queue
        self.running = True

    async def emit_result(self, query):
        try:
            result = await self.translater.async_trans(query)
        except AssertionError as e:
            result = str(e)
        self.signal.emit(result)

    async def _run(self):
        while self.running:
            if self.queue:
                # print(self.queue)
                query = self.queue.popleft()
                if query:
                    asyncio.create_task(self.emit_result(query))
                else:
                    loop = asyncio.get_running_loop()
                    loop.call_later(1, self.clear_signal.emit)
            else:
                await asyncio.sleep(0)
        # else:
        #     print('task shutdown')

    def run(self):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._run())
        self.loop.run_forever()


class TransApp(object):
    """docstring for TransApp."""

    def __init__(self):
        super(TransApp, self).__init__()
        self.clipboard = QApplication.clipboard()
        self.translater = BaiduTranslater()
        self.queue = deque([], maxlen=3)
        self.init_Ui()
        self.run_thread = TransThread(self.translater, self.queue)
        self.run_thread.signal.connect(self.show_result)
        self.run_thread.clear_signal.connect(self.clear_edit)
        self.ui.close_signal.connect(self.shutdown_loop)

    def init_Ui(self):
        init_from_lang = self.translater.FROM
        init_to_lang = self.translater.TO
        init_domain = self.translater.DOMAIN
        langs = self.translater._lang_map.keys()
        domains = self.translater._domain_map.keys()

        app = self

        class TransUi(QMainWindow, Ui_MainWindow):
            """docstring for TransUi."""

            close_signal = Signal()

            def __init__(self):
                super(TransUi, self).__init__()
                self.setupUi(self)

            def closeEvent(self, event):
                self.close_signal.emit()
                super(TransUi, self).closeEvent(event)

            def show(self):
                app.run_thread.running = True
                app.run_thread.start()
                super(TransUi, self).show()

        # self.ui = QUiLoader().load('ui_translate.ui')
        self.ui = TransUi()
        self.ui.from_comboBox.addItems(langs)
        self.ui.to_comboBox.addItems(langs)
        self.ui.domain_comboBox.addItems(domains)
        self.ui.from_comboBox.currentIndexChanged[str].connect(
            self.reset_data('from'))
        self.ui.to_comboBox.currentIndexChanged[str].connect(
            self.reset_data('to'))
        self.ui.domain_comboBox.currentIndexChanged[str].connect(
            self.reset_data('domain'))
        self.ui.from_comboBox.activated.connect(lambda x: self.add_to_queue())
        self.ui.to_comboBox.activated.connect(lambda x: self.add_to_queue())
        self.ui.domain_comboBox.activated.connect(lambda x: self.add_to_queue())
        self.ui.from_comboBox.setCurrentText(init_from_lang)
        self.ui.to_comboBox.setCurrentText(init_to_lang)
        self.ui.domain_comboBox.setCurrentText(init_domain)
        self.ui.from_textEdit.textChanged.connect(self.add_to_queue)
        self.ui.exchangeButton.clicked.connect(self.exchange)
        self.ui.action.triggered.connect(self.copy)
        # self.ui.show()

    def reset_data(self, key: Text) -> Callable:
        map_dict = self.translater._map_dict

        @Slot(str)
        def reset(current_text: Text):
            self.translater.data[key] = map_dict[current_text]
            # print(f'reset -> {current_text}')
            # print(self.translater.data)
        return reset

    @Slot()
    def clear_edit(self):
        if not self.ui.from_textEdit.toPlainText():
            self.ui.to_textEdit.clear()

    @Slot()
    def add_to_queue(self):
        query = self.ui.from_textEdit.toPlainText()
        self.queue.append(query)

    @Slot()
    def exchange(self):
        _from = self.ui.from_comboBox.currentText()
        to = self.ui.to_comboBox.currentText()
        self.ui.from_comboBox.setCurrentText(to)
        self.ui.to_comboBox.setCurrentText(_from)
        self.add_to_queue()

    @Slot()
    def copy(self):
        text = self.ui.to_textEdit.toPlainText()
        self.clipboard.setText(text)

    @Slot(str)
    def show_result(self, result: Text) -> NoReturn:
        self.ui.to_textEdit.setPlainText(result)

    @Slot()
    def shutdown_loop(self):
        self.run_thread.running = False
        self.run_thread.loop.stop()
        while self.run_thread.loop.is_running():
            pass
        else:
            # print('loop close')
            self.run_thread.loop.close()


if __name__ == '__main__':
    app = QApplication(argv)
    translater = TransApp()
    translater.ui.show()
    exit(app.exec_())
