# -*- coding: utf-8 -*-
# @Author: kewuaa
# @Date:   2022-01-21 18:36:13
# @Last Modified by:   None
# @Last Modified time: 2022-02-03 20:22:30
from io import BytesIO
import os
import sys
import base64
import asyncio

from hzy import fake_ua
from hzy.aiofile import aiofile
from PIL import Image
from aiohttp import ClientSession
from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QMainWindow
from PySide2.QtWidgets import QWidget
from PySide2.QtWidgets import QLabel
from PySide2.QtWidgets import QMessageBox
from PySide2.QtWidgets import QFileDialog
from PySide2.QtWidgets import QToolTip
from PySide2.QtWidgets import QVBoxLayout
from PySide2.QtWidgets import QHBoxLayout
from PySide2.QtWidgets import QToolButton
from PySide2.QtWidgets import QMenu
from PySide2.QtWidgets import QAction
from PySide2.QtMultimedia import QMediaPlayer
from PySide2.QtMultimedia import QMediaContent
from PySide2.QtMultimedia import QMediaPlaylist
from PySide2.QtCore import QModelIndex
from PySide2.QtCore import Qt
from PySide2.QtCore import QUrl
from PySide2.QtCore import Slot
from PySide2.QtCore import Signal
from PySide2.QtCore import QStringListModel
from PySide2.QtGui import QIcon
from PySide2.QtGui import QPixmap
from PySide2.QtUiTools import QUiLoader
from qasync import QEventLoop

try:
    from .pictures import *
    from .js_code import *
    from .ui_music_player import Ui_MainWindow
except ImportError:
    from pictures import *
    from js_code import *
    from ui_music_player import Ui_MainWindow


current_path, _ = os.path.split(__file__)
ua = fake_ua.UserAgent()


async def download(session, url, path):
    res = await session.get(url)
    async with aiofile.open_async(path, 'wb') as f:
        await f.write(await res.read())


async def download_img(session, url, path):
    res = await session.get(url)
    fp = BytesIO(await res.read())
    img = aiofile.AIOWrapper(Image.open(fp))
    img = aiofile.AIOWrapper(await img.resize((800, 800), Image.ANTIALIAS))
    img = aiofile.AIOWrapper(await img.convert('RGB'))
    await img.save(path)


class Musicer(object):
    """docstring for Musicer."""

    URL = 'https://music.163.com/weapi/song/enhance/player/url/v1?csrf_token='
    SEARCH_URL = 'https://music.163.com/weapi/cloudsearch/get/web?csrf_token='
    HEADERS = {
        'user-agent': '',
        'cookie': '_ntes_nuid=c6b62f4920356fba00bb02b9ab8ffb6e; _ntes_nnid=4fb10a59e820fa78798fdffdfa1dcbcc,1640399690494; hb_MA-BFF5-63705950A31C_source=www.google.com.hk; _iuqxldmzr_=32; NMTID=00OggMHqS2StC6LaUBPh-O98krfkUoAAAF-ex-VNw; WNMCID=jdtbkl.1642743174682.01.0; WEVNSM=1.0.0; WM_TID=uWRfK7Me84ZEFBBBRUJq6C7aA7%2BSeboU; WM_NI=%2FV4N5Jo6ZG3%2Bv6UpcZc6GBokJ3fGI31ctIO85DwBFe8aql29Cgj%2B6C8RTxQtzi4JxuKk1r0MYn8cNmzTsNpcQDbpoHE3FgwRJGll5ZqpwtEcLRGdij7Vuf233QYfsei%2Fc0U%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6ee84c63bf5b49ea6b66ba1968eb7d14f939f9abaf47b81f5e1acb64d8796bc82fc2af0fea7c3b92aed8afcafb842adea96a8b221b6ea9b91e252a1ea89a3ee618aedaaa8cc39f3bdbcaace34f79cb9a3c56ffcbfbf8db13381b9a6ccb549fc8e9c99d480bab7fcabfb7f89b5ff93b3479bb2f884cd67ba8cbdaff369939cae91ee4487ed8a88f65a8a99a48db852b3b0b995c869908e8985ec62f58abb87b139f18f9787f26fabf1add1d837e2a3; JSESSIONID-WYYY=HW13SDPOfx%2FFqo97Onz0zaiUoV351m%2BAJoZlZl4G41KTcn3h2oVK9ZkTEd40r%2FTdh%2FzjuRhxykV2pcCWIi8gjuKmVlc2Mag0ZAV9pgwYh4HQZRYQyD4qS3bP8uM1CMy5Dm7cKvql964nmGPnOkPA4OoB5s4HHYV69a4xzgMh3rWKIe0u%3A1642822349624; playerid=16777386',
        'refer': 'https://music.163.com/',
    }
    ENCSECKEY = 'ddb9e95ecba455a303a46b36f291368947d49531f824f5c4adbea2ff7ce22a2e0615a837d727ced55fdbfa85b3590466a39b85749ee5845d29786a7727fd8f154f953ca755d533fe84aa0f100c767f6dbc8441a5ad35711706cb9cf662018025a4519405aa738af496cd3d01594d62821ed0f39b4af97dee184b26e655dd4737'
    JS_FILE_PATH = os.path.join(current_path, 'music.js')
    CMD = "node {path} {request_str}"
    INFO_REQUEST_STR = [r'{\"hlpretag\":\"<span class="s-fc7">\",\"hlposttag\":\"</span>\",\"s\":\"',
                        '',
                        r'\",\"type\":\"1\",\"offset\":\"0\",\"total\":\"true\",\"limit\":\"30\",\"csrf_token\":\"\"}']
    URL_REQUEST_STR = [r'{\"ids\":\"[',
                       '',
                       r']\",\"level\":\"standard\",\"encodeType\":\"aac\",\"csrf_token\":\"\"}']
    # '{"ids": "[1293886117]","level":"standard","encodeType":"aac","csrf_token":""}'

    def __init__(self):
        super(Musicer, self).__init__()
        if not os.path.exists(self.JS_FILE_PATH):
            with open(self.JS_FILE_PATH, 'w') as f:
                f.write(base64.b64decode(music_js.encode()).decode())

    async def _get_song_info(self, session, song):
        self.HEADERS['user-agent'] = ua.get_ua()
        self.INFO_REQUEST_STR[1] = song
        request_str = ''.join(self.INFO_REQUEST_STR)
        data = {
            'params': await self._get_params(request_str),
            'encSecKey': self.ENCSECKEY,
        }
        res = await session.post(self.SEARCH_URL, headers=self.HEADERS, data=data)
        try:
            result_dict = await res.json(content_type=None)
        except Exception as e:
            print('error:', e)
            raise e
        else:
            assert (songs := result_dict.get('result')) is not None, '出现未知错误'
            songs = songs['songs']
            # songs = [song['al'] for song in songs]
        return songs

    async def _get_song_url(self, session, _id):
        self.HEADERS['user-agent'] = ua.get_ua()
        self.URL_REQUEST_STR[1] = str(_id)
        request_str = ''.join(self.URL_REQUEST_STR)
        data = {
            'params': await self._get_params(request_str),
            'encSecKey': self.ENCSECKEY,
        }
        res = await session.post(self.URL, headers=self.HEADERS, data=data)
        try:
            result_dict = await res.json(content_type=None)
        except Exception as e:
            print('error:', e)
            raise e
        else:
            url = result_dict['data'][0]['url']
            assert url is not None, 'VIP或无版权歌曲，无法播放与下载'
        return url

    async def _get_params(self, request_str: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            self.CMD.format(path=self.JS_FILE_PATH, request_str=request_str),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        assert not stderr, f'subprocess err: {stderr.decode("gbk")}'
        result = stdout.decode('utf-8').strip()
        return result


class MyDict(dict):
    """docstring for MyDict."""

    def __init__(self, slm):
        super(MyDict, self).__init__()
        self.slm = slm

    def __setitem__(self, k, v):
        super(MyDict, self).__setitem__(k, v)
        self.slm.setStringList(self.keys())

    def pop(self, k):
        v = super(MyDict, self).pop(k)
        self.slm.setStringList(self.keys())
        return v

    def clear(self):
        super(MyDict, self).clear()
        self.slm.setStringList([])


class SongLabel(QLabel):
    """docstring for SongLabel."""

    doubleclicked = Signal()

    def __init__(self, *args, **kwargs):
        super(SongLabel, self).__init__(*args, **kwargs)

    def mouseDoubleClickEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.doubleclicked.emit()


class MusicApp(object):
    """docstring for MusicApp."""

    DATA_PATH = f'{current_path}\\data'
    IMG_PATH = os.path.join(DATA_PATH, 'images')
    DOWNLOAD_PATH = os.path.join(DATA_PATH, 'musics')

    def __init__(self):
        super(MusicApp, self).__init__()
        if not os.path.exists(self.DATA_PATH):
            os.mkdir(self.DATA_PATH)
            os.mkdir(self.IMG_PATH)
        elif not os.path.exists(self.IMG_PATH):
            os.mkdir(self.IMG_PATH)
        if not os.path.exists(self.DOWNLOAD_PATH):
            os.mkdir(self.DOWNLOAD_PATH)
        self.init_sess()
        self.song_length = 'oo'
        self.voice = 33
        self.is_mute = False
        self.mode_icon_list = []
        self.listen_slm = QStringListModel()
        self.download_slm = QStringListModel()
        self.initUi()
        self.musicer = Musicer()
        self.music_player = QMediaPlayer(parent=self.ui)
        self.music_play_list = QMediaPlaylist(parent=self.ui)
        self.music_play_list.setPlaybackMode(QMediaPlaylist.Loop)
        self.music_play_list.currentIndexChanged.connect(self.media_list_index_change)
        self.music_player.durationChanged.connect(self.get_duration_length)
        self.music_player.positionChanged.connect(self.position_change)
        self.music_player.setVolume(self.voice)
        self.current_search = None
        self.to_listen = MyDict(self.listen_slm)
        self.to_download = MyDict(self.download_slm)

    def init_sess(self):
        async def get_sess():
            return ClientSession()
        loop = asyncio.get_running_loop()
        self.session = loop.run_until_complete(get_sess())

    def initUi(self):
        app = self

        class PlayerUi(QMainWindow, Ui_MainWindow):
            """docstring for PlayerUi."""

            def __init__(self):
                super(PlayerUi, self).__init__()
                self.setupUi(self)
                self.setWindowIcon(app._get_icon(window_icon_py))

            def closeEvent(self, event):
                result = QMessageBox.question(self, '请确认', '是否确认关闭',
                                              QMessageBox.Yes | QMessageBox.No)
                if result == QMessageBox.Yes:
                    asyncio.create_task(app.session.close())
                    event.accept()
                else:
                    event.ignore()
        # self.ui = QUiLoader().load('ui_music_player.ui')
        self.ui = PlayerUi()
        self.ui.searchButton.clicked.connect(self.search)
        self.layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(self.layout)
        self.menu = QMenu()
        self.change_download_path_action = QAction(
            '改变下载路径', triggered=self.change_download_path)
        self.menu.addAction(self.change_download_path_action)
        self.ui.resultscrollArea.setWidget(widget)
        self.ui.voicepushButton.clicked.connect(self.mute)
        self.ui.playpushButton.clicked.connect(self.play_pause)
        self.ui.playpushButton.setEnabled(False)
        self.ui.lastpushButton.clicked.connect(self.last_song)
        self.ui.lastpushButton.setEnabled(False)
        self.ui.nextpushButton.clicked.connect(self.next_song)
        self.ui.nextpushButton.setEnabled(False)
        self.ui.modepushButton.clicked.connect(self.change_mode())
        self.ui.modepushButton.setEnabled(False)
        self.ui.downloadtoolButton.clicked.connect(self.download)
        self.ui.downloadtoolButton.setMenu(self.menu)
        self.ui.hideandshowpushButton.clicked.connect(self.hide_and_show)
        self.ui.listenlistView.setModel(self.listen_slm)
        self.ui.listenlistView.doubleClicked.connect(self.list_play)
        self.ui.listenlistView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.listenlistView.customContextMenuRequested.connect(
            self.listen_listview_contextmenu)
        self.ui.downloadlistView.setModel(self.download_slm)
        self.ui.downloadlistView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.downloadlistView.customContextMenuRequested.connect(
            self.download_listview_contextmenu)
        self.ui.voicehorizontalSlider.setRange(0, 100)
        self.ui.voicehorizontalSlider.setValue(self.voice)
        self.ui.voicehorizontalSlider.valueChanged.connect(self.voice_change)
        self.ui.processhorizontalSlider.sliderMoved.connect(self.process_change)
        self.ui.processhorizontalSlider.setEnabled(False)
        self.ui.downloadgroupBox.hide()
        asyncio.create_task(self.add_style())

    async def add_style(self):
        def add_style_func():
            self.ui.statusBar().showMessage(
                f'current download path: {self.DOWNLOAD_PATH}')
            self.ui.searchButton.setToolTip('<b>点击此按钮进行搜索</b>')
            self.ui.voicepushButton.setToolTip('<b>音量</b>')
            self.ui.playpushButton.setToolTip('<b>播放</b>')
            self.ui.nextpushButton.setToolTip('<b>下一曲</b>')
            self.ui.lastpushButton.setToolTip('<b>上一曲</b>')
            self.ui.modepushButton.setToolTip('<b>列表循环</b>')
            self.ui.downloadtoolButton.setToolTip('<b>下载</b>')
            self.ui.hideandshowpushButton.setToolTip('<b>展开下载列表</b>')
            self.ui.voicehorizontalSlider.setToolTip(f'<b>{self.voice}</b>')
            self.add_icon = self._get_icon(add_py)
            self.add_music_icon = self._get_icon(add_music_py)
            self.play_icon = self._get_icon(play_py)
            self.pause_icon = self._get_icon(pause_py)
            self.sound_on_icon = self._get_icon(sound_on_py)
            self.sound_off_icon = self._get_icon(sound_off_py)
            self.mode_icon_list.extend([
                list_loop_icon := self._get_icon(list_loop_py),
                self._get_icon(random_py),
                self._get_icon(item_loop_py)])
            self.hide_icon = self._get_icon(hide_py)
            self.show_icon = self._get_icon(show_py)
            self.ui.playpushButton.setIcon(self.play_icon)
            self.ui.voicepushButton.setIcon(self.sound_on_icon)
            self.ui.hideandshowpushButton.setIcon(self.hide_icon)
            self.ui.searchButton.setIcon(self._get_icon(search_py))
            self.ui.nextpushButton.setIcon(self._get_icon(next_py))
            self.ui.lastpushButton.setIcon(self._get_icon(previous_py))
            self.ui.modepushButton.setIcon(list_loop_icon)
            self.ui.downloadtoolButton.setIcon(self._get_icon(download_py))
            self.change_download_path_action.setIcon(self._get_icon(download_path_setting_py))
        await asyncio.get_running_loop().run_in_executor(None, add_style_func)

    def listen_listview_contextmenu(self, pos):
        # pos = self.ui.listenlistView.viewport().mapFromGlobal(pos)
        popmenu = QMenu()
        remove_action = QAction('移出播放列表', triggered=self.remove_listen_item)
        add_to_download_action = QAction('添加至下载列表', triggered=self.listen_to_download)
        clear_action = QAction('清空播放列表', triggered=self.clear_listen_items)
        if self.ui.listenlistView.indexAt(pos).isValid():
            popmenu.addAction(remove_action)
            popmenu.addAction(add_to_download_action)
        popmenu.addAction(clear_action)
        popmenu.exec_(self.ui.listenlistView.mapToGlobal(pos))

    @Slot()
    def listen_to_download(self):
        index = self.ui.listenlistView.currentIndex()
        song_info = index.data()
        self.to_download[song_info] = self.to_listen[song_info]

    def download_listview_contextmenu(self, pos):
        popmenu = QMenu()
        remove_action = QAction('移出下载列表', triggered=self.remove_download_item)
        clear_action = QAction('清空下载列表', triggered=self.clear_download_items)
        if self.ui.downloadlistView.indexAt(pos).isValid():
            popmenu.addAction(remove_action)
        popmenu.addAction(clear_action)
        popmenu.exec_(self.ui.downloadlistView.mapToGlobal(pos))

    @Slot(int)
    def media_list_index_change(self, index):
        qmodel_index = self.listen_slm.index(index)
        self.ui.listenlistView.setCurrentIndex(qmodel_index)
        self.ui.processhorizontalSlider.setToolTip(f'<b>{qmodel_index.data()}</b>')

    @Slot()
    def remove_listen_item(self):
        index = self.ui.listenlistView.currentIndex()
        self.to_listen.pop(index.data())
        self.music_play_list.removeMedia(index.row())

    @Slot()
    def remove_download_item(self):
        index = self.ui.downloadlistView.currentIndex()
        self.to_download.pop(index.data())

    @Slot()
    def clear_listen_items(self):
        if self.to_listen:
            self.to_listen.clear()
            self.music_play_list.clear()

    @Slot()
    def clear_download_items(self):
        if self.to_download:
            self.to_download.clear()

    def change_mode(self):
        mode_list = [QMediaPlaylist.Loop,
                     QMediaPlaylist.Random,
                     QMediaPlaylist.CurrentItemInLoop]
        tips = ['<b>列表循环</b>', '<b>随机播放</b>', '<b>单曲循环</b>']
        push_times = 0

        @Slot()
        def connect_func():
            nonlocal push_times
            push_times += 1
            index = push_times % 3
            self.ui.modepushButton.setIcon(self.mode_icon_list[index])
            self.ui.modepushButton.setToolTip(tips[index])
            self.music_play_list.setPlaybackMode(mode_list[index])
        return connect_func

    @staticmethod
    def _get_icon(b64str: str) -> QIcon:
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(b64str.encode()))
        return QIcon(pixmap)

    @staticmethod
    def format_time(time: int) -> str:
        seconds = time / 1000
        return f'{int(seconds // 60)}:{int(seconds % 60)}'

    def clear_layout(self):
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            self.layout.removeItem(item)

    @Slot()
    def search(self):
        song = self.ui.inputlineEdit.text()
        if song and song != self.current_search:
            self.clear_layout()
            asyncio.create_task(self.search_song(song))
            self.current_search = song

    @Slot()
    def hide_and_show(self):
        if self.ui.downloadgroupBox.isHidden():
            self.ui.downloadgroupBox.show()
            self.ui.hideandshowpushButton.setIcon(self.show_icon)
            self.ui.hideandshowpushButton.setToolTip('<b>收起下载列表</b>')
        else:
            self.ui.downloadgroupBox.hide()
            self.ui.hideandshowpushButton.setIcon(self.hide_icon)
            self.ui.hideandshowpushButton.setToolTip('<b>展开下载列表</b>')

    @Slot(int)
    def get_duration_length(self, song_length):
        if song_length:
            self.ui.processhorizontalSlider.setRange(0, song_length)
            self.song_length = self.format_time(song_length)
        else:
            self.ui.timelabel.setText('--/--')

    @Slot()
    def position_change(self, pos):
        if (current_pos := self.format_time(pos)) != self.song_length:
            if not self.ui.processhorizontalSlider.isEnabled():
                self.ui.processhorizontalSlider.setEnabled(True)
            if not self.ui.playpushButton.isEnabled():
                self.ui.playpushButton.setEnabled(True)
            self.ui.processhorizontalSlider.setValue(pos)
            self.ui.timelabel.setText(f'{current_pos}/{self.song_length}')
        else:
            self.ui.processhorizontalSlider.setValue(0)
            self.ui.processhorizontalSlider.setEnabled(False)
            self.ui.processhorizontalSlider.setToolTip('')
            self.ui.playpushButton.setEnabled(False)
            self.ui.timelabel.setText('--/--')
            self.ui.playpushButton.setIcon(self.play_icon)
            self.ui.playpushButton.setToolTip('<b>播放</b>')

    @Slot()
    def download(self):
        if self.to_download:
            for song_info in self.to_download.keys():
                if not os.path.exists(
                        path := os.path.join(
                            self.DOWNLOAD_PATH, f'{self.to_download[song_info]}.m4a')):
                    asyncio.create_task(self.download_music(song_info, path))
                else:
                    async def pop_item():
                        self.to_download.pop(song_info)
                    asyncio.create_task(pop_item())
        else:
            QMessageBox.warning(self.ui, 'warning', 'no song could be downloaded')

    def add_listen(self, song_info, _id):
        async def add_func():
            if _id not in self.to_listen.values():
                self.music_play_list.addMedia(await self._get_mediacontent(_id))
                self.to_listen[song_info] = _id
            # print('listen:', self.to_listen)

        @Slot()
        def connect_func():
            asyncio.create_task(add_func())
        return connect_func

    def add_download(self, song_info, _id):
        @Slot()
        def connect_func():
            self.to_download[song_info] = _id
            # print('download:', self.to_download)
        return connect_func

    @Slot()
    def mute(self):
        if self.is_mute:
            self.is_mute = False
            self.ui.voicehorizontalSlider.setValue(self.voice)
            self.ui.voicepushButton.setIcon(self.sound_on_icon)
            self.music_player.setVolume(self.voice)
        else:
            self.is_mute = True
            self.ui.voicehorizontalSlider.setValue(0)
            self.ui.voicepushButton.setIcon(self.sound_off_icon)
            self.music_player.setVolume(0)

    @Slot(int)
    def voice_change(self, value):
        self.music_player.setVolume(value)
        self.ui.voicehorizontalSlider.setToolTip(f'<b>{value}</b>')
        if not self.is_mute:
            self.voice = value

    @Slot(int)
    def process_change(self, value):
        self.music_player.setPosition(value)

    @Slot()
    def change_download_path(self):
        file_open = QFileDialog()
        file_open.setFileMode(QFileDialog.Directory)
        if file_open.exec_():
            path, *_ = file_open.selectedFiles()
            self.DOWNLOAD_PATH = path
            self.ui.statusBar().showMessage(
                f'current download path: {self.DOWNLOAD_PATH}')

    async def _get_mediacontent(self, _id):
        if os.path.exists(
                path := os.path.join(self.DOWNLOAD_PATH, f'{_id}.m4a')):
            mediacontent = QMediaContent(QUrl.fromLocalFile(path))
        else:
            url = await self._get_url(_id)
            mediacontent = QMediaContent(QUrl(url))
        return mediacontent

    def single_play(self, song_info, _id):
        async def play():
            self.music_player.setMedia(await self._get_mediacontent(_id))
            self.ui.playpushButton.setIcon(self.pause_icon)
            self.ui.playpushButton.setToolTip('<b>暂停</b>')
            self.ui.processhorizontalSlider.setToolTip(f'<b>{song_info}</b>')
            self.ui.lastpushButton.setEnabled(False)
            self.ui.nextpushButton.setEnabled(False)
            self.ui.modepushButton.setEnabled(False)
            self.music_player.play()

        @Slot()
        def connect_func():
            asyncio.create_task(play())
        return connect_func

    @Slot()
    def list_play(self, qmodel_list):
        index = qmodel_list.row()
        self.ui.playpushButton.setIcon(self.pause_icon)
        self.music_player.setPlaylist(self.music_play_list)
        self.music_play_list.setCurrentIndex(index)
        self.music_player.play()
        self.ui.processhorizontalSlider.setToolTip(f'<b>{qmodel_list.data()}</b>')
        self.ui.lastpushButton.setEnabled(True)
        self.ui.nextpushButton.setEnabled(True)
        self.ui.modepushButton.setEnabled(True)

    @Slot()
    def last_song(self):
        if not self.music_play_list.currentIndex():
            self.music_play_list.setCurrentIndex(
                self.music_play_list.mediaCount() - 1)
        else:
            self.music_play_list.previous()

    @Slot()
    def play_pause(self):
        if self.ui.processhorizontalSlider.isEnabled():
            if self.music_player.state() - 1:
                self.music_player.play()
                self.ui.playpushButton.setIcon(self.pause_icon)
                self.ui.playpushButton.setToolTip('<b>暂停</b>')
            else:
                self.music_player.pause()
                self.ui.playpushButton.setIcon(self.play_icon)
                self.ui.playpushButton.setToolTip('<b>播放</b>')

    @Slot()
    def next_song(self):
        if self.music_play_list.currentIndex() == self.music_play_list.mediaCount() - 1:
            self.music_play_list.setCurrentIndex(0)
        else:
            self.music_play_list.next()

    async def search_song(self, song):
        try:
            result = await self.musicer._get_song_info(self.session, song)
        except AssertionError as e:
            QMessageBox.critical(self.ui, 'error', str(e))
            self.ui.inputlineEdit.clear()
        else:
            for song_info in result:
                item = QWidget()
                item.setLayout(hbox := QHBoxLayout())
                hbox.addWidget(add_music_button := QToolButton())
                hbox.addWidget(add_button := QToolButton())
                hbox.addWidget(
                    song_label := SongLabel(
                        text := f"{song_info['name']}-->{song_info['ar'][0]['name']}-->《{song_info['al']['name']}》"))
                add_music_button.clicked.connect(self.add_listen(text, song_info['id']))
                add_button.clicked.connect(self.add_download(text, song_info['id']))
                song_label.doubleclicked.connect(self.single_play(text, song_info['id']))
                add_music_button.setIcon(self.add_music_icon)
                add_button.setIcon(self.add_icon)
                add_music_button.setAutoRaise(True)
                add_button.setAutoRaise(True)
                add_music_button.setToolTip('<b>添加至播放列表</b>')
                add_button.setToolTip('<b>添加至下载列表</b>')
                self.layout.addWidget(item)
                if not os.path.exists(
                        path := os.path.join(
                            self.IMG_PATH, f"{song_info['al']['pic']}.jpg")):
                    asyncio.create_task(
                        download_img(self.session, song_info['al']['picUrl'], path))
                song_label.setToolTip(f'<img src={path} >')
                await asyncio.sleep(0)

    async def download_music(self, song_info, path):
        try:
            url = await self._get_url(self.to_download[song_info])
            await download(self.session, url, path)
        finally:
            self.to_download.pop(song_info)

    async def _get_url(self, _id):
        try:
            url = await self.musicer._get_song_url(self.session, _id)
        except AssertionError as e:
            asyncio.current_task().cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                if _id not in self.to_download.values():
                    song = ''
                else:
                    song = [song_info
                            for song_info, id_ in self.to_download.items()
                            if id_ == _id][0]
                QMessageBox.critical(
                    self.ui, 'error',
                    f'{song}: {str(e)}' if song else str(e))
                raise
        else:
            return url

    def show(self):
        self.ui.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        music_app = MusicApp()
        music_app.show()
        loop.run_forever()
