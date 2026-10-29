# -*- coding: utf-8 -*-
"""摄像头采集线程。

独立线程持续抓取最新帧并打上时间戳，主循环始终处理"最新一帧"，
避免因处理耗时导致的帧堆积，从而降低端到端延迟（性能优化点）。
"""
import threading
import time

import cv2


class CameraStream:
    """后台线程摄像头读取器。

    用法:
        cam = CameraStream(0).start()
        ok, frame, ts = cam.read()   # ts 为该帧的采集时刻 time.time()
    """

    def __init__(self, index=0, width=640, height=480, fps=30):
        self.index = index
        # 后端优先级：MSMF 满帧率(实测30fps) > DSHOW(实测仅15fps，仅作回退) > 默认
        # MSMF 冷启动比 DSHOW 慢约1秒，但采集帧率决定整体延迟，值得取舍
        self.cap = None
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                self.cap = cap
                break
        if self.cap is None:
            raise RuntimeError(
                f"无法打开摄像头 {index}。请检查设备连接，"
                f"或用 --video <文件路径> 使用离线视频测试。"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0
        self._running = False
        self._thread = None

    # -- 线程控制 ------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._ts = time.time()

    def read(self):
        """返回 (成功, 最新帧, 采集时间戳)；无新帧时成功为 False。"""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._ts

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    @staticmethod
    def list_cameras(max_probe=4):
        """探测可用摄像头索引，便于现场排查设备。"""
        available = []
        for i in range(max_probe):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok, _ = cap.read()
            cap.release()
            if ok:
                available.append(i)
        return available
