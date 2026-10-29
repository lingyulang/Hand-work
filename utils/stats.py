# -*- coding: utf-8 -*-
"""性能监控：FPS 统计、端到端延迟统计、CSV 逐帧记录。

延迟口径说明（用于评估报告）：
  pipeline_latency = 处理完成时刻 - 该帧采集时刻
  它覆盖 采集排队 + 骨骼推理 + 特征 + 渲染 全链路，
  不含屏幕刷新一行的时间（通常 <17ms）。
"""
import csv
import time
from collections import deque


class PerfMonitor:
    def __init__(self, fps_window=30, csv_path=None):
        self._frame_times = deque(maxlen=fps_window)
        self._latencies = deque(maxlen=fps_window)
        self._csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._t0 = time.time()
        if csv_path:
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["t_s", "fps", "latency_ms", "openness_raw",
                 "openness_smooth", "n_hands", "smooth_on"]
            )

    def tick_frame(self):
        """每帧调用一次：记录当前时刻用于 FPS 计算。"""
        self._frame_times.append(time.time())

    def tick_latency(self, frame_ts, done_ts=None):
        """记录一帧的端到端处理延迟。"""
        if frame_ts > 0:
            self._latencies.append((done_ts or time.time()) - frame_ts)

    def log_row(self, fps, lat_ms, openness_raw, openness_smooth, n_hands, smooth_on):
        if self._csv_writer:
            self._csv_writer.writerow([
                f"{time.time() - self._t0:.3f}", f"{fps:.2f}", f"{lat_ms:.1f}",
                f"{openness_raw:.4f}", f"{openness_smooth:.4f}",
                n_hands, int(smooth_on),
            ])
            self._csv_file.flush()  # 逐行落盘：异常终止(如强制结束进程)不丢数据

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    @property
    def latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies) * 1000.0

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
