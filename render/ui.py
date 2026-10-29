# -*- coding: utf-8 -*-
"""分屏 UI：左侧摄像头+骨骼连线，右侧五瓣花动画，底部信息栏。

性能设计：PIL 中文文本整帧转换约 8ms，是渲染主要瓶颈之一。
本模块将文本拆为三类，避免逐帧 PIL 调用：
  * 静态层  : 标题/按键说明/分隔装饰 —— 只渲染一次；
  * 动态层  : FPS/延迟/状态等数值 —— 以 10Hz 低频重绘进缓存贴片，
              按键切换（花色/平滑/手臂）时 invalidate() 立即重建；
  * 提示层  : "未检测到手" 按状态惰性重建。
每帧只做 numpy 贴片（copyto+mask），文本开销摊销后 <1ms/帧。
"""
import time
from collections import deque

import cv2
import numpy as np

from .text_cn import draw_texts_cn

_ORANGE = (80, 170, 255)
_WHITE = (255, 255, 255)
_GRAY = (170, 170, 170)
_CYAN = (255, 220, 120)
_RED = (90, 90, 230)
_DYN_PERIOD = 0.1  # 动态文本层重建周期（秒）


class AppUI:
    """组装 1280x720 主窗口画面。"""

    def __init__(self, cfg, cam_w=640, cam_h=480):
        self.cfg = cfg
        self.cam_w, self.cam_h = cam_w, cam_h
        # 摄像头面板垂直居中区域
        self.cam_y0 = (cfg.WINDOW_H - cfg.INFO_BAR_H - cam_h) // 2
        # 手部运动轨迹缓冲（归一化坐标）
        self._trail = deque(maxlen=cfg.TRAJECTORY_LEN)
        self._trail_time = deque(maxlen=cfg.TRAJECTORY_LEN)
        # ---- 文本缓存层 ----
        self._top_layer = None      # (bgr, mask3) 标题区 y0..44
        self._bar_static = None     # 底栏背景+按键说明（静态）
        self._bar_dyn = None        # (bgr, mask3) 底栏动态数值
        self._dyn_ts = 0.0          # 上次动态层重建时间
        self._tip_layer = None      # (bgr, mask3) 未检测到手提示
        self._tip_state = None
        self._dirty = True          # 按键切换后强制重建

    # ------------------------------------------------------------------
    def invalidate(self):
        """外部状态切换（花色/平滑/手臂）后调用，立即重建文本层。"""
        self._dirty = True

    def push_trail(self, center_norm):
        """记录手部中心轨迹点。"""
        self._trail.append(center_norm)
        self._trail_time.append(time.time())

    def clear_trail(self):
        self._trail.clear()

    # ------------------------------------------------------------------
    # 文本层构建
    # ------------------------------------------------------------------
    @staticmethod
    def _to_layer(strip_bgr):
        """文本贴片 -> (bgr, 3通道mask)，供 copyto 就地贴字。"""
        mask = np.any(strip_bgr != 0, axis=2)
        return strip_bgr, np.repeat(mask[:, :, None], 3, axis=2)

    def _build_top_layer(self, stats, cfg, cam_pw):
        W = cfg.WINDOW_W
        strip = np.zeros((44, W, 3), np.uint8)
        jobs = [
            ("摄像头画面 | 手部骨骼", (14, 12), 20, _WHITE, True),
            ("五瓣花动画 | 隔空控制", (cam_pw + 14, 12), 20, _WHITE, True),
            (f"花色: {stats.get('palette_name', '')}", (W - 120, 12), 16, _CYAN, False),
        ]
        if stats.get("pose_on"):
            jobs.append(("手臂姿态: 开", (cam_pw - 130, 12), 16, _CYAN, False))
        draw_texts_cn(strip, jobs)
        self._top_layer = self._to_layer(strip)

    def _build_bar_static(self, cfg):
        W, H, bar = cfg.WINDOW_W, cfg.WINDOW_H, cfg.INFO_BAR_H
        strip = np.zeros((bar, W, 3), np.uint8)
        strip[:] = (18, 16, 14)
        cv2.line(strip, (0, 0), (W, 0), (90, 90, 90), 2)
        draw_texts_cn(strip, [
            ("[S]平滑开关 [M]算法 [C]花色 [P]手臂 [H]帮助 [Q]退出",
             (16, 38), 14, _GRAY, False),
        ])
        self._bar_static = strip

    def _build_bar_dyn(self, stats, cfg):
        """底栏动态数值层（10Hz 或脏标记时重建）。"""
        W, bar = cfg.WINDOW_W, cfg.INFO_BAR_H
        strip = np.zeros((bar, W, 3), np.uint8)

        fps = stats.get("fps", 0.0)
        lat = stats.get("latency_ms", 0.0)
        bloom = stats.get("bloom", 0.0)
        fps_color = (120, 220, 120) if fps >= 25 else (_ORANGE if fps >= 15 else _RED)
        lat_color = (120, 220, 120) if lat < 100 else _RED
        smooth_on = stats.get("smooth_on")
        jobs = [
            (f"FPS {fps:5.1f}", (16, 12), 20, fps_color, True),
            (f"端到端延迟 {lat:4.0f} ms", (150, 12), 20, lat_color, False),
            (f"伸展度 {bloom:.2f}", (408, 15), 15, _WHITE, False),
            (f"平滑: {'开' if smooth_on else '关'} ({stats.get('smoother_name', '')})",
             (680, 12), 18, (140, 230, 140) if smooth_on else _GRAY, False),
        ]
        state = "盛开" if bloom > 0.8 else ("绽放中" if bloom > 0.25 else "花苞")
        nf = stats.get("n_flowers", 1)
        jobs.append((f"状态: {state}" + (f" x{nf} (双手各控一朵)" if nf == 2 else ""),
                     (680, 36), 15, _CYAN, False))
        draw_texts_cn(strip, jobs)
        self._bar_dyn = self._to_layer(strip)

    def _build_tip_layer(self, cfg, cam_pw):
        strip = np.zeros((cfg.WINDOW_H - cfg.INFO_BAR_H, cam_pw, 3), np.uint8)
        tw = 320
        cv2.rectangle(strip, (cam_pw // 2 - tw // 2 - 10, 340),
                      (cam_pw // 2 + tw // 2 + 10, 380), (30, 30, 30), -1)
        draw_texts_cn(strip, [("未检测到手部，请将手置于摄像头前",
                               (cam_pw // 2 - tw // 2 + 6, 348), 18,
                               (120, 200, 255), False)])
        self._tip_layer = self._to_layer(strip)

    # ------------------------------------------------------------------
    def compose(self, cam_frame, flower_panel, stats, cfg):
        """拼装最终窗口画面。

        Parameters
        ----------
        cam_frame : ndarray|None  摄像头帧（640x480 BGR）
        flower_panel : ndarray    FlowerRenderer 输出面板 (640x660)
        stats : dict              FPS / 延迟 / 平滑状态等监控数据
        """
        W, H, bar = cfg.WINDOW_W, cfg.WINDOW_H, cfg.INFO_BAR_H
        cam_pw = cfg.CAM_PANEL_W
        canvas = np.zeros((H, W, 3), np.uint8)

        # ============ 左侧：摄像头面板 ============
        canvas[:, :cam_pw] = (28, 26, 24)
        if cam_frame is not None:
            frame = cam_frame.copy()
            # 画手部运动轨迹（渐隐圆点）
            now = time.time()
            for i, (nx, ny) in enumerate(self._trail):
                age = now - self._trail_time[i]
                fade = max(0.0, 1.0 - age * 1.2)
                if fade <= 0:
                    continue
                px, py = int(nx * self.cam_w), int(ny * self.cam_h)
                color = tuple(int(c * fade) for c in _ORANGE)
                cv2.circle(frame, (px, py), int(3 + 4 * fade), color, -1, cv2.LINE_AA)

            y0 = self.cam_y0
            canvas[y0:y0 + self.cam_h, 0:cam_pw] = frame

        # 未检测到手提示（惰性缓存层）
        if stats.get("hand_lost"):
            if self._tip_state is not True:
                self._build_tip_layer(cfg, cam_pw)
                self._tip_state = True
            bgr, m3 = self._tip_layer
            np.copyto(canvas[:H - bar, :cam_pw], bgr, where=m3)
        else:
            self._tip_state = False

        # 分隔线
        cv2.line(canvas, (cam_pw, 0), (cam_pw, H - bar), (90, 90, 90), 2)

        # ============ 右侧：花朵面板 ============
        fx0 = cam_pw
        fh = flower_panel.shape[0]
        fy0 = (H - bar - fh) // 2
        canvas[fy0:fy0 + fh, fx0:fx0 + flower_panel.shape[1]] = flower_panel
        cv2.rectangle(canvas, (fx0, 0), (W - 1, H - bar - 1), (70, 70, 70), 1)

        # ============ 底部信息栏 ============
        if self._bar_static is None:
            self._build_bar_static(cfg)
        canvas[H - bar:] = self._bar_static

        # 伸展度进度条（cv2 矩形逐帧画，开销可忽略）
        bx, by, bw, bh = 400, 14, 240, 22
        bloom = stats.get("bloom", 0.0)
        by_abs = H - bar + by
        cv2.rectangle(canvas, (bx, by_abs), (bx + bw, by_abs + bh), (60, 60, 60), -1)
        cv2.rectangle(canvas, (bx, by_abs), (bx + int(bw * bloom), by_abs + bh),
                      (150, 130, 255), -1)
        cv2.rectangle(canvas, (bx, by_abs), (bx + bw, by_abs + bh), (150, 150, 150), 1)

        # ============ 文本层（缓存贴片） ============
        now = time.time()
        if self._dirty or self._top_layer is None \
                or now - self._dyn_ts > _DYN_PERIOD:
            self._build_top_layer(stats, cfg, cam_pw)
            self._build_bar_dyn(stats, cfg)
            self._dyn_ts = now
            self._dirty = False
        bgr, m3 = self._bar_dyn
        np.copyto(canvas[H - bar:], bgr, where=m3)
        bgr, m3 = self._top_layer
        np.copyto(canvas[:44], bgr, where=m3)
        return canvas

    # ------------------------------------------------------------------
    @staticmethod
    def draw_help(frame):
        """叠加帮助信息。"""
        lines = [
            "操作说明:",
            "  握拳 = 花苞   张开手掌 = 盛开",
            "  移动手掌     = 花茎随之摆动",
            "  拇指食指捏合 = 切换花色 (或按 C)",
            "  双手同屏     = 左右双花, 各手控制一朵 (附加功能)",
            "",
            "按键: S 平滑开关 | M 切换EMA/卡尔曼",
            "      P 手臂姿态 | H 帮助 | Q/ESC 退出",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (30, 150), (430, 150 + 34 * len(lines) + 20),
                      (20, 18, 16), -1)
        frame[:] = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
        draw_texts_cn(frame, [(ln, (46, 166 + i * 34), 19, _WHITE, False)
                              for i, ln in enumerate(lines)])
        return frame
