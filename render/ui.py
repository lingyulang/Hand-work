# -*- coding: utf-8 -*-
"""分屏 UI：左侧摄像头+骨骼连线，右侧五瓣花动画，底部信息栏。"""
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

    # ------------------------------------------------------------------
    def push_trail(self, center_norm):
        """记录手部中心轨迹点。"""
        self._trail.append(center_norm)
        self._trail_time.append(time.time())

    def clear_trail(self):
        self._trail.clear()

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
        jobs = []  # 文本批量绘制任务（性能关键：整帧只做一次 PIL 转换）

        # ============ 左侧：摄像头面板 ============
        # 面板底色 + 边框
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

        # 标题栏
        jobs.append(("摄像头画面 | 手部骨骼", (14, 12), 20, _WHITE, True))
        if stats.get("pose_on"):
            jobs.append(("手臂姿态: 开", (cam_pw - 130, 12), 16, _CYAN, False))

        # 未检测到手时的提示
        if stats.get("hand_lost"):
            tip = "未检测到手部，请将手置于摄像头前"
            tw = 320
            cv2.rectangle(canvas, (cam_pw // 2 - tw // 2 - 10, 340),
                          (cam_pw // 2 + tw // 2 + 10, 380), (30, 30, 30), -1)
            jobs.append((tip, (cam_pw // 2 - tw // 2 + 6, 348), 18, (120, 200, 255), False))

        # 分隔线
        cv2.line(canvas, (cam_pw, 0), (cam_pw, H - bar), (90, 90, 90), 2)

        # ============ 右侧：花朵面板 ============
        fx0 = cam_pw
        fh = flower_panel.shape[0]
        fy0 = (H - bar - fh) // 2
        canvas[fy0:fy0 + fh, fx0:fx0 + flower_panel.shape[1]] = flower_panel
        jobs.append(("五瓣花动画 | 隔空控制", (fx0 + 14, 12), 20, _WHITE, True))
        cv2.rectangle(canvas, (fx0, 0), (W - 1, H - bar - 1), (70, 70, 70), 1)

        # ============ 底部信息栏 ============
        canvas[H - bar:] = (18, 16, 14)
        cv2.line(canvas, (0, H - bar), (W, H - bar), (90, 90, 90), 2)

        fps = stats.get("fps", 0.0)
        lat = stats.get("latency_ms", 0.0)
        fps_color = (120, 220, 120) if fps >= 25 else (_ORANGE if fps >= 15 else _RED)
        jobs.append((f"FPS {fps:5.1f}", (16, H - bar + 12), 20, fps_color, True))
        lat_color = (120, 220, 120) if lat < 100 else _RED
        jobs.append((f"端到端延迟 {lat:4.0f} ms", (150, H - bar + 12), 20, lat_color, False))

        # 伸展度进度条
        bx, by, bw, bh = 400, H - bar + 14, 240, 22
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        bloom = stats.get("bloom", 0.0)
        cv2.rectangle(canvas, (bx, by), (bx + int(bw * bloom), by + bh), (150, 130, 255), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (150, 150, 150), 1)
        jobs.append((f"伸展度 {bloom:.2f}", (bx + 8, by + 1), 15, _WHITE, False))

        smooth_txt = f"平滑: {'开' if stats.get('smooth_on') else '关'} ({stats.get('smoother_name', '')})"
        jobs.append((smooth_txt, (680, H - bar + 12), 18,
                     (140, 230, 140) if stats.get('smooth_on') else _GRAY, False))

        # 状态与提示
        state = "盛开" if bloom > 0.8 else ("绽放中" if bloom > 0.25 else "花苞")
        nf = stats.get("n_flowers", 1)
        state_txt = f"状态: {state}" + (f" x{nf} (双手各控一朵)" if nf == 2 else "")
        jobs.append((state_txt, (680, H - bar + 36), 15, _CYAN, False))
        jobs.append(("[S]平滑开关 [M]算法 [C]花色 [P]手臂 [H]帮助 [Q]退出",
                     (16, H - bar + 38), 14, _GRAY, False))

        # 花色名称（右上角）
        jobs.append((f"花色: {stats.get('palette_name', '')}",
                     (W - 120, 12), 16, _CYAN, False))

        draw_texts_cn(canvas, jobs)
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
