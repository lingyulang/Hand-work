# -*- coding: utf-8 -*-
"""手部 21 关键点检测（MediaPipe Hands 封装）。

MediaPipe Hands 输出 21 个归一化关键点 (x, y, z)：
  0  手腕 WRIST
  1-4   拇指  (CMC, MCP, IP, TIP)
  5-8   食指  (MCP, PIP, DIP, TIP)
  9-12  中指
  13-16 无名指
  17-20 小指
"""
import cv2
import mediapipe as mp

# 骨骼连线定义（官方 HAND_CONNECTIONS），用于绘制骨架图
MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils
MP_STYLES = mp.solutions.drawing_styles


class HandTracker:
    """封装 MediaPipe Hands，提供 process / draw 接口。"""

    def __init__(self, max_num_hands=2, det_conf=0.6, track_conf=0.5):
        self.hands = MP_HANDS.Hands(
            static_image_mode=False,       # 视频流模式，启用内部追踪
            max_num_hands=max_num_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def process(self, frame_bgr):
        """输入 BGR 帧，返回 MultiHandLandmarks 列表（可能为空）。"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # 关闭写保护以允许 MediaPipe 零拷贝优化
        result = self.hands.process(rgb)
        return result.multi_hand_landmarks or []

    def draw(self, frame_bgr, hand_landmarks_list):
        """在帧上绘制 21 点骨架连线（就地修改并返回）。"""
        for lm in hand_landmarks_list:
            MP_DRAW.draw_landmarks(
                frame_bgr,
                lm,
                MP_HANDS.HAND_CONNECTIONS,
                MP_STYLES.get_default_hand_landmarks_style(),
                MP_STYLES.get_default_hand_connections_style(),
            )
        return frame_bgr

    def close(self):
        self.hands.close()
