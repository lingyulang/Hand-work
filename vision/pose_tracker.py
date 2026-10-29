# -*- coding: utf-8 -*-
"""手臂骨骼检测（MediaPipe Pose，可选模块，--pose 启用）。

Pose 关键点中与手臂相关的编号：
  11/12 左右肩, 13/14 左右肘, 15/16 左右腕, 17-22 手部相关点
启用 Pose 会增加一定计算量，默认关闭以保证帧率。
"""
import cv2
import mediapipe as mp

MP_POSE = mp.solutions.pose
MP_DRAW = mp.solutions.drawing_utils
MP_STYLES = mp.solutions.drawing_styles

# 只绘制手臂链路：肩-肘-腕
ARM_CONNECTIONS = [
    (11, 13), (13, 15),          # 左臂
    (12, 14), (14, 16),          # 右臂
    (11, 12),                    # 双肩连线
]
ARM_IDS = {11, 12, 13, 14, 15, 16}


class PoseTracker:
    """封装 MediaPipe Pose，仅提取并绘制手臂骨骼。"""

    def __init__(self, model_complexity=1):
        self.pose = MP_POSE.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_and_draw(self, frame_bgr):
        """检测并把手臂骨骼画在帧上，返回画面帧。"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        if result.pose_landmarks:
            h, w = frame_bgr.shape[:2]
            pts = {}
            for idx in ARM_IDS:
                lm = result.pose_landmarks.landmark[idx]
                pts[idx] = (int(lm.x * w), int(lm.y * h))
            # 画手臂连线
            for a, b in ARM_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame_bgr, pts[a], pts[b], (255, 180, 60), 3, cv2.LINE_AA)
            for p in pts.values():
                cv2.circle(frame_bgr, p, 4, (90, 200, 255), -1, cv2.LINE_AA)
        return frame_bgr

    def close(self):
        self.pose.close()
