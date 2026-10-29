# -*- coding: utf-8 -*-
"""手臂骨骼检测（MediaPipe Pose 轻量化封装，默认开启）。

性能设计（保证手+臂同开时 FPS >= 20）：
  1. model_complexity=0 轻量模型；
  2. 推理输入降采样到 ~320px 宽（关键点为归一化坐标，绘制时直接映射全分辨率）；
  3. 隔帧推理（POSE_CADENCE），中间帧复用上一次手臂点做廉价重绘——
     手臂运动速度远低于手部，15Hz 骨骼视觉上完全连续。

Pose 关键点编号：11/12 左右肩, 13/14 左右肘, 15/16 左右腕。
"""
import cv2
import mediapipe as mp

MP_POSE = mp.solutions.pose

# 只绘制手臂链路：肩-肘-腕
ARM_CONNECTIONS = [
    (11, 13), (13, 15),          # 左臂
    (12, 14), (14, 16),          # 右臂
    (11, 12),                    # 双肩连线
]
ARM_IDS = (11, 12, 13, 14, 15, 16)


class PoseTracker:
    """封装 MediaPipe Pose，仅提取并绘制手臂骨骼（轻量化推理）。"""

    def __init__(self, model_complexity=0, input_w=320, cadence=2):
        self.pose = MP_POSE.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.input_w = input_w
        self.cadence = max(1, cadence)
        self._frame_idx = 0
        self._last_pts = None   # {id: (x, y)} 全分辨率像素坐标缓存

    def _infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        iw = min(self.input_w, w)
        ih = max(1, int(h * iw / w))
        small = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        if result.pose_landmarks:
            self._last_pts = {
                idx: (int(result.pose_landmarks.landmark[idx].x * w),
                      int(result.pose_landmarks.landmark[idx].y * h))
                for idx in ARM_IDS
            }

    def process_and_draw(self, frame_bgr):
        """检测并把手臂骨骼画在帧上，返回画面帧（隔帧推理+复用缓存）。"""
        if self._frame_idx % self.cadence == 0 or self._last_pts is None:
            self._infer(frame_bgr)
        self._frame_idx += 1

        if self._last_pts:
            pts = self._last_pts
            for a, b in ARM_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame_bgr, pts[a], pts[b],
                             (255, 180, 60), 3, cv2.LINE_AA)
            for p in pts.values():
                cv2.circle(frame_bgr, p, 4, (90, 200, 255), -1, cv2.LINE_AA)
        return frame_bgr

    def close(self):
        self.pose.close()
