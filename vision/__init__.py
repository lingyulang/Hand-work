# -*- coding: utf-8 -*-
"""视觉层：视频采集、骨骼关键点检测、手势特征提取。

子模块采用惰性导入：仅 features 等纯计算模块可在无 mediapipe
环境下独立使用（便于单元测试）。
"""
from .camera import CameraStream
from .features import GestureFeatures, extract_features


def __getattr__(name):
    # 惰性加载重量级模块（依赖 mediapipe）
    if name == "HandTracker":
        from .hand_tracker import HandTracker
        return HandTracker
    if name == "PoseTracker":
        from .pose_tracker import PoseTracker
        return PoseTracker
    raise AttributeError(f"module 'vision' has no attribute {name!r}")


__all__ = [
    "CameraStream",
    "HandTracker",
    "PoseTracker",
    "GestureFeatures",
    "extract_features",
]
