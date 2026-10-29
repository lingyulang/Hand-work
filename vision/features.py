# -*- coding: utf-8 -*-
"""手势特征提取：将 21 个关键点量化为可映射的手势参数。

核心特征
--------
1. openness 伸展度 (0~1)：五指尖到掌心的平均距离 / 掌径，
   经线性映射与截断得到。握拳 -> 0（花苞），完全张开 -> 1（盛开）。
2. pinch 捏合：拇指尖-食指尖距离 / 掌径，带滞回判定（防止边界抖动）。
3. center 手部中心：手腕与中指根的中点，用于位置跟随与轨迹可视化。
4. n_hands 手数：用于双手交互（双手间距控制花朵大小）。

使用"掌径归一化"是为了消除手掌远近（成像大小）带来的尺度影响，
这是保证特征在用户前后移动时依然稳定的关键设计。
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# 关键点编号常量
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17


@dataclass
class HandFeature:
    """单只手的特征（双花模式：每只手独立驱动一朵花）。"""
    openness_raw: float = 0.0       # 伸展度 0~1
    pinch: bool = False             # 捏合状态（滞回后）
    center: Tuple[float, float] = (0.5, 0.5)  # 手部中心（归一化）


@dataclass
class GestureFeatures:
    """单帧手势特征结果。"""
    n_hands: int = 0
    openness_raw: float = 0.0        # 主手伸展度原始值 0~1（未平滑）
    center: Tuple[float, float] = (0.5, 0.5)   # 主手中心（归一化坐标）
    pinch: bool = False              # 主手是否处于捏合状态（滞回后）
    two_hand_dist: float = 0.0       # 双手中心间距（仅双手出现时有意义）
    centers: List[Tuple[float, float]] = field(default_factory=list)
    hands: List[HandFeature] = field(default_factory=list)  # 逐手特征


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def extract_features(hand_landmarks_list, cfg) -> GestureFeatures:
    """从 MediaPipe 关键点列表提取手势特征。

    Parameters
    ----------
    hand_landmarks_list : list[NormalizedLandmarkList]
        HandTracker.process() 的输出。
    cfg : module
        全局配置（OPENNESS_LOW / OPENNESS_HIGH / PINCH_* 等）。
    """
    feats = GestureFeatures()
    if not hand_landmarks_list:
        return feats

    feats.n_hands = len(hand_landmarks_list)

    # 逐手提取特征：掌径归一化 -> 伸展度 / 捏合 / 手心
    for lm in hand_landmarks_list:
        P = lm.landmark
        cx = (P[WRIST].x + P[MIDDLE_MCP].x) / 2.0
        cy = (P[WRIST].y + P[MIDDLE_MCP].y) / 2.0

        palm = _dist(P[WRIST], P[MIDDLE_MCP])
        if palm < 1e-6:
            feats.hands.append(HandFeature(0.0, False, (cx, cy)))
            continue

        tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        spread = sum(_dist(P[t], P[WRIST]) for t in tips) / (len(tips) * palm)
        low, high = cfg.OPENNESS_LOW, cfg.OPENNESS_HIGH
        openness = float(np.clip((spread - low) / max(high - low, 1e-6), 0.0, 1.0))
        pinch_ratio = _dist(P[THUMB_TIP], P[INDEX_TIP]) / palm
        feats.hands.append(HandFeature(openness, pinch_ratio < cfg.PINCH_ON, (cx, cy)))

    feats.centers = [h.center for h in feats.hands]

    # 主手 = 画面中位置最低（y 最大）的一只，兼容单花场景
    if feats.hands:
        main = max(feats.hands, key=lambda h: h.center[1])
        feats.center = main.center
        feats.openness_raw = main.openness_raw
        feats.pinch = main.pinch

    # 双手间距（归一化坐标下的欧氏距离）
    if feats.n_hands >= 2:
        (x1, y1), (x2, y2) = feats.centers[0], feats.centers[1]
        feats.two_hand_dist = float(np.hypot(x1 - x2, y1 - y2))

    return feats
