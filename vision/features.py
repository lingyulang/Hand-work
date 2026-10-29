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
class GestureFeatures:
    """单帧手势特征结果。"""
    n_hands: int = 0
    openness_raw: float = 0.0        # 伸展度原始值 0~1（未平滑）
    center: Tuple[float, float] = (0.5, 0.5)   # 主手中心（归一化坐标）
    pinch: bool = False              # 是否处于捏合状态（滞回后）
    two_hand_dist: float = 0.0       # 双手中心间距（仅双手出现时有意义）
    centers: List[Tuple[float, float]] = field(default_factory=list)


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

    # 逐手计算中心，主手取图像中靠下（离摄像头近/自然下垂态更稳）的一只，
    # 双手交互时另用两手间距。
    centers = []
    for lm in hand_landmarks_list:
        cx = (lm.landmark[WRIST].x + lm.landmark[MIDDLE_MCP].x) / 2.0
        cy = (lm.landmark[WRIST].y + lm.landmark[MIDDLE_MCP].y) / 2.0
        centers.append((cx, cy))
    feats.centers = centers
    feats.center = max(centers, key=lambda c: c[1])  # 取 y 最大（画面下方）为主手
    idx_main = centers.index(feats.center)

    lm = hand_landmarks_list[idx_main]
    P = lm.landmark

    # 掌径：手腕 -> 中指根，作为尺度基准
    palm = _dist(P[WRIST], P[MIDDLE_MCP])
    if palm < 1e-6:
        return feats

    # 五指尖到掌心（手腕）的归一化平均距离 -> 伸展度原始值
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    spread = sum(_dist(P[t], P[WRIST]) for t in tips) / (len(tips) * palm)

    # 线性映射 [OPENNESS_LOW, OPENNESS_HIGH] -> [0, 1]，越界截断
    low, high = cfg.OPENNESS_LOW, cfg.OPENNESS_HIGH
    openness = (spread - low) / max(high - low, 1e-6)
    feats.openness_raw = float(np.clip(openness, 0.0, 1.0))

    # 捏合判定（滞回）：dist(拇指尖, 食指尖)/掌径
    pinch_ratio = _dist(P[THUMB_TIP], P[INDEX_TIP]) / palm
    feats.pinch = pinch_ratio < cfg.PINCH_ON

    # 双手间距（归一化坐标下的欧氏距离）
    if feats.n_hands >= 2:
        (x1, y1), (x2, y2) = centers[0], centers[1]
        feats.two_hand_dist = float(np.hypot(x1 - x2, y1 - y2))

    return feats
