# -*- coding: utf-8 -*-
"""手势 -> 动画参数映射器。

将视觉层提取的手势特征，经平滑滤波后映射为渲染层所需的动画参数：
  bloom     绽放度 0~1（平滑后，主参数）
  pos_x     花朵水平位置偏移 -1~1（跟随手部移动，体现运动轨迹）
  scale     花朵缩放（双手交互时由双手间距控制，单手时为 1）
  pinch_rise 捏合事件上升沿（用于切换花色）
"""
from dataclasses import dataclass

from .easing import SpringValue, approach
from .smoother import Smoother


@dataclass
class MappedParams:
    bloom: float = 0.0          # 渲染层使用的绽放度（已含缓动）
    bloom_target: float = 0.0   # 平滑后的目标值（用于监控显示）
    pos_x: float = 0.0          # -1~1 花朵水平偏移
    scale: float = 1.0          # 花朵整体缩放
    pinch_rise: bool = False    # 捏合上升沿事件
    hand_lost: bool = True      # 当前帧是否未检测到手


class GestureMapper:
    """串联 平滑滤波 -> 数值映射 -> 缓动 的完整逻辑链。"""

    def __init__(self, cfg, smoother: Smoother):
        self.cfg = cfg
        self.smoother = smoother
        self._pos = 0.0                    # 位置跟随（EMA）
        self._pinch_prev = False
        self._scale = 1.0
        self._spring = SpringValue(
            stiffness=cfg.SPRING_STIFFNESS, damping=cfg.SPRING_DAMPING, value=0.0
        )

    @property
    def smoother_name(self) -> str:
        return self.smoother.name

    def set_smoother(self, smoother: Smoother) -> None:
        self.smoother = smoother

    def update(self, feats, dt: float) -> MappedParams:
        """输入单帧手势特征，返回映射后的动画参数。"""
        out = MappedParams()
        out.hand_lost = feats.n_hands == 0

        if out.hand_lost:
            # 手离开画面：花朵缓缓回落至半开待机状态，位置回中
            target_bloom = 0.35
            self._pos = approach(self._pos, 0.0, 0.4, dt)
            out.pinch_rise = False
            self._pinch_prev = False
        else:
            # 1) 抖动消除：伸展度先经过 EMA/卡尔曼滤波
            smoothed = self.smoother.update(feats.openness_raw)
            target_bloom = smoothed

            # 2) 捏合上升沿检测（切换花色的手势指令，附加分）
            out.pinch_rise = feats.pinch and not self._pinch_prev
            self._pinch_prev = feats.pinch

            # 3) 位置跟随：手部中心 x (0~1) 映射为 -1~1
            self._pos = approach(
                self._pos, (feats.center[0] - 0.5) * 2.0, 0.15, dt
            )

            # 4) 双手交互：双手间距 -> 花朵缩放（附加分）
            if self.cfg.TWO_HAND_SCALE and feats.n_hands >= 2:
                # 间距 0.15~0.6 映射为缩放 0.5~1.6
                s = 0.5 + (feats.two_hand_dist - 0.15) / (0.6 - 0.15) * 1.1
                s = min(max(s, 0.5), 1.6)
                self._scale = approach(self._scale, s, 0.2, dt)
            else:
                self._scale = approach(self._scale, 1.0, 0.2, dt)

        out.bloom_target = target_bloom
        # 5) 弹簧缓动：目标值 -> 显示值，带轻微过冲，避免生硬切换
        out.bloom = self._spring.step(target_bloom, dt)
        out.bloom = min(max(out.bloom, 0.0), 1.0)
        out.pos_x = self._pos
        out.scale = self._scale
        return out
