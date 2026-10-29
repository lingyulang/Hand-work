# -*- coding: utf-8 -*-
"""缓动函数库（动画过渡考核点）。

花瓣绽放不做生硬切换，而是让参数经过缓动曲线映射，
获得自然的加速-减速视觉节奏。
"""
import math


def ease_in_out_cubic(x: float) -> float:
    """三次缓入缓出：花瓣几何形态的主映射，中段绽放速度最快。"""
    x = min(max(x, 0.0), 1.0)
    if x < 0.5:
        return 4.0 * x ** 3
    return 1.0 - (-2.0 * x + 2.0) ** 3 / 2.0


def ease_out_cubic(x: float) -> float:
    """三次缓出：起步快、收尾柔，适合花瓣展开的角度过渡。"""
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def ease_out_back(x: float, s: float = 1.70158) -> float:
    """回弹缓出：终点处轻微过冲回落，模拟花瓣张开的弹性。"""
    x = min(max(x, 0.0), 1.0)
    x1 = x - 1.0
    return 1.0 + (s + 1.0) * x1 ** 3 + s * x1 ** 2


def approach(current: float, target: float, tau: float, dt: float) -> float:
    """帧率无关的指数趋近（一阶惯性环节）。

    current += (target - current) * (1 - exp(-dt/tau))
    tau 越小越跟手，越大越柔和。以 dt 归一化保证不同帧率下观感一致。
    """
    if tau <= 0 or dt <= 0:
        return target
    return current + (target - current) * (1.0 - math.exp(-dt / tau))


class SpringValue:
    """弹簧-阻尼数值动画：产生自然的轻微过冲，用于绽放的灵动感。

    二阶系统:  a = k*(target - x) - c*v
    """

    def __init__(self, stiffness=120.0, damping=14.0, value=0.0):
        self.k = stiffness
        self.c = damping
        self.x = value
        self.v = 0.0

    def step(self, target: float, dt: float) -> float:
        dt = min(dt, 0.05)  # 防止卡顿导致的数值发散
        # 半隐式欧拉积分，稳定性优于显式欧拉
        a = self.k * (target - self.x) - self.c * self.v
        self.v += a * dt
        self.x += self.v * dt
        return self.x

    def set(self, value: float) -> None:
        self.x = value
        self.v = 0.0
