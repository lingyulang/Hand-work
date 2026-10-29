# -*- coding: utf-8 -*-
"""抖动消除滤波器（核心考核点）。

提供两种可切换的平滑算法：
  1. EMASmoother      指数移动平均：实现简单、计算量为 O(1)，
                      alpha 越小平滑越强、延迟越大。
  2. KalmanSmoother1D 一维卡尔曼滤波（常值运动模型）：
                      基于最小方差估计，在"跟手性"与"平滑度"之间
                      可以通过 Q/R 独立调节，理论上是噪声已知时的
                      最优线性滤波器。

两种滤波器均支持 reset()，切换平滑开关（S 键）时直通原始值，
便于现场演示"开启/关闭平滑"的视觉差异。
"""
import math
from typing import Protocol


class Smoother(Protocol):
    """平滑器统一接口。"""

    def update(self, value: float) -> float: ...
    def reset(self) -> None: ...
    @property
    def name(self) -> str: ...


class EMASmoother:
    """指数移动平均: y_k = (1-a)*y_{k-1} + a*x_k"""

    def __init__(self, alpha=0.35):
        assert 0 < alpha <= 1.0, "alpha 必须在 (0, 1]"
        self.alpha = alpha
        self._y = None

    def update(self, value: float) -> float:
        if self._y is None:
            self._y = value
        else:
            self._y = (1.0 - self.alpha) * self._y + self.alpha * value
        return self._y

    def reset(self) -> None:
        self._y = None

    @property
    def name(self) -> str:
        return f"EMA(a={self.alpha:.2f})"


class KalmanSmoother1D:
    """一维卡尔曼滤波（常值模型）。

    预测:  x_pred = x;           P_pred = P + Q
    更新:  K = P_pred/(P_pred+R); x = x_pred + K*(z - x_pred); P = (1-K)*P_pred
    """

    def __init__(self, q=2.5e-3, r=4.0e-2):
        self.q = q   # 过程噪声方差：手真实运动的剧烈程度
        self.r = r   # 观测噪声方差：检测抖动大小
        self._x = None
        self._p = 1.0

    def update(self, z: float) -> float:
        if self._x is None:
            self._x = z
            self._p = self.r
            return self._x
        # ---- 预测 ----
        p_pred = self._p + self.q
        # ---- 更新 ----
        k = p_pred / (p_pred + self.r)
        self._x = self._x + k * (z - self._x)
        self._p = (1.0 - k) * p_pred
        return self._x

    def reset(self) -> None:
        self._x = None
        self._p = 1.0

    @property
    def name(self) -> str:
        return f"Kalman(Q={self.q:.1e},R={self.r:.1e})"


class PassthroughSmoother:
    """直通滤波器（关闭平滑时使用，作为对比基准）。"""

    def update(self, value: float) -> float:
        return value

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "关闭(直通)"


def make_smoother(kind: str, cfg) -> Smoother:
    """工厂函数：按配置创建平滑器。kind: 'kalman' | 'ema' | 'off'"""
    kind = (kind or "kalman").lower()
    if kind == "ema":
        return EMASmoother(alpha=cfg.EMA_ALPHA)
    if kind == "off":
        return PassthroughSmoother()
    return KalmanSmoother1D(q=cfg.KALMAN_Q, r=cfg.KALMAN_R)


def jitter_metrics(signal):
    """抖动量化：一阶差分绝对值均值（相邻帧波动强度）与标准差。

    用于评估报告中对"平滑前后"的客观对比。
    """
    n = len(signal)
    if n < 3:
        return {"diff_mean": 0.0, "std": 0.0}
    diffs = [abs(signal[i] - signal[i - 1]) for i in range(1, n)]
    mean = sum(signal) / n
    var = sum((v - mean) ** 2 for v in signal) / n
    return {"diff_mean": sum(diffs) / len(diffs), "std": math.sqrt(var)}
