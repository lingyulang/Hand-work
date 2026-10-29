# -*- coding: utf-8 -*-
"""手势 -> 动画参数映射器（支持单花/双花两种模式）。

将视觉层提取的手势特征，经平滑滤波后映射为渲染层所需的动画参数。

模式
----
* 单手/无手  : 一朵主花居中，跟随主手开合与水平位置；
* 双手同屏  : 左右各一朵花，按手心 x 排序分配（左手->左花，右手->右花），
              每只手拥有独立的平滑器与弹簧缓动，互不干扰。

平滑器以"槽位"为单位持有（copy.deepcopy 克隆出第二份），
S/M 键切换算法时两朵花同步切换。
"""
import copy
from dataclasses import dataclass, field
from typing import List

from .easing import SpringValue, approach
from .smoother import Smoother

DOUBLE_FLOWER_SCALE = 0.75   # 双花模式下每朵花的相对缩放
DOUBLE_HOME_A = -0.45        # 左花基准位置（-1~1）
DOUBLE_HOME_B = 0.45         # 右花基准位置


@dataclass
class FlowerState:
    """一朵花的动画参数。"""
    bloom: float = 0.0          # 绽放度（平滑+弹簧缓动后）
    bloom_target: float = 0.0   # 平滑后的目标值（监控显示）
    pos_x: float = 0.0          # -1~1 水平位置
    scale: float = 1.0          # 相对面板的缩放
    hand_lost: bool = True      # 该槽位当前是否无手


@dataclass
class MappedParams:
    """单帧映射结果。"""
    n_flowers: int = 1
    flowers: List[FlowerState] = field(
        default_factory=lambda: [FlowerState(), FlowerState()])
    pinch_rise: bool = False    # 任一只手捏合的上升沿事件
    hand_lost: bool = True      # 画面中是否完全没有手
    # ---- 主花兼容字段（= flowers[0]，便于单花场景代码不变） ----
    bloom: float = 0.0
    bloom_target: float = 0.0
    pos_x: float = 0.0
    scale: float = 1.0

    @property
    def render_items(self):
        """渲染层输入: [(bloom, pos_x, scale_rel), ...]"""
        if self.n_flowers == 2:
            return [(f.bloom, f.pos_x, f.scale) for f in self.flowers]
        return [(self.flowers[0].bloom, self.flowers[0].pos_x,
                 self.flowers[0].scale)]


class _Slot:
    """一朵花的滤波/缓动状态。"""

    def __init__(self, smoother: Smoother):
        self.smoother = smoother
        self.pos = 0.0
        self.spring = SpringValue(stiffness=0, damping=0, value=0.0)
        # stiffness/damping 在 mapper 中统一按 cfg 赋值


class GestureMapper:
    """串联 平滑滤波 -> 数值映射 -> 缓动 的完整逻辑链（双槽位）。"""

    def __init__(self, cfg, smoother: Smoother):
        self.cfg = cfg
        self._spring_cfg = (cfg.SPRING_STIFFNESS, cfg.SPRING_DAMPING)
        self.smoother = smoother          # 对外保留主花平滑器引用
        self._slots = [self._new_slot(smoother),
                       self._new_slot(copy.deepcopy(smoother))]
        self._pinch_prev = False

    def _new_slot(self, smoother):
        st, dp = self._spring_cfg
        s = _Slot(smoother)
        s.spring = SpringValue(stiffness=st, damping=dp, value=0.0)
        return s

    @property
    def smoother_name(self) -> str:
        return self.smoother.name

    def set_smoother(self, smoother: Smoother) -> None:
        """切换平滑算法（两朵花同步使用同类新实例）。"""
        self.smoother = smoother
        self._slots[0].smoother = smoother
        self._slots[1].smoother = copy.deepcopy(smoother)

    def _step_slot(self, slot, target_bloom, home_x, follow_x, dt, lost):
        """单槽位更新：平滑 -> 弹簧 -> 位置跟随。"""
        if lost:
            target = 0.35                       # 待机半开
            slot.pos = approach(slot.pos, home_x * 0.35, 0.4, dt)
        else:
            target = slot.smoother.update(target_bloom)
            slot.pos = approach(slot.pos, follow_x, 0.15, dt)
        slot.bloom_target = target
        b = slot.spring.step(target, dt)
        slot.bloom = min(max(b, 0.0), 1.0)
        slot.hand_lost = lost

    def update(self, feats, dt: float) -> MappedParams:
        """输入单帧手势特征，返回映射后的动画参数。"""
        out = MappedParams()
        out.hand_lost = feats.n_hands == 0
        a, b_slot = self._slots

        if feats.n_hands == 0:
            # 无手：主花待机居中，副花回落隐藏位
            out.n_flowers = 1
            self._step_slot(a, 0.0, 0.0, 0.0, dt, lost=True)
            self._step_slot(b_slot, 0.0, DOUBLE_HOME_B, DOUBLE_HOME_B, dt, lost=True)
            out.pinch_rise = False
            self._pinch_prev = False
        elif feats.n_hands == 1:
            # 单手：主花全幅跟随
            out.n_flowers = 1
            self._step_slot(a, feats.openness_raw, 0.0,
                            (feats.center[0] - 0.5) * 2.0, dt, lost=False)
            self._step_slot(b_slot, 0.0, DOUBLE_HOME_B, DOUBLE_HOME_B, dt, lost=True)
            out.pinch_rise = feats.pinch and not self._pinch_prev
            self._pinch_prev = feats.pinch
        else:
            # 双手：按手心 x 从左到右分配给左花/右花（各手独立平滑）
            out.n_flowers = 2
            hands = sorted(feats.hands, key=lambda h: h.center[0])
            self._step_slot(a, hands[0].openness_raw, DOUBLE_HOME_A,
                            DOUBLE_HOME_A + (hands[0].center[0] - 0.5) * 0.5,
                            dt, lost=False)
            self._step_slot(b_slot, hands[1].openness_raw, DOUBLE_HOME_B,
                            DOUBLE_HOME_B + (hands[1].center[0] - 0.5) * 0.5,
                            dt, lost=False)
            out.pinch_rise = any(h.pinch for h in hands) and not self._pinch_prev
            self._pinch_prev = any(h.pinch for h in hands)

        for slot, st in zip(self._slots, out.flowers):
            st.bloom, st.bloom_target = slot.bloom, slot.bloom_target
            st.pos_x, st.hand_lost = slot.pos, slot.hand_lost
            st.scale = DOUBLE_FLOWER_SCALE if out.n_flowers == 2 else 1.0

        # 主花兼容字段
        main = out.flowers[0]
        out.bloom, out.bloom_target = main.bloom, main.bloom_target
        out.pos_x, out.scale = main.pos_x, main.scale
        return out
