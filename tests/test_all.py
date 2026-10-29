# -*- coding: utf-8 -*-
"""单元测试（无摄像头依赖，纯逻辑验证）。

运行方式:
    cd gesture_flower && python tests/test_all.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
from logic.easing import (SpringValue, approach, ease_in_out_cubic,  # noqa: E402
                          ease_out_back, ease_out_cubic)
from logic.mapper import GestureMapper  # noqa: E402
from logic.smoother import (EMASmoother, KalmanSmoother1D,  # noqa: E402
                            PassthroughSmoother, jitter_metrics, make_smoother)
from render.flower import FlowerRenderer  # noqa: E402
from vision.features import GestureFeatures, extract_features  # noqa: E402


# ---------------- 平滑器 ----------------
def test_ema_converges():
    s = EMASmoother(alpha=0.5)
    out = [s.update(1.0) for _ in range(60)]
    assert abs(out[-1] - 1.0) < 0.01, "EMA 应收敛到常值输入"
    s2 = EMASmoother(alpha=0.3)
    ramp = [s2.update(v) for v in [0.0] * 10 + [1.0] * 60]
    assert ramp[9] < 0.05 and abs(ramp[-1] - 1.0) < 0.02, "阶跃响应应平滑收敛"
    print("  [通过] EMA 收敛性")


def test_kalman_converges_and_smooths():
    import random
    random.seed(42)
    k = KalmanSmoother1D(q=2.5e-3, r=4.0e-2)
    truth, noisy, filtered = 0.7, [], []
    for _ in range(200):
        z = truth + random.gauss(0, 0.05)
        noisy.append(z)
        filtered.append(k.update(z))
    j_noisy = jitter_metrics(noisy)
    j_filt = jitter_metrics(filtered)
    assert abs(filtered[-1] - truth) < 0.05, "卡尔曼应无偏收敛"
    assert j_filt["diff_mean"] < j_noisy["diff_mean"] * 0.5, \
        f"卡尔曼应显著降低帧间抖动: {j_filt['diff_mean']:.4f} vs {j_noisy['diff_mean']:.4f}"
    print(f"  [通过] 卡尔曼: 帧间抖动 {j_noisy['diff_mean']:.4f} -> {j_filt['diff_mean']:.4f}")


def test_passthrough():
    p = PassthroughSmoother()
    assert p.update(0.123) == 0.123
    print("  [通过] 直通滤波器")


def test_factory():
    assert isinstance(make_smoother("ema", cfg), EMASmoother)
    assert isinstance(make_smoother("kalman", cfg), KalmanSmoother1D)
    assert isinstance(make_smoother("off", cfg), PassthroughSmoother)
    print("  [通过] 平滑器工厂")


# ---------------- 缓动 ----------------
def test_easing():
    assert abs(ease_in_out_cubic(0.0)) < 1e-9
    assert abs(ease_in_out_cubic(1.0) - 1.0) < 1e-9
    assert abs(ease_in_out_cubic(0.5) - 0.5) < 1e-9
    assert ease_out_cubic(0.5) > 0.5, "ease_out 前半程应更快"
    b = ease_out_back(1.0)
    assert abs(b - 1.0) < 1e-9
    peak = max(ease_out_back(i / 50) for i in range(51))
    assert peak > 1.0, "ease_out_back 应出现过冲"
    assert abs(approach(0.0, 1.0, 0.1, 100.0) - 1.0) < 1e-6, "tau 远小于 dt 时应到位"
    s = SpringValue(120, 14, 0.0)
    for _ in range(400):
        s.step(1.0, 1 / 60)
    assert abs(s.x - 1.0) < 0.01, "弹簧应收敛"
    print("  [通过] 缓动函数与弹簧")


# ---------------- 特征提取（合成关键点） ----------------
def _synth_landmarks(spread_factor):
    """构造合成手部关键点：spread_factor=0 握拳, 1 完全张开。

    掌心位于原点附近，指尖沿径向分布。wrist 在 (0, +palm)，中指根在 (0, 0)。
    """
    import numpy as np

    class LM:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Hand:
        def __init__(self, lms):
            self.landmark = lms

    palm = 0.15  # 掌径
    wrist = LM(0.5, 0.5 + palm)
    mcp_ids = {5: (-0.6, 0.0), 9: (0.0, 0.0), 13: (0.6, 0.0), 17: (1.2, 0.1)}
    lms = [LM(0.5, 0.5 + palm)] * 21
    lms[0] = wrist
    # 食/中/无名/小指 MCP
    for idx, (dx, dy) in mcp_ids.items():
        lms[idx] = LM(0.5 + dx * palm, 0.5 + dy * palm)
    # 指尖：握拳时折回掌心（距离 ~0.55*palm），张开时伸直（~1.6*palm）
    tip_dist = palm * (0.55 + (1.6 - 0.55) * spread_factor)
    tips = {4: (-1.2, -0.55), 8: (-0.6, -1.0), 12: (0.0, -1.05),
            16: (0.6, -1.0), 20: (1.2, -0.9)}
    for idx, (dx, dy) in tips.items():
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        lms[idx] = LM(0.5 + ux * tip_dist, 0.5 + palm + uy * tip_dist)
    return [Hand(lms)]


def test_features_fist_vs_open():
    fist = extract_features(_synth_landmarks(0.0), cfg)
    open_hand = extract_features(_synth_landmarks(1.0), cfg)
    assert fist.n_hands == 1 and open_hand.n_hands == 1
    assert fist.openness_raw < 0.15, f"握拳伸展度应接近0, 实际 {fist.openness_raw:.3f}"
    assert open_hand.openness_raw > 0.85, f"张开伸展度应接近1, 实际 {open_hand.openness_raw:.3f}"
    empty = extract_features([], cfg)
    assert empty.n_hands == 0 and empty.openness_raw == 0.0
    print(f"  [通过] 特征提取: 握拳={fist.openness_raw:.3f}, 张开={open_hand.openness_raw:.3f}")


def test_mapper_pipeline():
    cfg_local = cfg
    m = GestureMapper(cfg_local, make_smoother("kalman", cfg_local))
    # 手离开画面
    empty = GestureFeatures(n_hands=0)
    out = m.update(empty, 0.033)
    assert out.hand_lost
    # 手出现并张开 -> bloom 逐渐上升
    for _ in range(120):
        out = m.update(extract_features(_synth_landmarks(1.0), cfg_local), 0.033)
    assert out.bloom > 0.8, f"持续张开应驱动 bloom 上升, 实际 {out.bloom:.3f}"
    # 握拳 -> bloom 下降
    for _ in range(120):
        out = m.update(extract_features(_synth_landmarks(0.0), cfg_local), 0.033)
    assert out.bloom < 0.2, f"持续握拳应驱动 bloom 下降, 实际 {out.bloom:.3f}"
    print(f"  [通过] 映射链路: 张开 bloom={out.bloom:.2f} 后握拳回落")


# ---------------- 渲染器（离线可跑） ----------------
def test_flower_render_states():
    r = FlowerRenderer(w=320, h=330)  # 小尺寸加快测试
    for b in (0.0, 0.5, 1.0):
        img = r.render(b, pos_x=0.0, scale=1.0, t=1.0)
        assert img.shape == (330, 320, 3), "输出尺寸应与面板一致"
        assert img.dtype == np.uint8
        assert img.std() > 10, "画面不应是纯黑"
    # 缩放 <1（双手靠近）不得因 int 截断出现 0 尺寸（曾致 cv2.resize 崩溃）
    for s in (0.5, 0.6, 1.6):
        img = r.render(0.9, pos_x=0.0, scale=s, t=1.0)
        assert img.std() > 10, f"scale={s} 时渲染不应异常"
    for _ in range(4):
        name = r.cycle_palette()
    assert isinstance(name, str), "花色切换应返回名称"
    print("  [通过] 五瓣花渲染器三种状态")


if __name__ == "__main__":
    import numpy as np  # noqa: F401  (供测试环境)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"运行 {len(tests)} 项测试:")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  [失败] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [异常] {t.__name__}: {type(e).__name__}: {e}")
    print(f"结果: {len(tests) - failed}/{len(tests)} 通过")
    sys.exit(1 if failed else 0)
