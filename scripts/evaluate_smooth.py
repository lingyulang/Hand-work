# -*- coding: utf-8 -*-
"""平滑算法对比评估（无需摄像头，可离线运行）。

方法
----
以"手部伸展度"信号为对象，构造合成信号：
    真实动作(低频开合) + 手部生理性震颤(6~12Hz 高频抖动) + 检测噪声
分别通过 关闭/EMA/卡尔曼 三种滤波器，输出：
    1. 时域对比曲线图        docs/平滑对比曲线.png
    2. 定量指标表(打印+CSV)  帧间抖动 diff_mean、与真实值RMSE(延迟代价)

运行:
    python scripts/evaluate_smooth.py
"""
import csv
import math
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")  # 无窗口环境
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
from logic.smoother import EMASmoother, KalmanSmoother1D, PassthroughSmoother, jitter_metrics  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def make_signal(n=600, fps=30):
    """合成手部伸展度信号：慢速开合 + 生理震颤 + 检测噪声。"""
    random.seed(2026)
    dt = 1.0 / fps
    truth, noisy = [], []
    for i in range(n):
        t = i * dt
        # 真实动作：0.25Hz 的两次开合 + 缓慢漂移
        base = 0.5 + 0.35 * math.sin(2 * math.pi * 0.25 * t) + 0.05 * math.sin(2 * math.pi * 0.08 * t)
        base = min(max(base, 0.0), 1.0)
        # 生理性震颤：8Hz 主频 + 11Hz 次频，幅度约 0.02
        tremor = 0.018 * math.sin(2 * math.pi * 8 * t) + 0.010 * math.sin(2 * math.pi * 11 * t + 1.3)
        # 检测随机噪声
        noise = random.gauss(0, 0.022)
        truth.append(base)
        noisy.append(min(max(base + tremor + noise, 0.0), 1.0))
    return truth, noisy


def rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(out_dir, exist_ok=True)

    truth, noisy = make_signal()
    filters = {
        "关闭(直通)": PassthroughSmoother(),
        f"EMA(α={cfg.EMA_ALPHA})": EMASmoother(alpha=cfg.EMA_ALPHA),
        f"Kalman(Q={cfg.KALMAN_Q:.0e},R={cfg.KALMAN_R:.0e})": KalmanSmoother1D(cfg.KALMAN_Q, cfg.KALMAN_R),
    }

    results = {}
    for name, f in filters.items():
        results[name] = [f.update(z) for z in noisy]

    # ---------- 定量指标 ----------
    j_raw = jitter_metrics(noisy)
    rows = []
    print(f"{'滤波器':<28}{'帧间抖动':>10}{'抖动降低':>10}{'RMSE':>10}")
    print("-" * 60)
    for name, sig in results.items():
        j = jitter_metrics(sig)
        err = rmse(sig, truth)
        reduce_pct = (1 - j["diff_mean"] / j_raw["diff_mean"]) * 100
        rows.append([name, f"{j['diff_mean']:.4f}", f"{reduce_pct:.1f}%", f"{err:.4f}"])
        print(f"{name:<28}{j['diff_mean']:>10.4f}{reduce_pct:>9.1f}%{err:>10.4f}")

    with open(os.path.join(out_dir, "平滑对比指标.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["滤波器", "帧间抖动(diff_mean)", "抖动降低比例", "相对真实值RMSE"])
        w.writerows(rows)

    # ---------- 时域曲线 ----------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = [i / 30.0 for i in range(len(noisy))]
    panels = list(results.items())
    for ax, (name, sig) in zip(axes, panels):
        ax.plot(t, noisy, lw=0.6, color="#bbbbbb", label="原始(未平滑)")
        ax.plot(t, sig, lw=1.6, color="#d94f6b", label=f"平滑后: {name}")
        ax.plot(t, truth, lw=1.0, ls="--", color="#3a7bd5", label="真实动作")
        ax.set_ylabel("伸展度")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{name}", fontsize=11, loc="left")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("平滑前后伸展度信号对比（合成信号：慢速开合 + 8~11Hz 生理震颤 + 检测噪声）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(out_dir, "平滑对比曲线.png")
    fig.savefig(out_png, dpi=150)

    # ---------- 局部放大：抖动细节 ----------
    fig2, ax = plt.subplots(figsize=(10, 4))
    seg = slice(200, 320)
    ax.plot(t[seg], noisy[seg], lw=0.8, color="#999999", marker=".", ms=2, label="原始(未平滑)")
    ax.plot(t[seg], results["关闭(直通)"][seg], lw=1.0, color="#999999", alpha=0.5)
    for (name, sig), color in zip(panels[1:], ["#e08214", "#3a7bd5"]):
        ax.plot(t[seg], sig[seg], lw=2.0, color=color, label=f"平滑后: {name}")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("伸展度")
    ax.set_title("局部放大：高频震颤被显著抑制，动作趋势完整保留", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png2 = os.path.join(out_dir, "平滑对比局部放大.png")
    fig2.savefig(out_png2, dpi=150)

    print(f"\n图表已保存:\n  {out_png}\n  {out_png2}")
    print(f"指标已保存: {os.path.join(out_dir, '平滑对比指标.csv')}")


if __name__ == "__main__":
    main()
