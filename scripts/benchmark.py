# -*- coding: utf-8 -*-
"""离线性能基准测试（无需摄像头）。

用合成帧实测各阶段处理耗时，输出统计与 CSV：
  - MediaPipe Hands 推理耗时
  - 手势特征提取耗时
  - 平滑+映射耗时
  - 花朵渲染耗时（含中文文本）
  - 单帧总耗时 -> FPS 上限与延迟估算

运行:  python scripts/benchmark.py [--frames 300]
"""
import argparse
import csv
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
from logic.mapper import GestureMapper  # noqa: E402
from logic.smoother import make_smoother  # noqa: E402
from render.flower import FlowerRenderer  # noqa: E402
from render.text_cn import draw_text_cn  # noqa: E402
from vision.features import extract_features  # noqa: E402
from vision.hand_tracker import HandTracker  # noqa: E402


def make_synthetic_frame(t):
    """生成带纹理与肤色块的合成帧，模拟真实画面复杂度。"""
    rng = np.random.RandomState(int(t * 30) % 1000)
    frame = rng.randint(0, 255, (cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    # 肤色椭圆模拟手部
    cx = int(cfg.FRAME_WIDTH / 2 + 100 * np.sin(t * 2))
    cy = int(cfg.FRAME_HEIGHT / 2 + 60 * np.cos(t * 1.5))
    cv2.ellipse(frame, (cx, cy), (70, 95), 20 * np.sin(t), 0, 360, (110, 150, 200), -1)
    return frame


def bench(times_dict, name, fn, n):
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        ts.append((time.perf_counter() - t0) * 1000)
    times_dict[name] = ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()
    n = args.frames

    tracker = HandTracker(cfg.HAND_MAX_NUM, cfg.HAND_DET_CONF, cfg.HAND_TRACK_CONF)
    mapper = GestureMapper(cfg, make_smoother("kalman", cfg))
    flower = FlowerRenderer(w=cfg.WINDOW_W - cfg.CAM_PANEL_W, h=cfg.WINDOW_H - cfg.INFO_BAR_H)

    frames = [make_synthetic_frame(i / 30) for i in range(n)]
    times = {}

    # 1. 推理
    def run_infer(i):
        tracker.process(frames[i])
    bench(times, "hands_infer", run_infer, n)

    # 2. 特征
    results = [tracker.process(f) for f in frames[:10]]
    feats_list = [extract_features(r or [], cfg) for r in results]
    def run_feat(i):
        extract_features(results[i % 10] or [], cfg)
    bench(times, "features", run_feat, n)

    # 3. 平滑+映射
    def run_map(i):
        mapper.update(feats_list[i % 10], 1 / 30)
    bench(times, "smooth_map", run_map, n)

    # 4. 渲染
    blooms = np.linspace(0, 1, n)
    def run_render(i):
        img = flower.render(blooms[i], 0.0, 1.0, i / 30)
        draw_text_cn(img, "FPS 30.0  延迟 50 ms", (10, 620), 18)
    bench(times, "render", run_render, n)

    total = [a + b + c + d for a, b, c, d in
             zip(times["hands_infer"], times["features"],
                 times["smooth_map"], times["render"])]

    # 输出
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    with open(os.path.join(out_dir, "benchmark.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["阶段", "均值ms", "中位数ms", "P95ms", "最大ms"])
        rows = []
        for name, ts in list(times.items()) + [("total_per_frame", total)]:
            row = [name, f"{statistics.mean(ts):.2f}",
                   f"{statistics.median(ts):.2f}",
                   f"{sorted(ts)[int(len(ts) * 0.95)]:.2f}", f"{max(ts):.2f}"]
            w.writerow(row)
            rows.append(row)
            print(f"{name:<16} 均值 {row[1]:>7} ms  中位 {row[2]:>7} ms  P95 {row[3]:>7} ms")

    mean_total = statistics.mean(total)
    print(f"\n单帧总耗时约 {mean_total:.1f} ms -> 处理上限 FPS ~{1000 / mean_total:.0f}")
    print(f"估算端到端延迟 ≈ 采集(≤33ms) + 处理({mean_total:.0f}ms) + 显示(≤17ms) "
          f"≈ {33 + mean_total + 17:.0f} ms")
    print(f"结果已保存: {os.path.join(out_dir, 'benchmark.csv')}")
    tracker.close()


if __name__ == "__main__":
    main()
