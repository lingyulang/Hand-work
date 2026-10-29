# -*- coding: utf-8 -*-
"""汇总现场实测 stats.csv 的 FPS / 延迟 / 伸展度统计（写入报告用）。"""
import csv
import os
import statistics

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "stats.csv")
with open(path, encoding="utf-8-sig") as fp:
    rows = list(csv.DictReader(fp))

fps = [float(r["fps"]) for r in rows]
lat = [float(r["latency_ms"]) for r in rows]
raw = [float(r["openness_raw"]) for r in rows]
n = len(rows)

def pct(v, q):
    s = sorted(v)
    return s[int(len(s) * q)]

print(f"总帧数: {n} (约 {n/30:.0f}s @30fps)")
print(f"FPS   均值 {statistics.mean(fps):.1f} | 中位 {statistics.median(fps):.1f} | P05 {pct(fps,0.05):.1f} | 最小 {min(fps):.1f}")
print(f"延迟  均值 {statistics.mean(lat):.1f} ms | 中位 {statistics.median(lat):.1f} | P95 {pct(lat,0.95):.1f} | 最大 {max(lat):.1f}")
stable = [f for f in fps[30:]]  # 跳过启动预热前1秒
print(f"预热后FPS均值 {statistics.mean(stable):.1f} | P05 {pct(stable,0.05):.1f}")

# 按检测到的手数分组（验证"手入画后掉帧"是否解决）
for nh in (0, 1, 2):
    grp = [float(r["fps"]) for r in rows if int(r["n_hands"]) == nh][30:]
    glat = [float(r["latency_ms"]) for r in rows if int(r["n_hands"]) == nh][30:]
    if grp:
        print(f"{nh} 手场景: {len(grp)} 帧 | FPS 均值 {statistics.mean(grp):.1f}"
              f" | 延迟均值 {statistics.mean(glat):.1f} ms")
