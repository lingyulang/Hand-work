# -*- coding: utf-8 -*-
"""离线渲染五瓣花三种典型状态，用于 README/报告展示（无需摄像头）。"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render.flower import FlowerRenderer  # noqa: E402
from render.text_cn import draw_text_cn  # noqa: E402


def imwrite_unicode(path, img):
    """cv2.imwrite 不支持中文路径，改用 imencode + 字节写入。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError("图像编码失败")
    with open(path, "wb") as fp:
        fp.write(buf.tobytes())


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(out_dir, exist_ok=True)
    states = [(0.02, "花苞（握拳）"), (0.5, "绽放中（半张）"), (0.98, "盛开（手掌张开）")]
    panels = []
    for bloom, label in states:
        r = FlowerRenderer(w=480, h=520)
        img = r.render(bloom, pos_x=0.0, scale=1.0, t=2.0)
        # 顶部半透明标题条
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (480, 44), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        draw_text_cn(img, label, (14, 8), 22, (255, 255, 255), bold=True)
        panels.append(img)
    combined = np.hstack(panels)
    out = os.path.join(out_dir, "flower_states.png")
    imwrite_unicode(out, combined)
    print(f"已保存: {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
