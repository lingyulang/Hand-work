# -*- coding: utf-8 -*-
"""中文文本绘制工具。

OpenCV 的 putText 不支持中文，这里用 PIL + 系统字体
（微软雅黑/黑体）绘制后转回 numpy 数组，开销可控。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}
_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/simsun.ttc",  # 宋体
]


def _get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            break
        except OSError:
            continue
    if font is None:  # 全部失败则退化为默认字体（中文可能显示为方块）
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_texts_cn(frame_bgr, jobs):
    """批量绘制多条中文文本（整帧只做一次 PIL 来回转换，性能关键）。

    jobs : list[(text, (x, y), size, color_bgr, bold)]
    """
    img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB，仅一次
    draw = ImageDraw.Draw(img)
    for text, org, size, color, bold in jobs:
        font = _get_font(size)
        if bold:
            x, y = org
            rgb = tuple(int(c) for c in color[::-1])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    draw.text((x + dx, y + dy), text, font=font, fill=rgb)
        else:
            draw.text(org, text, font=font,
                      fill=tuple(int(c) for c in color[::-1]))
    frame_bgr[:, :, :] = np.asarray(img)[:, :, ::-1]
    return frame_bgr


def draw_text_cn(frame_bgr, text, org, size=18, color=(255, 255, 255), bold=False):
    """单条文本绘制（多条请用 draw_texts_cn，避免重复全帧转换）。"""
    return draw_texts_cn(frame_bgr, [(text, org, size, color, bold)])


def text_size(text, size=18):
    """返回 (宽, 高)，供布局计算使用。"""
    font = _get_font(size)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top
