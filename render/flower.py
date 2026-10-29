# -*- coding: utf-8 -*-
"""五瓣花绽放渲染器（纯 numpy/OpenCV 绘制，无外部美术资源）。

动画设计
--------
* 绽放度 bloom ∈ [0,1]：0=花苞（花瓣收拢抱合）, 1=盛开（花瓣全展）。
* 花瓣几何由 bloom 经 easeInOutCubic 缓动后驱动：
    - 花瓣长度 L、宽度 W 随 bloom 增大；
    - 花瓣张角从"聚拢在正上方 ±16°"展开为"均匀分布 5×72°"；
    - 展开顺序 1→2→3→4→0，形成依次绽放的层次感。
* 花茎随手部水平位置柔性摆动（运动轨迹的可视化延伸）。
* 盛开时持续飘落花瓣粒子（粒子特效）。

花色可通过捏合手势或 C 键循环切换（附加分）。
"""
import math

import cv2
import numpy as np

from logic.easing import ease_in_out_cubic
from .particles import ParticleSystem

# 花色调色板 (名称, 花瓣BGR, 花瓣浅色BGR, 花蕊BGR, 背景顶BGR, 背景底BGR)
PALETTES = [
    ("樱粉", (150, 130, 255), (200, 185, 255), (60, 200, 255), (60, 45, 60), (25, 18, 30)),
    ("紫罗兰", (220, 110, 200), (245, 180, 230), (70, 220, 255), (55, 35, 70), (22, 15, 32)),
    ("旭日橙", (60, 160, 255), (120, 205, 255), (80, 220, 255), (55, 45, 65), (25, 20, 28)),
    ("青碧", (180, 220, 130), (220, 245, 190), (80, 230, 255), (45, 55, 60), (18, 24, 26)),
]

_GREEN = (90, 160, 70)        # 花茎/花萼
_GREEN_LIGHT = (120, 190, 90)
_GOLD = (60, 200, 250)


class FlowerRenderer:
    """渲染一块 (w, h) 的花画面面板。"""

    def __init__(self, w=640, h=660, palette_idx=0, scale_base=1.0):
        self.w, self.h = w, h
        self.palette_idx = palette_idx % len(PALETTES)
        self.scale_base = scale_base
        self.particles = ParticleSystem()
        self._glow = self._make_glow_sprite()
        self._glow_bgr = self._tint_glow()
        self._bg = self._make_background()
        self._sways = [0.0, 0.0]  # 每朵花独立的摆动平滑值（最多双花）

    def refresh_assets(self):
        """换花色后重建背景与光晕贴图（供按键切换时调用）。"""
        self._bg = self._make_background()
        self._glow_bgr = self._tint_glow()

    def _tint_glow(self):
        """把光晕精灵按当前花色预染成 3 通道 BGR（免去每帧浮点混合）。"""
        tint = np.array(self.palette[1], np.float32) / 255.0 * 0.9
        bgr = (self._glow[..., None].astype(np.float32) * tint[None, None, :])
        return np.clip(bgr, 0, 255).astype(np.uint8)

    # ---------------- 调色板 ----------------
    @property
    def palette(self):
        return PALETTES[self.palette_idx]

    def cycle_palette(self, step=1):
        self.palette_idx = (self.palette_idx + step) % len(PALETTES)
        return self.palette[0]

    # ---------------- 静态背景 ----------------
    def _make_background(self):
        """预生成渐变背景 + 光斑（bokeh），每帧直接拷贝，零重复计算。"""
        w, h = self.w, self.h
        _, _, _, _, top, bottom = self.palette
        top = np.array(top, dtype=np.float32)
        bottom = np.array(bottom, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        bg = (top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp)
        bg = np.repeat(bg, w, axis=1).astype(np.uint8)

        rng = np.random.RandomState(7)  # 固定种子保证观感稳定
        for _ in range(18):
            x, y = rng.randint(0, w), rng.randint(0, int(h * 0.85))
            r = rng.randint(12, 46)
            alpha = rng.uniform(0.04, 0.10)
            color = np.array(self.palette[1], dtype=np.float32)
            roi = bg[max(0, y - r):y + r, max(0, x - r):x + r]
            if roi.size == 0:
                continue
            yy, xx = np.mgrid[max(0, y - r):min(h, y + r),
                              max(0, x - r):min(w, x + r)]
            d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
            mask = np.clip(1 - d, 0, 1) ** 2 * alpha
            roi[:] = np.clip(
                roi.astype(np.float32) * (1 - mask[..., None])
                + color * mask[..., None], 0, 255
            ).astype(np.uint8)
        return bg

    def _make_glow_sprite(self, size=220):
        """预生成径向光晕精灵，盛开时叠加在花后，提升氛围。"""
        r = size // 2
        yy, xx = np.mgrid[-r:r, -r:r]
        d = np.sqrt(xx ** 2 + yy ** 2) / r
        glow = np.clip(1 - d, 0, 1) ** 2.2
        return (glow * 255).astype(np.uint8)

    # ---------------- 花瓣几何 ----------------
    @staticmethod
    def _petal_polygon(length, width, n=20):
        """单瓣轮廓：两条二次贝塞尔曲线构成的对称花瓣（局部坐标，基部在原点）。"""
        t = np.linspace(0, 1, n)[:, None]
        # 右边缘: P0(0,0) -> P1(width, -length*0.55) -> P2(0, -length)
        right = (1 - t) ** 2 * np.array([0.0, 0.0]) \
            + 2 * (1 - t) * t * np.array([width, -length * 0.55]) \
            + t ** 2 * np.array([0.0, -length])
        left = right * np.array([-1.0, 1.0])
        pts = np.vstack([left[::-1], right])  # 闭合轮廓
        return pts

    def _draw_petal(self, canvas, base_xy, angle_deg, length, width, colors):
        """在画布上绘制一片旋转后的花瓣（主色+内层浅色+描边）。"""
        pts = self._petal_polygon(length, width)
        a = math.radians(angle_deg)
        rot = np.array([[math.cos(a), -math.sin(a)],
                        [math.sin(a), math.cos(a)]])
        pts = pts @ rot.T + np.array(base_xy, dtype=np.float64)
        poly = np.round(pts).astype(np.int32)

        cv2.fillPoly(canvas, [poly], colors[0], lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [poly], True, colors[2], 2, cv2.LINE_AA)
        # 内层浅色小瓣，制造渐变层次
        pts2 = self._petal_polygon(length * 0.72, width * 0.62) @ rot.T \
            + np.array(base_xy, dtype=np.float64)
        cv2.fillPoly(canvas, [np.round(pts2).astype(np.int32)],
                     colors[1], lineType=cv2.LINE_AA)

    # ---------------- 主渲染 ----------------
    def render(self, bloom, pos_x=0.0, scale=1.0, t=0.0):
        """渲染单朵花（兼容接口）。"""
        return self.render_multi([(bloom, pos_x, scale)], t=t)

    def render_multi(self, items, t=0.0):
        """渲染 1~2 朵花。

        items : [(bloom, pos_x, scale_rel), ...]
                双花时分别绘制在面板左右 27%/73% 处。
        """
        canvas = self._bg.copy()
        if len(items) <= 1:
            b, px, s = items[0] if items else (0.35, 0.0, 1.0)
            self._draw_flower(canvas, 0, b, px, s, t, home_x=0.5)
        else:
            self._draw_flower(canvas, 0, items[0][0], items[0][1],
                              items[0][2], t, home_x=0.27)
            self._draw_flower(canvas, 1, items[1][0], items[1][1],
                              items[1][2], t, home_x=0.73)
        self.particles.update(1 / 30.0, self.w, self.h)
        self.particles.draw(canvas)
        return canvas

    def _draw_flower(self, canvas, idx, bloom, pos_x=0.0, scale=1.0, t=0.0,
                     home_x=0.5):
        """在画布上绘制一朵五瓣花（含茎、叶、光晕、花蕊）。

        idx : 花朵槽位号（每朵花独立的摆动平滑状态）
        home_x : 花朵基准水平位置（面板宽度比例 0~1）
        """
        w, h = self.w, self.h

        # 手部位置 -> 花茎摆动（渲染层内再做一次轻度平滑）
        self._sways[idx] += (pos_x - self._sways[idx]) * 0.12
        sway = self._sways[idx]

        base = self.scale_base * scale
        cx = w * home_x + sway * w * 0.06    # 花头随摆动轻微移动
        cy = h * 0.44
        stem_base_x = w * home_x
        stem_base_y = h - 8

        # ---- 光晕（盛开时，预染色贴图 + cv2.add 饱和叠加） ----
        if bloom > 0.45:
            g = max(16, int((140 + 170 * bloom) * base))  # 下限保护：缩放<1时不得为0
            glow = cv2.resize(self._glow_bgr, (g, g))
            gx, gy = int(cx - g / 2), int(cy - g / 2)
            x0, y0 = max(0, gx), max(0, gy)
            x1, y1 = min(w, gx + g), min(h, gy + g)
            if x1 > x0 and y1 > y0:
                roi = canvas[y0:y1, x0:x1]
                spr = glow[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
                canvas[y0:y1, x0:x1] = cv2.add(roi, spr)

        # ---- 花茎（二次贝塞尔弯曲） + 叶片 ----
        ts = np.linspace(0, 1, 24)[:, None]
        p0 = np.array([stem_base_x, stem_base_y])
        p1 = np.array([stem_base_x + sway * 30, (stem_base_y + cy) / 2])
        p2 = np.array([cx, cy])
        stem = (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * p1 + ts ** 2 * p2
        cv2.polylines(canvas, [np.round(stem).astype(np.int32)],
                      False, _GREEN, max(4, int(9 * base)), cv2.LINE_AA)
        # 两片叶子
        for side, ty in ((-1, 0.68), (1, 0.58)):
            leaf_c = (1 - ty) ** 2 * p0 + 2 * (1 - ty) * ty * p1 + ty ** 2 * p2
            lx, ly = leaf_c[0] + side * 34 * base, leaf_c[1]
            ellipse = cv2.ellipse2Poly(
                (int(lx), int(ly)),
                (int(36 * base), int(14 * base)),
                int(side * -28), 0, 360, 20)
            cv2.fillPoly(canvas, [ellipse], _GREEN_LIGHT, lineType=cv2.LINE_AA)

        # ---- 花瓣（缓动驱动几何） ----
        b = ease_in_out_cubic(bloom)
        petal_len = (78 + 92 * b) * base          # 长度 78 -> 170
        petal_w = (26 + 66 * b) * base            # 宽度 26 -> 92
        spread = 0.16 + 0.84 * b                  # 张角系数 0.16 -> 1.0
        tilt = sway * 8                            # 随手部位置轻微倾斜
        idle = 4 * math.sin(t * 0.6)               # 静止时缓慢呼吸式缓摆

        colors = (self.palette[1], self.palette[2], (255, 255, 255))
        order = [1, 2, 3, 4, 0]   # 展开层次：从侧瓣到顶瓣最后画
        for k in order:
            angle = -90 + 72 * k * spread + tilt + idle
            self._draw_petal(canvas, (cx, cy), angle, petal_len, petal_w, colors)

        # ---- 花萼（花苞时包裹花瓣基部） ----
        if b < 0.75:
            cup_h = (46 + 20 * b) * base
            cup_w = (34 + 26 * b) * base
            cup = np.array([
                [cx - cup_w, cy + cup_h * 0.25],
                [cx - cup_w * 0.42, cy - cup_h * 0.55],
                [cx + cup_w * 0.42, cy - cup_h * 0.55],
                [cx + cup_w, cy + cup_h * 0.25],
                [cx, cy + cup_h],
            ], np.int32)
            cv2.fillPoly(canvas, [cup], _GREEN, lineType=cv2.LINE_AA)

        # ---- 花蕊 ----
        core_r = int((10 + 9 * b) * base)
        cv2.circle(canvas, (int(cx), int(cy)), core_r + 4, (40, 150, 210), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(cx), int(cy)), core_r, _GOLD, -1, cv2.LINE_AA)
        if b > 0.45:  # 盛开后出现花丝
            n_st = 10
            for i in range(n_st):
                a = 2 * math.pi * i / n_st + t * 0.15
                r2 = core_r + int((10 + 6 * b) * base)
                x2, y2 = cx + r2 * math.cos(a), cy + r2 * math.sin(a)
                cv2.line(canvas, (int(cx), int(cy)), (int(x2), int(y2)),
                         _GOLD, 2, cv2.LINE_AA)
                cv2.circle(canvas, (int(x2), int(y2)), 3, _GOLD, -1, cv2.LINE_AA)

        # ---- 粒子：完全盛开时飘落花瓣 ----
        if bloom > 0.85 and np.random.rand() < 0.35:
            self.particles.emit(cx, cy, [self.palette[1], self.palette[2]])

        return canvas
