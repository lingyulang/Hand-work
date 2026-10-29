# -*- coding: utf-8 -*-
"""花瓣粒子特效：完全盛开时飘落的花瓣，增强视觉美感。"""
import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float      # 剩余寿命（秒）
    max_life: float
    size: int
    color: tuple     # BGR
    phase: float     # 摇摆相位


class ParticleSystem:
    """轻量级粒子池：盛开(>=阈值)时按概率生成飘落花瓣。"""

    def __init__(self, max_particles=60):
        self.particles = []
        self.max_particles = max_particles

    def emit(self, x, y, colors, n=1, spread=120):
        """在 (x, y) 附近生成 n 个花瓣粒子。"""
        for _ in range(n):
            if len(self.particles) >= self.max_particles:
                return
            self.particles.append(
                Particle(
                    x=x + random.uniform(-spread, spread),
                    y=y + random.uniform(-spread * 0.4, spread * 0.4),
                    vx=random.uniform(-18, 18),
                    vy=random.uniform(25, 60),
                    life=random.uniform(2.0, 4.0),
                    max_life=4.0,
                    size=random.randint(5, 11),
                    color=random.choice(colors),
                    phase=random.uniform(0, 6.28),
                )
            )

    def update(self, dt, w, h):
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.phase += dt * 2.5
            # 水平方向正弦摇摆，模拟花瓣飘落
            p.x += (p.vx + 22.0 * np.sin(p.phase)) * dt
            p.y += p.vy * dt
            if p.y < h + 20 and 0 < p.x < w:
                alive.append(p)
        self.particles = alive

    def draw(self, canvas):
        for p in self.particles:
            alpha = max(0.0, min(1.0, p.life / p.max_life))
            color = tuple(int(c * alpha) for c in p.color)
            # 旋转的椭圆花瓣
            angle = np.degrees(p.phase) % 180
            axes = (p.size, max(2, p.size // 2))
            pts = cv2.ellipse2Poly(
                (int(p.x), int(p.y)), axes, int(angle), 0, 360, 30
            )
            cv2.fillPoly(canvas, [pts], color, lineType=cv2.LINE_AA)
