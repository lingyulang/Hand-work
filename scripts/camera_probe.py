# -*- coding: utf-8 -*-
"""摄像头采集探针：对比后端(DSHOW/MSMF)与编码格式(RAW/MJPG)的实际送帧率。"""
import time

import cv2


def probe(backend_name, backend, use_mjpg=False):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print(f"{backend_name:>8} + {'MJPG' if use_mjpg else 'RAW ':4}: 打开失败")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(10):  # 预热/自动曝光收敛
        cap.read()
    t0, n = time.time(), 0
    while n < 60 and time.time() - t0 < 10:
        ok, _ = cap.read()
        if ok:
            n += 1
    dt = time.time() - t0
    actual = cap.get(cv2.CAP_PROP_FPS)
    print(f"{backend_name:>8} + {'MJPG' if use_mjpg else 'RAW ':4}: 采集 {n/dt:5.1f} FPS "
          f"(摄像头上报 {actual:.0f})")
    cap.release()


for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
    probe(backend_name, backend, use_mjpg=False)
    probe(backend_name, backend, use_mjpg=True)
