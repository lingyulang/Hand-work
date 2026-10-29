# -*- coding: utf-8 -*-
"""基于骨骼关键点检测的隔空手势动态交互系统 —— 五瓣花盛开。

架构（三层解耦）:
    视觉层 vision/   摄像头采集、MediaPipe 手部/手臂关键点、手势特征
    逻辑层 logic/    EMA/卡尔曼平滑（抖动消除）、缓动、手势->动画映射
    渲染层 render/   五瓣花绽放动画、粒子特效、分屏 UI

运行:
    python main.py                       # 默认摄像头 0
    python main.py --camera 1            # 指定摄像头
    python main.py --video test.mp4      # 离线视频测试
    python main.py --smoother ema        # 指定平滑算法 kalman/ema/off
    python main.py --log stats.csv       # 逐帧记录性能数据（用于评估报告）

按键:
    S 平滑开关(现场对比) | M 切换 EMA/卡尔曼 | C 切换花色
    P 手臂姿态开关 | H 帮助 | Q/ESC 退出
"""
import argparse
import time
import sys

import cv2
import numpy as np

import config as cfg
from logic.mapper import GestureMapper
from logic.smoother import make_smoother
from render.flower import FlowerRenderer
from render.ui import AppUI
from utils.stats import PerfMonitor
from vision.camera import CameraStream
from vision.features import extract_features
from vision.hand_tracker import HandTracker
from vision.pose_tracker import PoseTracker


def parse_args():
    p = argparse.ArgumentParser(description="隔空手势动态交互系统（五瓣花盛开）")
    p.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX, help="摄像头索引")
    p.add_argument("--video", type=str, default=None, help="离线视频文件路径（替代摄像头）")
    p.add_argument("--smoother", type=str, default=cfg.SMOOTHER_TYPE,
                   choices=["kalman", "ema", "off"], help="平滑算法")
    p.add_argument("--pose", action="store_true", help="启用手臂姿态检测(增加耗帧)")
    p.add_argument("--log", type=str, default=cfg.STATS_CSV, help="性能CSV输出路径")
    p.add_argument("--no-cam-panel", action="store_true", help="隐藏摄像头面板(纯动画)")
    return p.parse_args()


class VideoSource:
    """统一封装摄像头/离线视频两种输入。"""

    def __init__(self, args):
        self.args = args
        self.is_camera = args.video is None
        if self.is_camera:
            self.src = CameraStream(args.camera, cfg.FRAME_WIDTH,
                                    cfg.FRAME_HEIGHT, cfg.CAMERA_FPS).start()
        else:
            self.cap = cv2.VideoCapture(args.video)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {args.video}")
            self.last_frame = None

    def read(self):
        """返回 (成功, 帧, 时间戳)。视频文件循环播放。"""
        if self.is_camera:
            return self.src.read()
        ok, frame = self.cap.read()
        if not ok:  # 循环
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok:
            self.last_frame = frame
        return ok, self.last_frame, time.time()

    def release(self):
        if self.is_camera:
            self.src.release()
        else:
            self.cap.release()


def main():
    args = parse_args()
    print("=" * 56)
    print("  隔空手势动态交互系统 —— 五瓣花盛开")
    print("  握拳=花苞  张开=盛开  捏合=换花色  双手=双花各手控一朵")
    print("  手部21点骨骼 + 手臂骨骼(肩肘腕) 实时叠加")
    print("=" * 56)

    source = VideoSource(args)

    # ---------- 各层初始化 ----------
    tracker = HandTracker(
        max_num_hands=cfg.HAND_MAX_NUM,
        det_conf=cfg.HAND_DET_CONF,
        track_conf=cfg.HAND_TRACK_CONF,
        model_complexity=cfg.HAND_MODEL_COMPLEXITY,
    )
    pose = PoseTracker(cfg.POSE_MODEL_COMPLEXITY, cfg.POSE_INPUT_W,
                       cfg.POSE_CADENCE) if (args.pose or cfg.POSE_ENABLE_DEFAULT) else None
    mapper = GestureMapper(cfg, make_smoother(args.smoother, cfg))
    flower = FlowerRenderer(
        w=cfg.WINDOW_W - cfg.CAM_PANEL_W,
        h=cfg.WINDOW_H - cfg.INFO_BAR_H,
    )
    ui = AppUI(cfg, cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
    perf = PerfMonitor(csv_path=args.log)

    smooth_on = cfg.SMOOTH_DEFAULT and args.smoother != "off"
    smoother_kind = args.smoother if args.smoother != "off" else cfg.SMOOTHER_TYPE
    mapper.set_smoother(make_smoother(smoother_kind if smooth_on else "off", cfg))

    show_help = False
    palette_name = flower.palette[0]
    win_name = "Gesture Flower  |  隔空手势动态交互系统"

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    prev_t = time.time()
    pinch_hand_lost_prev = False

    try:
        while True:
            ok, frame, frame_ts = source.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = max(min(now - prev_t, 0.1), 0.0)  # 防时钟跳变导致的负步长
            prev_t = now
            perf.tick_frame()

            # ================= 视觉层 =================
            hands = tracker.process(frame)
            if pose is not None:
                frame = pose.process_and_draw(frame)
            frame = tracker.draw(frame, hands)
            feats = extract_features(hands, cfg)

            if feats.n_hands > 0:
                ui.push_trail(feats.center)
            else:
                # 轨迹自然衰减
                if ui._trail:
                    ui._trail.popleft()
                    ui._trail_time.popleft()

            # ================= 逻辑层 =================
            params = mapper.update(feats, dt)
            if params.pinch_rise:               # 捏合上升沿 -> 切换花色
                palette_name = flower.cycle_palette()
                flower.refresh_assets()          # 背景色同步换

            # ================= 渲染层 =================
            panel = flower.render_multi(params.render_items, t=now)

            stats = {
                "fps": perf.fps,
                "latency_ms": perf.latency_ms,
                "bloom": params.bloom,
                "smooth_on": smooth_on,
                "smoother_name": mapper.smoother_name,
                "hand_lost": params.hand_lost,
                "n_hands": feats.n_hands,
                "n_flowers": params.n_flowers,
                "pose_on": pose is not None,
                "palette_name": palette_name,
            }

            if args.no_cam_panel:
                view = panel
                view = draw_stats_simple(view, stats)
            else:
                view = ui.compose(frame, panel, stats, cfg)
            if show_help:
                view = AppUI.draw_help(view)

            perf.tick_latency(frame_ts)
            perf.log_row(perf.fps, perf.latency_ms, feats.openness_raw,
                         params.bloom_target, feats.n_hands, smooth_on)

            cv2.imshow(win_name, view)
            # pollKey 非阻塞处理 UI 事件（Windows 上 waitKey(1) 每帧白等 ~10-15ms）
            _poll = getattr(cv2, "pollKey", None)
            key = (_poll() if _poll else cv2.waitKey(1)) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                smooth_on = not smooth_on
                mapper.set_smoother(make_smoother(smoother_kind if smooth_on else "off", cfg))
                print(f"[切换] 平滑: {'开启' if smooth_on else '关闭'}")
            elif key in (ord('m'), ord('M')):
                smoother_kind = "ema" if smoother_kind == "kalman" else "kalman"
                mapper.set_smoother(make_smoother(smoother_kind if smooth_on else "off", cfg))
                print(f"[切换] 平滑算法: {smoother_kind}")
            elif key in (ord('c'), ord('C')):
                palette_name = flower.cycle_palette()
                flower.refresh_assets()
            elif key in (ord('p'), ord('P')):
                if pose is None:
                    pose = PoseTracker(cfg.POSE_MODEL_COMPLEXITY,
                                       cfg.POSE_INPUT_W, cfg.POSE_CADENCE)
                    print("[切换] 手臂姿态: 开启")
                else:
                    pose.close()
                    pose = None
                    print("[切换] 手臂姿态: 关闭")
            elif key in (ord('h'), ord('H')):
                show_help = not show_help

            pinch_hand_lost_prev = params.hand_lost
    finally:
        tracker.close()
        if pose is not None:
            pose.close()
        source.release()
        perf.close()
        cv2.destroyAllWindows()
    print("已退出。性能数据" + (f"已保存: {args.log}" if args.log else "未记录(--log 可开启)"))


def draw_stats_simple(panel, stats):
    """纯动画模式下的简单信息叠加。"""
    from render.text_cn import draw_text_cn
    draw_text_cn(panel, f"FPS {stats['fps']:.1f}  延迟 {stats['latency_ms']:.0f}ms",
                 (10, 620), 18, (140, 230, 140))
    return panel


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"[错误] {e}", file=sys.stderr)
        print("提示: 可用 --video <mp4路径> 进行离线测试，或 --camera 1 更换摄像头。")
        sys.exit(1)
