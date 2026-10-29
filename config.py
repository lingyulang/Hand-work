# -*- coding: utf-8 -*-
"""全局配置文件。

所有可调参数集中在此，便于现场调参与性能权衡。
分层约定：视觉层(vision) -> 逻辑层(logic) -> 渲染层(render)。
"""

# ---------------- 摄像头 ----------------
CAMERA_INDEX = 0          # 摄像头编号，默认 0
FRAME_WIDTH = 640         # 采集分辨率宽
FRAME_HEIGHT = 480        # 采集分辨率高
CAMERA_FPS = 30           # 采集帧率（仅作为摄像头提示值）

# ---------------- MediaPipe ----------------
HAND_MAX_NUM = 2          # 最多检测的手数（2 = 支持双手交互附加分）
HAND_DET_CONF = 0.6       # 手部检测置信度阈值
HAND_TRACK_CONF = 0.5     # 手部追踪置信度阈值
# 逆光/暗光鲁棒性：降低置信度阈值可提高召回，但可能引入误检；现场可按光照调整

POSE_ENABLE_DEFAULT = False   # 是否默认启用手臂姿态（--pose 开启，耗时会增加）
POSE_MODEL_COMPLEXITY = 1     # Pose 模型复杂度 0(轻量)/1(完整)

# ---------------- 手势特征 ----------------
# 伸展度原始值 = 5 指指尖相对掌心距离 / 掌径 的均值
# 握拳时约 0.7~0.9，张开时约 1.4~1.7（不同手型有差异，可微调）
OPENNESS_LOW = 0.72       # 握拳参考值 -> 映射为 0（花苞）
OPENNESS_HIGH = 1.48      # 完全张开参考值 -> 映射为 1（盛开）

PINCH_ON = 0.28           # 拇指-食指捏合判定阈值（占掌径比例），触发
PINCH_OFF = 0.40          # 松开阈值（滞回，防止边界抖动）

TRAJECTORY_LEN = 30       # 手部运动轨迹残留点数（轨迹可视化）

# ---------------- 平滑（抖动消除，核心考核点） ----------------
SMOOTH_DEFAULT = True     # 默认开启平滑（按 S 键现场对比开/关效果）
SMOOTHER_TYPE = "kalman"  # "kalman" | "ema"，按 M 键切换

# EMA 参数：alpha 越小平滑越强、延迟越大
EMA_ALPHA = 0.35

# 一维卡尔曼滤波参数（常值模型：x_k = x_{k-1} + w）
# Q 过程噪声小 -> 更平滑；R 观测噪声小 -> 更跟手。二者权衡延迟与抖动。
# 当前取值经 scripts/evaluate_smooth.py 信号网格扫描获得（见性能评估报告）。
KALMAN_Q = 5.0e-4
KALMAN_R = 2.0e-3

# 坐标平滑（用于花茎摆动/位置跟随），独立于数值平滑开关
POS_EMA_ALPHA = 0.25

# ---------------- 映射与动画 ----------------
BLOOM_TAU = 0.09          # 显示值趋近目标值的时间常数(秒)，指数缓动
SPRING_STIFFNESS = 120.0  # 弹簧刚度（绽放微过冲，增加灵动感）
SPRING_DAMPING = 14.0     # 弹簧阻尼

FLOWER_SCALE = 1.0        # 花朵基础缩放
TWO_HAND_SCALE = True     # 双手交互：双手间距控制花朵大小（附加分）

# ---------------- UI / 窗口 ----------------
WINDOW_W = 1280
WINDOW_H = 720
CAM_PANEL_W = 640         # 左侧摄像头面板宽
INFO_BAR_H = 60           # 底部信息栏高
FPS_TEXT = "FPS: {:.1f}"
LATENCY_TEXT = "延迟: {:.0f} ms"

# ---------------- 统计输出 ----------------
STATS_CSV = None          # 传入 --log path 后逐帧写 CSV
