# -*- mode: python ; coding: utf-8 -*-
"""GestureFlower 打包配置（PyInstaller spec）。

关键点：
  1. mediapipe 的模型文件(.binarypb/.tflite)与动态库必须整体收集，
     否则运行时报 FileNotFoundError；
  2. 采用 onedir 模式：双击 dist/GestureFlower/GestureFlower.exe 启动，
     首启快（免解压），现场演示体验优于 onefile；
  3. 保留控制台：便于查看 FPS 日志与错误信息，现场排障友好。

构建:  .venv\\Scripts\\python.exe -m PyInstaller GestureFlower.spec --noconfirm
产物:  dist\\GestureFlower\\  （整个文件夹一起分发，双击其中 GestureFlower.exe）
"""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("mediapipe")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 注意：不能排除 matplotlib —— mediapipe.python.solutions.drawing_styles 依赖它
        "tkinter",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GestureFlower",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="GestureFlower",
)
