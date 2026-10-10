"""Unicode 安全的图像 / 视频读写。

为什么需要这个文件：素材放在 ``D:\\帝国理工\\Computer Vision`` 下，路径含非 ASCII
字符。Windows 上 ``cv2.imread`` / ``cv2.imwrite`` 走的是窄字符（ANSI）文件名 API，
遇到中文路径会**静默返回 None**，不会抛异常 —— 正是那种最难查的坑。
``cv2.VideoCapture`` 同理。

所以本包内所有图像读写一律走 ``np.fromfile`` / ``ndarray.tofile``，
视频则先把路径转成 Windows 8.3 短路径再交给 VideoCapture。
"""

from __future__ import annotations

import ctypes
import os
import sys

import cv2
import numpy as np

# 素材根目录：本包的上一级
ASSET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DIR = os.path.dirname(os.path.abspath(__file__))


def asset(*parts: str) -> str:
    """拼出素材目录下的绝对路径。"""
    return os.path.join(ASSET_DIR, *parts)


def calib(*parts: str) -> str:
    """拼出 calibration/ 目录下的绝对路径。"""
    return os.path.join(CALIB_DIR, *parts)


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读图；路径可含中文。读不到直接抛异常，不返回 None。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"图片不存在: {path}")
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError(f"图片解码失败（文件可能损坏或不是图片）: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """写图；路径可含中文。编码失败直接抛异常。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"图片编码失败: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf.tofile(path)


def _short_path(path: str) -> str:
    """把含非 ASCII 的 Windows 路径转成 8.3 短路径，喂给 VideoCapture。"""
    if sys.platform != "win32":
        return path
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
    if n == 0 or not buf.value:
        return path
    return buf.value


def open_video(path: str) -> cv2.VideoCapture:
    """打开视频；路径可含中文。打不开直接抛异常。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"视频不存在: {path}")
    cap = cv2.VideoCapture(_short_path(path))
    if not cap.isOpened():
        raise IOError(f"视频打不开（缺编解码器？）: {path}")
    return cap


def video_resolution(path: str) -> tuple[int, int]:
    """返回视频的 (width, height)。"""
    cap = open_video(path)
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def read_video_frame(path: str, index: int) -> np.ndarray:
    """抽取第 ``index`` 帧。抽不到直接抛异常。

    只用于取一张背景图做可视化 —— 标定本身是一次性的，
    本包任何地方都不会遍历视频帧。
    """
    cap = open_video(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if index < 0 or (total > 0 and index >= total):
            raise IndexError(f"帧号 {index} 越界，视频共 {total} 帧")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IOError(f"读取第 {index} 帧失败: {path}")
        return frame
    finally:
        cap.release()
