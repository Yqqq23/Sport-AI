"""3x3 篮球场单应标定。

固定机位，整段视频共用一个单应矩阵，纯 OpenCV 解析几何，不训练模型。

流程：
    1. python -m calibration.pick_points     # 在 real.png 上按编号点 13 个点
    2. python -m calibration.calibrate       # 解矩阵，写 court_3x3.json
    3. python -m calibration.validate        # 误差报告 + 两张可视化图
"""

__all__ = ["court_config", "homography", "pixel_frames", "io_utils"]
