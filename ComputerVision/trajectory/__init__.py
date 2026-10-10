"""按球衣号抽取球员的俯视轨迹。

把两边接起来：
    basketball-v2   检测 + MOT + 球衣号识别（已有，不重写）
    calibration/    米制单应标定（本项目，已验收）

流程：
    python -m trajectory.cli check
    python -m trajectory.cli extract --video game.mp4 --out outputs/run1
    python -m trajectory.cli players --csv outputs/run1/tracks.csv
    python -m trajectory.cli plot    --csv outputs/run1/tracks.csv --jersey 7
"""

__all__ = ["bridge", "court", "extract", "paths", "viz"]
