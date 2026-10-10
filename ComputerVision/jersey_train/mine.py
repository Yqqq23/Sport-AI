"""从**你自己的视频**里挖球衣 crop，供网页人工标注 / 微调用。

    python -m jersey_train.cli mine --video game.mp4 --out data/mined \
        --annotation-db .../backend/jersey_active_learning.db --video-id 88

为什么要挖自己的数据
--------------------
公开的 Roboflow 数据集是 NBA 转播的 224px 清晰特写，41 个类。
你的部署目标是固定广角机位、号码只有 20–40 px、球馆灯光、而且**号码集合完全不同**
（每个视频都不一样，FIBA 3x3 允许 0、00、1–99）。

在源域上训出来的模型，验证集好看，真视频上崩掉 —— 而且崩得很安静：
它会对着一个从没见过的号码给出高置信度的错误答案。

⚠ crop 必须由 basketball-v2 的函数产生
--------------------------------------
本模块不含任何裁剪几何常数。框、归属、补边、门限全部调 basketball-v2：

    检测框优先        tracked.jerseys（detect.CLASS_JERSEY == 3）
    没有才回退        jersey.torso_region_from_player_box()（15%–55% 高，中间 60% 宽）
    归属              pipeline._nearest_player()（owner_max_distance_ratio）
    补边              jersey.padded_box()（pad_ratio，crop_jersey 用的同一个）
    门限              jersey.gate_reason()（min_box_width / min_box_height）

自己另写一套裁剪，等于人为制造训练-推理不一致，而这种不一致在验证集上
完全看不出来 —— 它只在上线那天出现。

存盘存原始像素，不 resize
-------------------------
``crop_jersey`` 会把 crop 上采样到 ``upscale_to_height``（默认 96）再交给
识别器。那一步是**推理时**的事。存训练数据要存原始像素：

  - 上采样是确定性的，训练时随时能补上，存进 PNG 只是把插值烘死；
  - 存原始像素才能保证 ``crop == frame[y1:y2, x1:x2]``，manifest 里的
    ``jersey_bbox`` 因此是可复现的，出问题能一眼对回原帧。

门限判定仍然按推理的方式来（先 ``crop_jersey`` 拿到上采样后的图，再
``gate_reason``），只是存盘存的是没上采样的那份。

帧号
----
``frame_idx`` 是**原视频的真实解码帧号**，和网页 ``data/frames/{id}/
frame_%06d.jpg`` 里的数字是同一个东西（见 B0 排查结论）。manifest 里只有
这一个帧号字段，不拆成 source/web 两个 —— 两个字段就意味着有一天它们会不等。

输出布局
--------
    <out>/crops/<video_name>/f000180__box_00.png
    <out>/crops/manifest.jsonl

注意：这个布局是给网页导入脚本（import_crops.py）用的，不是 ``label.py``
那个键盘标注器的 ``slot_XXX/`` 布局。两者不通用，标注改走网页了。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)

#: manifest 里 source 字段的取值。detector 的框和几何回退的框质量差一个量级，
#: 混在一起训练而不区分，等于把回退框的系统性偏差当成真值。
SOURCE_DETECTOR = "detector"
SOURCE_TORSO_FALLBACK = "torso_fallback"


class MineConfigurationError(RuntimeError):
    """参数或环境不对，不是数据问题。区分开是为了报错能说清该改什么。"""


@dataclass
class MineSummary:
    frames_processed: int = 0
    boxes_seen: int = 0
    crops_saved: int = 0
    skipped_gate: int = 0
    skipped_bad_crop: int = 0
    skipped_no_owner: int = 0
    skipped_per_track_cap: int = 0
    from_detector: int = 0
    from_torso: int = 0
    crop_widths: list[int] = field(default_factory=list)
    crop_heights: list[int] = field(default_factory=list)
    tracks_seen: set = field(default_factory=set)
    elapsed_sec: float = 0.0
    out_dir: Optional[str] = None
    manifest_path: Optional[str] = None
    stride: int = 0
    stride_source: str = ""

    def _dist(self, values: list[int]) -> str:
        if not values:
            return "n/a"
        return (f"min {min(values)}  中位 {int(statistics.median(values))}  "
                f"max {max(values)}")

    def report(self) -> str:
        total_src = self.from_detector + self.from_torso
        if total_src:
            pct_det = 100.0 * self.from_detector / total_src
            src_line = (f"{self.from_detector} detector ({pct_det:.1f}%) / "
                        f"{self.from_torso} torso_fallback ({100 - pct_det:.1f}%)")
        else:
            src_line = "n/a（一个 crop 都没存下来）"

        lines = [
            f"处理帧数     : {self.frames_processed}   (stride={self.stride}, 来源: {self.stride_source})",
            f"检测框数     : {self.boxes_seen}",
            f"裁剪成功     : {self.crops_saved}",
            f"涉及 track   : {len(self.tracks_seen)} 个",
            "",
            f"门限丢弃     : {self.skipped_gate}   (太小/太糊，推理时也会被挡，不要)",
            f"裁剪失败     : {self.skipped_bad_crop}   (补边后越界或空)",
            f"无归属       : {self.skipped_no_owner}   (球衣框找不到对应球员)",
            f"单 track 超额: {self.skipped_per_track_cap}",
            "",
            f"框来源占比   : {src_line}",
            f"crop 宽度    : {self._dist(self.crop_widths)}",
            f"crop 高度    : {self._dist(self.crop_heights)}",
            "",
            f"耗时         : {self.elapsed_sec:.0f} s",
            f"输出目录     : {self.out_dir}",
            f"manifest     : {self.manifest_path}",
        ]
        if self.from_torso and self.from_detector == 0:
            lines += [
                "",
                "⚠ 全部走的是 torso_fallback —— 检测器一个 jersey 框都没给出来。",
                "  这条路是降级路径：它不知道球员是不是背对镜头，侧身时裁到的",
                "  多半是空球衣。用这批数据微调之前先确认检测器权重带 jersey 类。",
            ]
        return "\n".join("  " + s if s else "" for s in lines)


def _safe_name(text: str) -> str:
    """视频名转成能当目录名的形式。中文保留，只挡路径分隔符和保留字符。"""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).strip(". ")
    return cleaned or "video"


def resolve_stride(
    explicit: Optional[int],
    annotation_db: Optional[str | Path],
    video_id: Optional[int],
) -> tuple[int, str]:
    """定出该用多大的 stride，并说清这个值是哪来的。

    优先级：显式 ``--stride`` > 网页库里该视频的 ``frame_interval`` >
    该视频 ``video_frames.frame_id`` 的差分。

    为什么不给默认值：网页只展示 ``frame_interval`` 的倍数帧，挖出来的
    crop 要能对上网页的帧，stride 必须整除 frame_interval。硬编码一个
    数（比如原来的 5）在 frame_interval 改成 25 的那天就会静默错位 ——
    不会报错，只会有 4/5 的 crop 找不到对应帧。
    """
    if explicit is not None:
        if explicit < 1:
            raise MineConfigurationError(f"--stride 必须 >= 1，收到 {explicit}")
        return explicit, "命令行 --stride"

    if annotation_db is None or video_id is None:
        raise MineConfigurationError(
            "定不出 stride。二选一：\n"
            "  1) 显式指定：  --stride 30\n"
            "  2) 从网页库读：--annotation-db <.../backend/jersey_active_learning.db> "
            "--video-id <该视频在网页里的 id>\n"
            "\n"
            "  不给默认值是有意的：stride 必须整除网页的 frame_interval，\n"
            "  否则挖出来的 crop 有一部分对不上任何一张网页帧，而且不报错。"
        )

    db_path = Path(annotation_db)
    if not db_path.is_file():
        raise MineConfigurationError(f"--annotation-db 不存在: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(video_assets)")}
        if "frame_interval" in columns:
            row = conn.execute(
                "SELECT frame_interval FROM video_assets WHERE id = ?", (video_id,)
            ).fetchone()
            if row is not None and row[0] is not None and int(row[0]) >= 1:
                return int(row[0]), f"video_assets.frame_interval (video_id={video_id})"

        # 老数据没有 frame_interval 字段/值，只能从帧号差分倒推。
        frame_ids = [
            int(r[0])
            for r in conn.execute(
                "SELECT frame_id FROM video_frames WHERE video_id = ? ORDER BY frame_id",
                (video_id,),
            )
        ]
    finally:
        conn.close()

    if len(frame_ids) < 2:
        raise MineConfigurationError(
            f"video_id={video_id} 在 {db_path} 里帧数不足（{len(frame_ids)} 行），"
            "既没有 frame_interval 也没法差分。用 --stride 手动指定。"
        )

    diffs = {b - a for a, b in zip(frame_ids, frame_ids[1:])}
    if len(diffs) != 1:
        raise MineConfigurationError(
            f"video_id={video_id} 的帧号间隔不均匀（观察到 {sorted(diffs)}），"
            "无法倒推 frame_interval。用 --stride 手动指定。"
        )
    step = diffs.pop()
    if step < 1:
        raise MineConfigurationError(f"倒推出的帧号间隔非法: {step}")
    LOGGER.warning(
        "video_assets 里没有 frame_interval，从 video_frames.frame_id 差分倒推得 %d。"
        "跑一次 scripts/migrate_db.py 之后新上传的视频就会直接带这个字段。", step
    )
    return step, f"video_frames.frame_id 差分倒推 (video_id={video_id})"


def mine(
    video: str | Path,
    out_dir: str | Path,
    *,
    stride: Optional[int] = None,
    annotation_db: str | Path | None = None,
    video_id: Optional[int] = None,
    repo: str | Path | None = None,
    config_path: str | Path | None = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_per_track: int = 80,
    allow_inexact_seek: bool = False,
    log_every: int = 200,
) -> MineSummary:
    # 这是本包**唯一**真正用到 basketball-v2 的地方。路径解析全在
    # trajectory/bridge.py 的 _candidates() 里，本文件不含任何写死的路径。
    #
    # 裸 import 失败时的 ModuleNotFoundError 看不出所以然（到底是 trajectory
    # 没上传、还是 sys.path 没带上项目根），所以在这里换成能定位问题的报错。
    try:
        from trajectory.bridge import Backend
    except ImportError as exc:
        raise MineConfigurationError(
            f"import 不到 trajectory.bridge：{exc}\n"
            f"  jersey_train 的 mine 子命令依赖同级的 trajectory 包（只用它的接入层）。\n"
            f"  期望的布局：\n"
            f"      {_PROJECT_ROOT}/\n"
            f"      ├── trajectory/     {'存在' if (_PROJECT_ROOT / 'trajectory').is_dir() else '缺失 <<<'}\n"
            f"      └── jersey_train/   存在\n"
            f"  已加入 sys.path 的项目根: {_PROJECT_ROOT}\n"
            f"  只想训练不挖数据的话，用 fetch / train / verify，它们不需要 trajectory。"
        ) from exc

    video = Path(video)
    if not video.exists():
        raise MineConfigurationError(f"视频不存在: {video}")

    # 风险 B：cap.set(CAP_PROP_POS_FRAMES) 在帧间编码上会落到最近的关键帧，
    # 但下面的 frame_idx 是按 start_frame 直接起算的。两者一旦不等，写出来的
    # 帧号就是错的，而且错得很安静 —— crop 会被导进网页的**另一张帧**上。
    if start_frame != 0 and not allow_inexact_seek:
        raise MineConfigurationError(
            f"--start-frame {start_frame} 被拒绝。\n"
            "  seek 用的是 cv2 的 CAP_PROP_POS_FRAMES，在帧间编码（H.264 等）上\n"
            "  会落到最近的关键帧，但帧号是按 start_frame 直接起算的。两者不等时\n"
            "  写出来的 frame_idx 是错的，crop 会被导到网页的另一张帧上，不报错。\n"
            "\n"
            "  正常用法是 --start-frame 0，配 --end-frame 限制范围。\n"
            "  确实要从中间开始、且已确认这段视频 seek 精确，加 --allow-inexact-seek。"
        )

    resolved_stride, stride_source = resolve_stride(stride, annotation_db, video_id)

    out_dir = Path(out_dir)
    video_name = _safe_name(video.stem)
    crops_root = out_dir / "crops"
    crop_dir = crops_root / video_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = crops_root / "manifest.jsonl"

    backend = Backend(repo)
    LOGGER.info(backend.describe())
    cfg_path = Path(config_path) if config_path else backend.default_config_path
    config = backend.load_config(cfg_path)

    tracker = backend.make_tracker(config.detector.weights, config.tracker)
    slot_tracker = backend.make_slot_tracker(config.identity.max_players)

    jersey = backend.jersey
    pipeline = backend.pipeline
    crop_jersey = jersey.crop_jersey
    gate_reason = jersey.gate_reason
    gate_ok = jersey.GATE_OK
    padded_box = jersey.padded_box
    torso_region = jersey.torso_region_from_player_box
    nearest_player = pipeline._nearest_player
    torso_box_type = pipeline._TorsoBox
    gate_cfg = config.jersey.gate

    LOGGER.info(
        "gate: min %dx%d  pad_ratio %.2f  upscale_to_height %d  |  torso %.2f-%.2f 高, %.2f 宽",
        gate_cfg.min_box_width, gate_cfg.min_box_height, gate_cfg.pad_ratio,
        gate_cfg.upscale_to_height, config.jersey.torso_top,
        config.jersey.torso_bottom, config.jersey.torso_width,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise MineConfigurationError(f"打不开视频: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    summary = MineSummary(
        out_dir=str(crop_dir), manifest_path=str(manifest_path),
        stride=resolved_stride, stride_source=stride_source,
    )
    per_track: dict[int, int] = {}
    started = time.perf_counter()
    frame_idx = int(start_frame)

    # 追加而不是覆盖：一个 manifest 可以装多个视频（crop_path 里带 video 目录，
    # 且 import 脚本按 (video, frame_idx, box_idx) 幂等）。重跑同一个视频会产生
    # 重复行，由导入侧去重 —— 在这里去重要先把整个文件读回来，得不偿失。
    manifest = manifest_path.open("a", encoding="utf-8")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if end_frame is not None and frame_idx >= end_frame:
                break
            if resolved_stride > 1 and (frame_idx - start_frame) % resolved_stride != 0:
                frame_idx += 1
                continue

            tracked = tracker.update(frame, frame_idx=frame_idx,
                                     timestamp_sec=frame_idx / max(fps, 1e-6))
            slots = slot_tracker.assign([p.bbox_xyxy for p in tracked.players], frame_idx)
            slot_of_player = {id(p): s for p, s in zip(tracked.players, slots)}

            # 和 pipeline.py:275-290 同一套优先级：检测器的 jersey 框优先，
            # 一个都没有才整帧退回几何躯干区域。
            if tracked.jerseys:
                jersey_sources = list(tracked.jerseys)
                box_source = SOURCE_DETECTOR
            elif config.jersey.fallback_to_torso:
                jersey_sources = [
                    torso_box_type(bbox_xyxy=tuple(torso_region(
                        p.bbox_xyxy, config.jersey.torso_top,
                        config.jersey.torso_bottom, config.jersey.torso_width)))
                    for p in tracked.players
                ]
                box_source = SOURCE_TORSO_FALLBACK
            else:
                jersey_sources = []
                box_source = SOURCE_DETECTOR

            summary.boxes_seen += len(jersey_sources)
            box_idx = 0

            for jersey_box in jersey_sources:
                owner = nearest_player(
                    jersey_box, tracked.players,
                    max_distance_ratio=config.jersey.owner_max_distance_ratio,
                )
                if owner is None:
                    summary.skipped_no_owner += 1
                    continue
                slot = slot_of_player.get(id(owner), 0)
                if slot == 0:
                    summary.skipped_no_owner += 1
                    continue
                if per_track.get(slot, 0) >= max_per_track:
                    summary.skipped_per_track_cap += 1
                    continue

                raw_bbox = tuple(float(v) for v in jersey_box.bbox_xyxy)

                # 门限判定完全照推理的顺序来（pipeline.py:331-341）：
                # 先拿到补边+上采样后的图，再用**原始框**判门限。
                gate_crop = crop_jersey(frame, raw_bbox, gate_cfg)
                if gate_crop is None:
                    summary.skipped_bad_crop += 1
                    continue
                if gate_reason(list(raw_bbox), gate_cfg, gate_crop) != gate_ok:
                    summary.skipped_gate += 1
                    continue

                # 存盘用同一个补边框，但不上采样 —— 见模块 docstring。
                box = padded_box(raw_bbox, gate_cfg, frame.shape)
                if box is None:
                    summary.skipped_bad_crop += 1
                    continue
                px1, py1, px2, py2 = box
                crop = frame[py1:py2, px1:px2]
                if crop.size == 0:
                    summary.skipped_bad_crop += 1
                    continue

                filename = f"f{frame_idx:06d}__box_{box_idx:02d}.png"
                path = crop_dir / filename
                ok_enc, buf = cv2.imencode(".png", crop)
                if not ok_enc:
                    summary.skipped_bad_crop += 1
                    continue
                buf.tofile(str(path))      # 中文路径下 imwrite 会静默失败

                record: dict[str, Any] = {
                    "video": video_name,
                    "frame_idx": frame_idx,
                    "box_idx": box_idx,
                    "crop_path": str(path.relative_to(crops_root)).replace("\\", "/"),
                    "player_bbox": [int(round(v)) for v in owner.bbox_xyxy],
                    "jersey_bbox": [px1, py1, px2, py2],
                    "source": box_source,
                    "crop_w": int(crop.shape[1]),
                    "crop_h": int(crop.shape[0]),
                    "track_id": int(owner.track_id) if owner.track_id is not None else None,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

                per_track[slot] = per_track.get(slot, 0) + 1
                summary.crops_saved += 1
                summary.tracks_seen.add(slot)
                summary.crop_widths.append(int(crop.shape[1]))
                summary.crop_heights.append(int(crop.shape[0]))
                if box_source == SOURCE_DETECTOR:
                    summary.from_detector += 1
                else:
                    summary.from_torso += 1
                box_idx += 1

            summary.frames_processed += 1
            if log_every and summary.frames_processed % log_every == 0:
                LOGGER.info("frame %d  crops %d  tracks %s",
                            frame_idx, summary.crops_saved, sorted(summary.tracks_seen))
            frame_idx += 1
    finally:
        manifest.close()
        cap.release()

    summary.elapsed_sec = time.perf_counter() - started
    return summary
