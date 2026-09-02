"""把 jersey_train mine 挖出来的 crop 导进标注库。

    # 第一步：帧号一致性自检。只出图，不写库。
    python scripts/import_crops.py --manifest .../crops/manifest.jsonl --video-id 88

    # 人工看过 data/debug/framecheck_*.png，确认左右是同一个画面之后
    python scripts/import_crops.py --manifest .../crops/manifest.jsonl --video-id 88 --confirmed

为什么要有第一步
----------------
帧号恒等映射（网页的 frame_%06d.jpg == mine 的 f%06d.png == 原视频第 N 帧）
是**代码推理**出来的结论，前提是转码 1:1 保帧。B1a 已经在抽帧侧加了帧数校验，
但那只挡得住帧数对不上的情况；帧数一样、内容错位一帧的情形它挡不住。

而这个错位是这条链路上最难发现的故障：图能显示，框能画，标注能保存，导出
能跑通 —— 只有号码和人对不上，而对不上又会被当成「标注员看错了」。

所以每换一个视频都要跑一次第一步，用眼睛确认。这一步不能自动化：能自动判
的东西 B1a 已经自动判了，剩下的正是只有人能判的部分。

导入做什么
----------
逐条读 manifest.jsonl，对每一条：

    校验     data/frames/{video_id}/frame_{N:06d}.jpg 真实存在，否则跳过并计数
    门限     crop_w/crop_h 低于 settings.min_crop_width/height 的跳过并计数
    拷贝     crop -> data/crops/{video_id}/<原文件名>
    落库     一行 JerseyPrediction，jersey_bbox 是 mine 补边后的框（和推理时
             crop_jersey 切的完全一致），source 记 mine_detector 或
             mine_torso_fallback，crop_path 指向拷过去的那张图

幂等键是 (video_id, crop_path)。mine 的文件名 f{frame_idx}__box_{box_idx}.png
对 (video, frame_idx, box_idx) 唯一，所以重复导入不会产生重复行。

导入完核对库里行数 == 磁盘上文件数，不一致直接非零退出 —— 这两个数只要
分家，后面所有的统计就都是错的。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

#: 自检抽几帧。5 张够看出整体错位，多了人不会真的一张张看。
FRAMECHECK_SAMPLES = 5


def _imread(path: Path):
    """中文路径下 cv2.imread 会静默返回 None，走 fromfile 绕开。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite(path: Path, image) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _read_manifest(manifest_path: Path, video_filter: str | None) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[FAIL] {manifest_path}:{line_no} 不是合法 JSON: {exc}")
            if video_filter and record.get("video") != video_filter:
                continue
            records.append(record)
    return records


def _frame_path(data_dir: Path, video_id: int, frame_idx: int) -> Path:
    return data_dir / "frames" / str(video_id) / f"frame_{frame_idx:06d}.jpg"


# --------------------------------------------------------------------------
# 第一步：帧号一致性自检
# --------------------------------------------------------------------------


def framecheck(
    records: list[dict],
    crops_root: Path,
    data_dir: Path,
    video_id: int,
    seed: int,
) -> int:
    frame_indices = sorted({int(r["frame_idx"]) for r in records})
    if not frame_indices:
        print("[FAIL] manifest 里一条记录都没有（--video 过滤掉了？）")
        return 1

    rng = random.Random(seed)
    sample = sorted(rng.sample(frame_indices, min(FRAMECHECK_SAMPLES, len(frame_indices))))
    by_frame: dict[int, dict] = {}
    for record in records:
        by_frame.setdefault(int(record["frame_idx"]), record)

    debug_dir = data_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("帧号一致性自检 —— 只出图，不写库")
    print("=" * 74)
    print(f"manifest 覆盖帧号: {len(frame_indices)} 个, "
          f"{frame_indices[0]} .. {frame_indices[-1]}")
    print(f"抽查: {sample}")
    print()

    written: list[Path] = []
    for frame_idx in sample:
        web_path = _frame_path(data_dir, video_id, frame_idx)
        record = by_frame[frame_idx]
        crop_path = crops_root / record["crop_path"]

        web_image = _imread(web_path) if web_path.is_file() else None
        if web_image is None:
            print(f"  [FAIL] 帧 {frame_idx}: 网页帧读不出来 -> {web_path}")
            print("         video_id 对不对？这个视频抽帧时的 frame_interval 是多少？")
            return 1

        # 右半边画的是 mine 那一帧上的框，画在网页的同一张帧上。
        # 直接贴 crop 小图看不出错位（20x30 的号码放大了也认不出是哪一帧），
        # 把框画回网页帧才是真正在验证「这两个帧号指的是同一画面」。
        annotated = web_image.copy()
        for r in records:
            if int(r["frame_idx"]) != frame_idx:
                continue
            x1, y1, x2, y2 = [int(v) for v in r["jersey_bbox"]]
            px1, py1, px2, py2 = [int(v) for v in r["player_bbox"]]
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 180, 0), 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label_left = f"WEB data/frames/{video_id}/frame_{frame_idx:06d}.jpg"
        label_right = f"MINE boxes @ frame_idx={frame_idx}  ({record['source']})"
        for image, text in ((web_image, label_left), (annotated, label_right)):
            cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.hstack([web_image, annotated])
        out_path = debug_dir / f"framecheck_{frame_idx:06d}.png"
        if not _imwrite(out_path, combined):
            print(f"  [FAIL] 写不出 {out_path}")
            return 1
        written.append(out_path)
        crop_note = "" if crop_path.is_file() else "   ⚠ crop 文件缺失"
        print(f"  帧 {frame_idx:>6}: {out_path}{crop_note}")

    print()
    print("-" * 74)
    print("现在用眼睛看这 5 张图：")
    for path in written:
        print(f"    {path}")
    print()
    print("  左边是网页会展示给标注员的那一帧，右边是同一张帧 + mine 在该帧号上")
    print("  给出的框（绿=球员框，红=球衣框，红框就是将来要标的那个）。")
    print()
    print("  要确认的是两件事：")
    print("    1. 左右是同一个画面（本来就是同一张图，主要看框有没有画歪）;")
    print("    2. 红框确实套在球员的号码区域上，而不是套在空地或别的球员身上。")
    print()
    print("  红框整体偏到别的球员身上 = 帧号错位，**不要导入**，回去查 B1a 的")
    print("  转码帧数校验和这个视频是不是 VFR 素材。")
    print("-" * 74)
    print()
    print("确认无误后，加 --confirmed 重跑同一条命令即可导入。")
    return 0


# --------------------------------------------------------------------------
# 第二步：导入
# --------------------------------------------------------------------------


def do_import(
    records: list[dict],
    crops_root: Path,
    data_dir: Path,
    video_id: int,
) -> int:
    from app.core.config import settings
    from app.db.session import SessionLocal, ensure_sqlite_schema
    from app.models.dataset import (
        MINE_SOURCE_MAP,
        PREDICTION_SOURCE_UNKNOWN,
        JerseyPrediction,
        VideoAsset,
        VideoFrame,
    )

    ensure_sqlite_schema()

    min_w = settings.min_crop_width
    min_h = settings.min_crop_height

    db = SessionLocal()
    try:
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if video is None:
            print(f"[FAIL] 库里没有 video_id={video_id}。先在网页上传这个视频。")
            return 1
        print(f"目标视频: id={video_id}  {video.filename}")
        print(f"          fps={video.fps}  frame_interval={video.frame_interval}  "
              f"decode_path={video.decode_path}")

        frame_rows = {
            int(row.frame_id): row
            for row in db.query(VideoFrame).filter(VideoFrame.video_id == video_id).all()
        }
        existing_crop_paths = {
            row.crop_path
            for row in db.query(JerseyPrediction)
            .filter(JerseyPrediction.video_id == video_id)
            .all()
            if row.crop_path
        }

        crops_dest = data_dir / "crops" / str(video_id)
        crops_dest.mkdir(parents=True, exist_ok=True)

        counts = Counter()
        source_counts = Counter()
        new_rows: list[JerseyPrediction] = []

        for record in records:
            frame_idx = int(record["frame_idx"])
            counts["manifest_rows"] += 1

            # 1. 帧必须真实存在。缺一张就跳过并计数 —— 静默失败在这条链路上
            #    等于把一批 crop 悄悄丢掉，而总数看起来还是对的。
            frame_row = frame_rows.get(frame_idx)
            if frame_row is None:
                counts["skipped_frame_not_in_db"] += 1
                continue
            frame_file = _frame_path(data_dir, video_id, frame_idx)
            if not frame_file.is_file():
                candidate = Path(frame_row.frame_path) if frame_row.frame_path else None
                if candidate is None or not candidate.is_file():
                    counts["skipped_frame_file_missing"] += 1
                    continue

            # 2. 尺寸门限。和 routes.list_review_items 读同一份配置。
            crop_w = int(record["crop_w"])
            crop_h = int(record["crop_h"])
            if crop_w < min_w or crop_h < min_h:
                counts["skipped_below_gate"] += 1
                continue

            # 3. 幂等：同一 (video_id, crop_path) 只进一行。
            rel_crop = str(Path(str(video_id)) / Path(record["crop_path"]).name).replace("\\", "/")
            if rel_crop in existing_crop_paths:
                counts["skipped_already_imported"] += 1
                continue

            source_crop = crops_root / record["crop_path"]
            if not source_crop.is_file():
                counts["skipped_crop_file_missing"] += 1
                continue

            shutil.copy2(source_crop, crops_dest / source_crop.name)

            manifest_source = str(record.get("source", ""))
            source = MINE_SOURCE_MAP.get(manifest_source, PREDICTION_SOURCE_UNKNOWN)
            if source == PREDICTION_SOURCE_UNKNOWN:
                counts["unknown_source_value"] += 1
            source_counts[source] += 1

            track_id = record.get("track_id")
            new_rows.append(
                JerseyPrediction(
                    video_id=video_id,
                    frame_row_id=frame_row.id,
                    frame_id=frame_idx,
                    # mine 没有 track 时填 0：这一列 NOT NULL，而 0 在本项目里
                    # 一直表示「没有归属」（见 SlotTracker 的 slot==0 约定）。
                    track_id=int(track_id) if track_id is not None else 0,
                    player_bbox=json.dumps([int(v) for v in record["player_bbox"]]),
                    jersey_bbox=json.dumps([int(v) for v in record["jersey_bbox"]]),
                    # mine 不跑 OCR，所以没有号码也没有置信度。留空而不是编一个，
                    # 空号码在 UI 上就是「等你填」，编出来的号码是要人去纠正的噪声。
                    predicted_number="",
                    confidence=0.0,
                    source=source,
                    crop_path=rel_crop,
                )
            )
            existing_crop_paths.add(rel_crop)
            counts["imported"] += 1

        if new_rows:
            db.add_all(new_rows)
            video.status = "recognition_ready"
            db.add(video)
            db.commit()

        # 4. 落库行数 vs 磁盘文件数。这两个数分家之后所有统计都是错的，
        #    所以直接非零退出，不让它变成一句 warning 被划过去。
        db_count = (
            db.query(JerseyPrediction)
            .filter(JerseyPrediction.video_id == video_id, JerseyPrediction.crop_path.isnot(None))
            .count()
        )
        disk_count = len([p for p in crops_dest.iterdir() if p.is_file()])
    finally:
        db.close()

    print("-" * 74)
    print(f"  manifest 行数        : {counts['manifest_rows']}")
    print(f"  导入                 : {counts['imported']}")
    print(f"  已存在（幂等跳过）   : {counts['skipped_already_imported']}")
    print(f"  门限丢弃 (<{min_w}x{min_h})   : {counts['skipped_below_gate']}")
    print(f"  帧不在库里           : {counts['skipped_frame_not_in_db']}")
    print(f"  帧文件缺失           : {counts['skipped_frame_file_missing']}")
    print(f"  crop 文件缺失        : {counts['skipped_crop_file_missing']}")
    if counts["unknown_source_value"]:
        print(f"  ⚠ source 值不认识    : {counts['unknown_source_value']}（已记为 unknown）")
    print()
    for source, n in sorted(source_counts.items()):
        print(f"  source={source:<22} {n}")
    print("-" * 74)
    print(f"  库中带 crop 的行     : {db_count}")
    print(f"  data/crops/{video_id} 文件数 : {disk_count}")

    if db_count != disk_count:
        print()
        print(f"[FAIL] 行数({db_count}) != 文件数({disk_count})。")
        print("       两者分家之后所有统计都是错的。可能原因：上一次导入中途失败、")
        print("       或者有人手动动过 data/crops。清掉这个视频的预测行重导：")
        print(f"       DELETE FROM jersey_predictions WHERE video_id={video_id};")
        return 1

    print()
    print("[ok] 导入完成，行数与文件数一致。")
    print(f"     打开网页选这个视频，点 Load Frames for Annotation 就能开始标。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 mine 挖出的 crop 导进标注库（默认只做帧号自检）")
    parser.add_argument("--manifest", required=True, help="mine 产出的 crops/manifest.jsonl")
    parser.add_argument("--video-id", type=int, required=True, help="网页库里的 video_id")
    parser.add_argument("--video", default=None,
                        help="manifest 里 video 字段的值。一个 manifest 装多个视频时用来过滤")
    parser.add_argument("--data-dir", default=None, help="数据目录，不给就用 DATA_DIR")
    parser.add_argument("--confirmed", action="store_true",
                        help="已经人工看过 framecheck 图，执行导入")
    parser.add_argument("--seed", type=int, default=0, help="自检抽帧的随机种子")
    args = parser.parse_args(argv)

    from app.core.config import settings

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        print(f"[FAIL] manifest 不存在: {manifest_path}")
        return 1
    crops_root = manifest_path.parent

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (_BACKEND / settings.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"[FAIL] 数据目录不存在: {data_dir}")
        return 1

    records = _read_manifest(manifest_path, args.video)
    if not records:
        print(f"[FAIL] {manifest_path} 里没有可用记录"
              + (f"（--video {args.video} 过滤后为空）" if args.video else ""))
        return 1

    videos_in_manifest = sorted({r.get("video", "?") for r in records})
    if len(videos_in_manifest) > 1:
        print(f"[FAIL] manifest 里有多个视频: {videos_in_manifest}")
        print("       用 --video <名字> 指定这次要导哪一个。混着导会把不同视频的")
        print("       帧号当成同一套，框会落到错误的帧上。")
        return 1

    print(f"manifest : {manifest_path}  ({len(records)} 条, video={videos_in_manifest[0]})")
    print(f"数据目录 : {data_dir}")
    print()

    if not args.confirmed:
        return framecheck(records, crops_root, data_dir, args.video_id, args.seed)
    return do_import(records, crops_root, data_dir, args.video_id)


if __name__ == "__main__":
    raise SystemExit(main())
