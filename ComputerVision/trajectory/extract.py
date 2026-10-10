"""逐帧抽取球员位置，投影成球场米坐标，流式写 CSV。

    python -m trajectory.cli extract --video game.mp4 --out outputs/run1

做什么
------
每一帧：
    1. basketball-v2 的 Tracker 给出球员框
    2. SlotTracker 按位置分配槽位（不依赖 MOT 的 track_id —— 那边实测经常给不出）
    3. 到点就跑球衣号 OCR，喂给 JerseyNumberStabilizer 投票
    4. PlayerIdentityRegistry 把槽位粘到球衣号上（某帧没跑 OCR 也能保持身份）
    5. **用本项目的米制单应**把检测框底边中点投影成球场坐标
    6. 一行一条记录写进 CSV

附加输出
--------
除轨迹 CSV 外，默认复用 basketball-v2 的原画面渲染器写出 annotated.mp4，
包含球员、稳定球衣号、篮球、篮筐和裁判检测框。裁判在 basketball-v2 中是
逐帧检测而非 MOT，不会显示跨帧 referee track id。

流式写盘
--------
26341 帧跑一趟要很久。CSV 是**边跑边写**的，中途挂了已经跑完的部分不丢，
用 ``--start-frame`` 接着跑就行。
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .bridge import Backend
from .court import CourtProjector

LOGGER = logging.getLogger(__name__)

FIELDS = [
    "frame_idx", "timestamp_sec", "slot", "identity_key", "jersey_number",
    "conf", "x1", "y1", "x2", "y2", "foot_px_x", "foot_px_y",
    "court_x_m", "court_y_m", "in_court", "sane",
]


@dataclass(frozen=True)
class _TorsoBox:
    """给球员框套一个几何躯干区域，当检测器没有 jersey 类时用。

    只 duck-type 出球衣那条链路需要的两个成员，这样它走的是和真检测框
    完全相同的 ownership / gate / crop / OCR 代码 —— 分开写就是让裁剪逻辑
    有第二个地方可以漂移。（做法照搬 basketball-v2/src/pipeline.py。）
    """

    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _nearest_player(jersey_box, players, max_distance_ratio: float = 0.25):
    """把一个球衣框判给最近的球员框。照搬 basketball-v2 的判据。"""
    if not players:
        return None
    jx, jy = jersey_box.center_xy
    best, best_d = None, None
    for p in players:
        x1, y1, x2, y2 = p.bbox_xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d = ((cx - jx) ** 2 + (cy - jy) ** 2) ** 0.5
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if diag <= 0 or d > diag * max_distance_ratio * 4:
            continue
        if best_d is None or d < best_d:
            best, best_d = p, d
    return best


@dataclass
class ExtractSummary:
    frames_read: int = 0
    frames_processed: int = 0
    rows_written: int = 0
    ocr_calls: int = 0
    ocr_hits: int = 0
    out_of_court: int = 0
    insane: int = 0
    elapsed_sec: float = 0.0
    numbers_seen: Optional[dict] = None
    annotated_video: Optional[str] = None
    #: 实际选中的 OCR 后端类名。配置里写的是 backend: auto，到底落到 local /
    #: parseq / paddleocr / easyocr / pytesseract 哪一个，只有运行时才知道。
    #: 不记进产出文件的话，事后拿着 tracks.csv 根本回答不了
    #: 「这些球衣号是什么认出来的」——这个坑踩过一次了。
    ocr_backend: Optional[str] = None
    ocr_backend_available: bool = False
    #: 视频写完后回读校验的结果。记进 json 是为了让你**下载之后**能和本地
    #: 文件比对：字节数对不上就是传输被截断，而不是代码或编码器的问题。
    #: 这个坑踩过 —— 一次下载在 5 MiB 处被砍断，mp4 丢了尾部的 moov 索引，
    #: 任何播放器都打不开，看起来像是「视频没生成」。
    annotated_bytes: int = 0
    annotated_frames: int = 0
    annotated_ok: bool = False

    def report(self) -> str:
        fps = self.frames_processed / max(self.elapsed_sec, 1e-9)
        lines = [
            f"读入帧数     : {self.frames_read}",
            f"处理帧数     : {self.frames_processed}",
            f"写出记录     : {self.rows_written}",
            f"OCR 后端     : {self.ocr_backend or '(无)'}"
            + ("" if self.ocr_backend_available else "   ← 加载失败，球衣号会全空"),
            f"OCR 调用/命中: {self.ocr_calls} / {self.ocr_hits}",
            f"投影出界     : {self.out_of_court}  （在场地外但还算合理，保留）",
            f"投影离谱     : {self.insane}  （离场地 >6 m，标记 sane=0）",
            f"耗时         : {self.elapsed_sec:.1f} s  ({fps:.2f} frame/s)",
        ]
        if self.numbers_seen:
            top = sorted(self.numbers_seen.items(), key=lambda kv: -kv[1])
            lines.append("识别到的球衣号（按出现帧数）:")
            for num, n in top:
                lines.append(f"    #{num:<4} {n} 帧")
        if self.annotated_video:
            lines.append(f"标注视频     : {self.annotated_video}")
            lines.append(
                f"             {self.annotated_bytes:,} 字节 "
                f"({self.annotated_bytes/1048576:.1f} MiB), 回读 {self.annotated_frames} 帧"
                + ("  ✅" if self.annotated_ok else "  ❌ 回读失败，文件可能没写完"))
            if self.annotated_ok:
                lines.append(
                    f"             下载后请核对字节数：不等于 {self.annotated_bytes:,} "
                    "就是传输被截断（mp4 的 moov 索引在文件末尾，缺了就打不开）")
        return "\n".join("  " + s for s in lines)


def extract(
    video: str | Path,
    out_csv: str | Path,
    *,
    repo: str | Path | None = None,
    config_path: str | Path | None = None,
    calibration: str | Path | None = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    duration_sec: Optional[float] = None,
    stride: int = 1,
    log_every: int = 200,
    annotated_video: str | Path | None = None,
    write_annotated: bool = True,
    annotated_fourcc: str = "mp4v",
    annotated_scale: float = 1.0,
) -> ExtractSummary:
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    backend = Backend(repo)
    LOGGER.info(backend.describe())
    cfg_path = Path(config_path) if config_path else backend.default_config_path
    if not Path(cfg_path).is_file():
        raise FileNotFoundError(
            f"basketball-v2 的配置文件不存在: {cfg_path}\n"
            "  用 --config 指定，或确认仓库里有 configs/default.yaml"
        )
    config = backend.load_config(cfg_path)

    projector = CourtProjector(calibration)
    LOGGER.info("court projector:\n%s", projector.describe())

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # 分辨率不符直接炸：单应和像素系绑死，硬算只会得到整体偏移的错误坐标
    projector.check_frame(width, height)
    LOGGER.info("video %s  %dx%d  %.2f fps  %d frames", video.name, width, height, fps, total)
    if duration_sec is not None:
        if duration_sec <= 0:
            raise ValueError("duration_sec must be greater than 0")
        duration_end = int(start_frame + round(duration_sec * fps))
        end_frame = duration_end if end_frame is None else min(end_frame, duration_end)

    tracker = backend.make_tracker(config.detector.weights, config.tracker)
    slot_tracker = backend.make_slot_tracker(config.identity.max_players)
    identities = backend.make_identity_registry(config.identity.max_players)
    stabilizer = backend.make_stabilizer(config.jersey.stabilizer)
    ocr = backend.make_ocr(config.jersey)
    crop_jersey, gate_reason, GATE_OK, torso_region = backend.jersey_helpers
    (
        PlayerRender,
        BallRender,
        RimRender,
        RefereeRender,
        FrameRenderData,
        render_annotated_frame,
    ) = backend.annotated_rendering

    ocr_ready = bool(config.jersey.enabled and ocr.is_available())
    if config.jersey.enabled and not ocr_ready:
        LOGGER.warning(
            "球衣号 OCR 打开了但后端加载不了（配置的是 %s）。"
            "轨迹仍然会产出，但 jersey_number 会全是空，只能按 slot 区分球员。",
            config.jersey.backend,
        )

    summary = ExtractSummary(numbers_seen={})
    summary.ocr_backend = type(ocr).__name__
    summary.ocr_backend_available = ocr_ready
    LOGGER.info("jersey OCR backend = %s  (配置 %s, 可用=%s)",
                summary.ocr_backend, config.jersey.backend, ocr_ready)
    started = time.perf_counter()
    annotated_writer = None
    if write_annotated:
        annotated_path = Path(annotated_video) if annotated_video else out_csv.parent / "annotated.mp4"
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        output_fps = fps / max(1, stride)
        # 缩放只作用于**写出的视频**，检测/投影/CSV 全部仍在原始 1920x1080 上做 ——
        # 缩小的只是给人看的那份，数据精度不受影响。
        # mp4v 编码效率很差（实测 60 秒 1080p ≈ 92 MiB），传输很容易断；
        # 0.5 倍就是 1/4 像素量，体积大致降到 1/4，球衣号仍然看得清。
        if not (0.05 <= annotated_scale <= 1.0):
            raise ValueError(f"annotated_scale 必须在 0.05~1.0 之间，给的是 {annotated_scale}")
        aw = int(round(width * annotated_scale)) // 2 * 2      # 编码器要求偶数边长
        ah = int(round(height * annotated_scale)) // 2 * 2
        annotated_writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*annotated_fourcc),
            output_fps,
            (aw, ah),
        )
        if annotated_scale != 1.0:
            LOGGER.info("标注视频缩放 %.2fx -> %dx%d（检测和 CSV 仍用原始 %dx%d）",
                        annotated_scale, aw, ah, width, height)
        if not annotated_writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"打不开标注视频写出器: {annotated_path} (fourcc={annotated_fourcc})。"
                "可尝试 --annotated-fourcc MJPG --annotated-video outputs/run1/annotated.avi"
            )
        summary.annotated_video = str(annotated_path)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        frame_idx = int(start_frame)
        try:
            while True:
                if end_frame is not None and frame_idx >= end_frame:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                summary.frames_read += 1
                if stride > 1 and (frame_idx - start_frame) % stride != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / max(fps, 1e-6)
                tracked = tracker.update(frame, frame_idx=frame_idx, timestamp_sec=timestamp)

                slots = slot_tracker.assign([p.bbox_xyxy for p in tracked.players], frame_idx)
                slot_of = {id(p): s for p, s in zip(tracked.players, slots)}

                # ---- 球衣号 OCR ------------------------------------------
                if ocr_ready and frame_idx % max(1, config.jersey.ocr_interval) == 0:
                    sources = tracked.jerseys
                    if not sources and config.jersey.fallback_to_torso:
                        sources = [
                            _TorsoBox(bbox_xyxy=tuple(torso_region(
                                p.bbox_xyxy, config.jersey.torso_top,
                                config.jersey.torso_bottom, config.jersey.torso_width)))
                            for p in tracked.players
                        ]
                    for jbox in sources:
                        owner = _nearest_player(
                            jbox, tracked.players,
                            max_distance_ratio=config.jersey.owner_max_distance_ratio)
                        if owner is None:
                            continue
                        slot = slot_of.get(id(owner), 0)
                        if slot == 0:
                            continue
                        if config.jersey.skip_locked_tracks and stabilizer.is_locked(slot):
                            continue
                        crop = crop_jersey(frame, jbox.bbox_xyxy, config.jersey.gate)
                        if crop is None:
                            continue
                        if gate_reason(list(jbox.bbox_xyxy), config.jersey.gate, crop) != GATE_OK:
                            continue
                        text, conf = ocr.recognize(crop)
                        summary.ocr_calls += 1
                        if text is not None:
                            summary.ocr_hits += 1
                        stabilizer.update(track_id=slot, raw_text=text,
                                          confidence=conf, frame_idx=frame_idx)

                # ---- 投影 + 写行 -----------------------------------------
                render_players = []
                for player in tracked.players:
                    slot = slot_of.get(id(player), 0)
                    number = stabilizer.get_stable_number(slot) if slot else None
                    ident = identities.resolve(slot if slot else None, number,
                                               frame_idx=frame_idx)
                    x1, y1, x2, y2 = [float(v) for v in player.bbox_xyxy]
                    proj = projector.project_box((x1, y1, x2, y2))
                    if not proj.in_court:
                        summary.out_of_court += 1
                    if not proj.sane:
                        summary.insane += 1
                    if ident.number is not None:
                        summary.numbers_seen[ident.number] = (
                            summary.numbers_seen.get(ident.number, 0) + 1)
                    render_players.append(PlayerRender(
                        track_id=slot or int(player.track_id or 0),
                        bbox_xyxy=[x1, y1, x2, y2],
                        jersey_number=number,
                        conf=float(getattr(player, "confidence", 0.0) or 0.0),
                        court_xy=(proj.x_m, proj.y_m) if proj.sane else None,
                        identity=ident,
                    ))
                    writer.writerow({
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(timestamp, 4),
                        "slot": slot,
                        "identity_key": ident.key,
                        "jersey_number": "" if ident.number is None else ident.number,
                        "conf": round(float(getattr(player, "confidence", 0.0) or 0.0), 4),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "foot_px_x": round((x1 + x2) / 2.0, 2),
                        "foot_px_y": round(y2, 2),
                        "court_x_m": "" if not np.isfinite(proj.x_m) else round(proj.x_m, 4),
                        "court_y_m": "" if not np.isfinite(proj.y_m) else round(proj.y_m, 4),
                        "in_court": int(proj.in_court),
                        "sane": int(proj.sane),
                    })
                    summary.rows_written += 1

                if annotated_writer is not None:
                    render_data = FrameRenderData(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp,
                        players=render_players,
                        ball=(
                            BallRender(
                                bbox_xyxy=list(tracked.ball.bbox_xyxy),
                                conf=float(tracked.ball.confidence or 0.0),
                                track_id=tracked.ball.track_id,
                            )
                            if tracked.ball is not None else None
                        ),
                        rims=[
                            RimRender(bbox_xyxy=list(obj.bbox_xyxy), conf=float(obj.confidence or 0.0))
                            for obj in tracked.rims
                        ],
                        referees=[
                            RefereeRender(bbox_xyxy=list(obj.bbox_xyxy), conf=float(obj.confidence or 0.0))
                            for obj in tracked.referees
                        ],
                    )
                    drawn = render_annotated_frame(frame, render_data)
                    if (drawn.shape[1], drawn.shape[0]) != (aw, ah):
                        drawn = cv2.resize(drawn, (aw, ah), interpolation=cv2.INTER_AREA)
                    annotated_writer.write(drawn)

                summary.frames_processed += 1
                if log_every and summary.frames_processed % log_every == 0:
                    fh.flush()                      # 中途挂掉也不丢已跑的部分
                    el = time.perf_counter() - started
                    LOGGER.info("frame %d  rows %d  %.2f frame/s",
                                frame_idx, summary.rows_written,
                                summary.frames_processed / max(el, 1e-9))
                frame_idx += 1
        finally:
            cap.release()
            if annotated_writer is not None:
                annotated_writer.release()

    summary.elapsed_sec = time.perf_counter() - started
    LOGGER.info("写出 -> %s", out_csv)

    # ---- 标注视频收尾校验 -------------------------------------------------
    # release() 之后再把文件回读一遍。mp4 的 moov 索引写在 mdat **之后**，
    # 所以「文件存在且不小」根本不能说明它能播；只有真的重新打开、读出帧，
    # 才知道收尾成功了。字节数记进 summary，下载后一比对就能认出截断。
    if summary.annotated_video:
        vp = Path(summary.annotated_video)
        summary.annotated_bytes = vp.stat().st_size if vp.is_file() else 0
        probe = cv2.VideoCapture(str(vp))
        if probe.isOpened():
            summary.annotated_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok_read, _ = probe.read()
            summary.annotated_ok = bool(ok_read and summary.annotated_frames > 0)
        probe.release()
        if summary.annotated_ok:
            LOGGER.info("标注视频校验通过: %s  %.1f MiB  %d 帧",
                        vp.name, summary.annotated_bytes / 1048576,
                        summary.annotated_frames)
            if summary.annotated_frames < summary.frames_processed * 0.9:
                LOGGER.warning(
                    "回读只有 %d 帧，但处理了 %d 帧 —— 可能有帧没写进去。",
                    summary.annotated_frames, summary.frames_processed)
        else:
            LOGGER.error(
                "标注视频回读失败: %s（%s 字节）。文件写出来了但打不开，"
                "多半是磁盘写满或编码器收尾失败。换 --annotated-fourcc MJPG "
                "并把后缀改成 .avi 再试。",
                vp, f"{summary.annotated_bytes:,}")
    return summary
