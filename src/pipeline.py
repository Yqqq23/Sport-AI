"""End-to-end pipeline: video in, three outputs out.

  output 1  annotated.mp4  original view + player/ball/rim boxes + jersey numbers
  output 2  topdown.mp4    players as dots on the court template, with trails
  output 3  shots.csv / shots.json   per-player made-shot records

Usage:
    python src/pipeline.py --input game.mp4 --config configs/default.yaml \
        --output-dir outputs/run1 [--tracker botsort.yaml]

Orchestration only. Every algorithm lives in its own module; this file decides
what runs on which frame and in what order, and owns nothing else.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Allow both `python src/pipeline.py` and `python -m src.pipeline`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from calib import CourtProjector, load_court_calibration
from config import AppConfig, load_config
from jersey import JerseyNumberStabilizer, PytesseractJerseyOCR, should_run_ocr
from render import (
    BallRender,
    CourtRenderConfig,
    FrameRenderData,
    PlayerRender,
    RimRender,
    TrailHistory,
    draw_half_court_template,
    make_ball_trail,
    render_annotated_frame,
    render_topdown_frame,
)
from shot import ShotAnalyzer, make_player_box
from stats import ShotRecorder
from track import Tracker

LOGGER = logging.getLogger(__name__)


def crop_bbox(frame, bbox) -> Optional["cv2.Mat"]:
    """Crop a bbox from a frame, or None if the box is degenerate."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def build_projector(config: AppConfig) -> Optional[CourtProjector]:
    """Load the one-shot homography, or None when no calibration is configured."""
    if config.calibration_json is None:
        LOGGER.warning(
            "No calibration_json configured. The top-down video and the court_x/court_y "
            "columns of output 3 will be unavailable."
        )
        return None
    calibration = load_court_calibration(config.calibration_json)
    projector = CourtProjector(calibration, canvas=config.canvas)
    LOGGER.info(
        "Homography solved once from %s calibration point(s): %s",
        len(calibration.point_names),
        config.calibration_json,
    )
    return projector


def run_pipeline(
    input_video: Path,
    config: AppConfig,
    tracker_override: Optional[str] = None,
) -> dict[str, Path]:
    """Process one video and write all three outputs."""
    input_video = Path(input_video)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if tracker_override:
        LOGGER.info("Tracker overridden on the command line: %s", tracker_override)
        config.tracker.tracker = tracker_override

    problems = config.validate()
    if problems:
        raise SystemExit(
            "Config is not usable:\n  - " + "\n  - ".join(problems)
        )

    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    LOGGER.info(
        "Input: %s  %sx%s  %.2f fps  %s frames", input_video.name, width, height, fps, source_frames
    )

    projector = build_projector(config)
    tracker = Tracker(weights=config.detector.weights, config=config.tracker)
    stabilizer = JerseyNumberStabilizer(config.jersey.stabilizer)
    ocr = PytesseractJerseyOCR()
    shot_analyzer = ShotAnalyzer(config.shot, fps=fps)
    recorder = ShotRecorder()

    if config.jersey.enabled and not ocr.is_available():
        LOGGER.warning(
            "Jersey OCR is enabled but the tesseract binary was not found. "
            "Numbers will stay unknown and output 3 will fall back to track ids. "
            "Install tesseract-ocr, or set jersey.enabled=false to silence this."
        )

    court_config: CourtRenderConfig = config.render.court
    court_template = draw_half_court_template(court_config)
    trails = TrailHistory(
        trail_length=court_config.trail_length, alpha=config.render.trail_alpha
    )
    ball_trail = make_ball_trail(court_config.trail_length)

    fourcc = cv2.VideoWriter_fourcc(*config.output.fourcc)
    annotated_path = output_dir / config.output.annotated_video_name
    topdown_path = output_dir / config.output.topdown_video_name
    writer_annotated = (
        cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
        if config.output.write_annotated
        else None
    )
    writer_topdown = (
        cv2.VideoWriter(str(topdown_path), fourcc, fps, (court_config.width, court_config.height))
        if config.output.write_topdown
        else None
    )

    frame_idx = 0
    ocr_calls = 0
    ocr_skipped = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if config.max_frames is not None and frame_idx >= config.max_frames:
                break

            timestamp_sec = frame_idx / max(fps, 1e-6)
            tracked = tracker.update(frame, frame_idx=frame_idx, timestamp_sec=timestamp_sec)

            # --- jersey numbers -------------------------------------------
            run_ocr_this_frame = (
                config.jersey.enabled
                and ocr.is_available()
                and frame_idx % max(1, config.jersey.ocr_interval) == 0
            )
            if run_ocr_this_frame:
                for jersey_box in tracked.jerseys:
                    # Jersey ownership is resolved against tracked.players ONLY.
                    # Referees (CLASS_REFEREE) are already absent from that list
                    # by the role split in src/track.py, so a jersey box drawn on
                    # a referee can never acquire an owner and never votes. That
                    # is intended: referee numbers are not player numbers, and
                    # letting one bind would corrupt a real player's vote pool.
                    owner = _nearest_player(jersey_box, tracked.players)
                    if owner is None or owner.track_id is None:
                        continue
                    crop = crop_bbox(frame, jersey_box.bbox_xyxy)
                    if crop is None or not should_run_ocr(
                        list(jersey_box.bbox_xyxy), config.jersey.gate, crop
                    ):
                        ocr_skipped += 1
                        continue
                    text, confidence = ocr.recognize(crop)
                    ocr_calls += 1
                    stabilizer.update(
                        track_id=owner.track_id,
                        raw_text=text,
                        confidence=confidence,
                        frame_idx=frame_idx,
                    )

            # --- shot detection -------------------------------------------
            rim_bbox = (
                max(tracked.rims, key=lambda r: r.confidence).bbox_xyxy if tracked.rims else None
            )
            ball_center = tracked.ball.center_xy if tracked.ball is not None else None
            player_boxes = [
                make_player_box(p.track_id, p.bbox_xyxy, p.ground_xy) for p in tracked.players
            ]
            event = shot_analyzer.update(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                ball_center=ball_center,
                rim_bbox=rim_bbox,
                players=player_boxes,
            )
            if event is not None:
                if event.shooter_track_id is not None:
                    event.shooter_jersey_number = stabilizer.get_stable_number(
                        event.shooter_track_id
                    )
                if projector is not None and event.shot_position_xy is not None:
                    event.court_xy = projector.project_point(*event.shot_position_xy)
                recorder.add_event(event)

            # --- rendering -------------------------------------------------
            render_data = _build_render_data(
                tracked=tracked,
                stabilizer=stabilizer,
                projector=projector,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            )
            if writer_annotated is not None:
                writer_annotated.write(
                    render_annotated_frame(
                        frame, render_data, line_width=config.render.line_width
                    )
                )
            if writer_topdown is not None:
                writer_topdown.write(
                    render_topdown_frame(
                        court_template=court_template,
                        data=render_data,
                        trails=trails,
                        config=court_config,
                        ball_trail=ball_trail,
                    )
                )

            if frame_idx % 100 == 0:
                LOGGER.info("frame %s", frame_idx)
            frame_idx += 1
    finally:
        capture.release()
        if writer_annotated is not None:
            writer_annotated.release()
        if writer_topdown is not None:
            writer_topdown.release()

    elapsed = time.perf_counter() - started
    video_seconds = frame_idx / max(fps, 1e-6)

    csv_path, json_path = recorder.write(
        output_dir=output_dir,
        stem=config.output.shots_stem,
        meta={
            "input_video": input_video.name,
            "frames_processed": frame_idx,
            "fps": round(fps, 3),
            "tracker": tracker.backend_name,
            "reid_active": tracker.reid_active,
            "weights": Path(config.detector.weights).name,
            "elapsed_sec": round(elapsed, 2),
            "realtime_ratio": round(elapsed / video_seconds, 3) if video_seconds > 0 else None,
            "ocr_calls": ocr_calls,
            "ocr_skipped_by_gate": ocr_skipped,
        },
    )

    print(recorder.format_summary())
    print(f"  tracker            {tracker.backend_name} (reid={tracker.reid_active})")
    print(f"  frames processed   {frame_idx}")
    print(
        f"  elapsed            {elapsed:.1f}s for {video_seconds:.1f}s of video "
        f"(x{elapsed / video_seconds:.2f} realtime)"
        if video_seconds > 0
        else f"  elapsed            {elapsed:.1f}s"
    )
    print(f"  jersey OCR         {ocr_calls} run, {ocr_skipped} skipped by the size/blur gate")
    if projector is None:
        print("  NOTE               no calibration, so court_x/court_y are empty")
    print()

    outputs = {"shots_csv": csv_path, "shots_json": json_path}
    if writer_annotated is not None:
        outputs["annotated"] = annotated_path
    if writer_topdown is not None:
        outputs["topdown"] = topdown_path
    for name, path in outputs.items():
        print(f"  {name:<18} {path}")
    return outputs


def _nearest_player(jersey_box, players):
    """Attach a jersey box to the player box that contains or is nearest its centre.

    ``players`` never contains referees: src/track.py drops CLASS_REFEREE during
    the role split. Jersey numbers therefore bind to players only, by
    construction rather than by a check here.
    """
    if not players:
        return None
    cx, cy = jersey_box.center_xy

    def distance(player) -> float:
        x1, y1, x2, y2 = player.bbox_xyxy
        dx = max(x1 - cx, 0.0, cx - x2)
        dy = max(y1 - cy, 0.0, cy - y2)
        return (dx * dx + dy * dy) ** 0.5

    best = min(players, key=distance)
    # A jersey box whose centre is outside every player box is not attachable.
    return best if distance(best) == 0.0 else None


def _build_render_data(tracked, stabilizer, projector, frame_idx, timestamp_sec) -> FrameRenderData:
    players: list[PlayerRender] = []
    for player in tracked.players:
        court_xy = None
        if projector is not None:
            court_xy = projector.project_point(*player.ground_xy)
        players.append(
            PlayerRender(
                track_id=player.track_id if player.track_id is not None else -1,
                bbox_xyxy=list(player.bbox_xyxy),
                jersey_number=(
                    stabilizer.get_stable_number(player.track_id)
                    if player.track_id is not None
                    else None
                ),
                conf=player.confidence,
                court_xy=court_xy,
            )
        )

    ball = None
    if tracked.ball is not None:
        ball_court = None
        if projector is not None:
            ball_court = projector.project_point(*tracked.ball.center_xy)
        ball = BallRender(
            bbox_xyxy=list(tracked.ball.bbox_xyxy),
            conf=tracked.ball.confidence,
            track_id=tracked.ball.track_id,
            court_xy=ball_court,
        )

    return FrameRenderData(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        players=players,
        ball=ball,
        rims=[RimRender(bbox_xyxy=list(r.bbox_xyxy), conf=r.confidence) for r in tracked.rims],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basketball 2D analysis: annotated video, top-down video, made-shot records"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="Config YAML"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output.output_dir from the config"
    )
    parser.add_argument(
        "--tracker",
        default=None,
        help="Override tracker.tracker, e.g. botsort.yaml or configs/trackers/bytetrack_sportcv.yaml",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    if args.max_frames is not None:
        config.max_frames = args.max_frames

    run_pipeline(input_video=args.input, config=config, tracker_override=args.tracker)


if __name__ == "__main__":
    main()
