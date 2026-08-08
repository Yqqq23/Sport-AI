"""Minimum end-to-end test that needs no trained weights.

Runs detect -> track -> render over a handful of frames with official yolo26n.pt
and checks that nothing crashes and that both videos are actually written.

This proves the plumbing works. It does NOT check that results are correct:
yolo26n.pt is a COCO model, so it finds "person" under class id 0, which this
project reads as "player". Ball, rim and jersey will be empty. That is expected.

    python scripts/smoke_test.py --input clip.mp4
    python scripts/smoke_test.py --input clip.mp4 --frames 60 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from detect import Detector, DetectorConfig, split_by_class  # noqa: E402
from render import (  # noqa: E402
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
from track import Tracker, TrackerConfig  # noqa: E402

PASS = "[ OK ]"
FAIL = "[FAIL]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline smoke test, no trained weights needed")
    parser.add_argument("--input", type=Path, required=True, help="Any short video")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo26n.pt"),
        help="Official checkpoint. Passed explicitly, so this is not a silent COCO fallback.",
    )
    parser.add_argument("--frames", type=int, default=30, help="Frames to process")
    parser.add_argument("--imgsz", type=int, default=640, help="Small for speed; not for real runs")
    parser.add_argument("--device", default="0", help='"0" or "cpu"')
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Backend to smoke test")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "smoke_test")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    import cv2

    print("=" * 66)
    print("basketball-v2 smoke test")
    print("  Plumbing only. yolo26n.pt is a COCO model, so 'player' is really")
    print("  'person' and ball / rim / jersey will be empty. That is expected.")
    print("=" * 66)

    if not args.input.exists():
        print(f"{FAIL} input video not found: {args.input}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    # ------------------------------------------------------------ detect
    print("\n1. Detector")
    try:
        detector = Detector(
            DetectorConfig(
                weights=args.weights,
                imgsz=args.imgsz,
                device=args.device,
                require_basketball_classes=False,
            )
        )
        print(f"{PASS} loaded {args.weights.name} ({len(detector.names)} classes)")
        print(f"       basketball 5-class model: {detector.is_basketball_model}")
    except Exception as exc:
        print(f"{FAIL} detector failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        print(f"{FAIL} cannot open video: {args.input}")
        return 1
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    ok, first_frame = capture.read()
    if not ok:
        print(f"{FAIL} cannot read a frame from {args.input}")
        capture.release()
        return 1
    try:
        detections = detector.detect(first_frame)
        grouped = split_by_class(detections)
        print(f"{PASS} detect() returned {len(detections)} detection(s) on frame 0")
        print(f"       by class: {({k: len(v) for k, v in grouped.items()})}")
        if detections:
            sample = detections[0]
            assert isinstance(sample.bbox_xyxy, tuple) and len(sample.bbox_xyxy) == 4
            print(f"{PASS} Detection dataclass shape verified: {sample}")
    except Exception as exc:
        print(f"{FAIL} detect() failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failures += 1
    capture.release()

    # ------------------------------------------------------------- track
    print("\n2. Tracker")
    try:
        tracker = Tracker(
            weights=args.weights,
            config=TrackerConfig(
                tracker=args.tracker,
                imgsz=args.imgsz,
                device=args.device,
                with_reid=False,
            ),
        )
        print(f"{PASS} backend resolved: {tracker.backend_name}")
    except Exception as exc:
        print(f"{FAIL} tracker init failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    # ------------------------------------------------- track + render loop
    print("\n3. Track + render loop")
    court_config = CourtRenderConfig()
    court_template = draw_half_court_template(court_config)
    trails = TrailHistory(trail_length=court_config.trail_length, alpha=0.5)
    ball_trail = make_ball_trail(court_config.trail_length)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    annotated_path = args.output_dir / "annotated.mp4"
    topdown_path = args.output_dir / "topdown.mp4"
    writer_annotated = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
    writer_topdown = cv2.VideoWriter(
        str(topdown_path), fourcc, fps, (court_config.width, court_config.height)
    )

    capture = cv2.VideoCapture(str(args.input))
    processed = 0
    total_players = 0
    ids_seen: set[int] = set()
    started = time.perf_counter()
    try:
        while processed < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            tracked = tracker.update(frame, frame_idx=processed, timestamp_sec=processed / fps)
            total_players += len(tracked.players)
            for player in tracked.players:
                if player.track_id is not None:
                    ids_seen.add(player.track_id)

            # Court projection is skipped: smoke test has no calibration, so
            # top-down points are laid out on a simple normalised grid purely to
            # exercise the renderer.
            render_data = FrameRenderData(
                frame_idx=processed,
                timestamp_sec=processed / fps,
                players=[
                    PlayerRender(
                        track_id=p.track_id if p.track_id is not None else -1,
                        bbox_xyxy=list(p.bbox_xyxy),
                        jersey_number=None,
                        conf=p.confidence,
                        court_xy=(
                            court_config.margin
                            + (p.ground_xy[0] / max(1, width))
                            * (court_config.width - 2 * court_config.margin),
                            court_config.margin
                            + (p.ground_xy[1] / max(1, height))
                            * (court_config.height - 2 * court_config.margin),
                        ),
                    )
                    for p in tracked.players
                ],
                ball=(
                    BallRender(bbox_xyxy=list(tracked.ball.bbox_xyxy), conf=tracked.ball.confidence)
                    if tracked.ball is not None
                    else None
                ),
                rims=[RimRender(bbox_xyxy=list(r.bbox_xyxy), conf=r.confidence) for r in tracked.rims],
            )

            writer_annotated.write(render_annotated_frame(frame, render_data, line_width=2))
            writer_topdown.write(
                render_topdown_frame(
                    court_template=court_template,
                    data=render_data,
                    trails=trails,
                    config=court_config,
                    ball_trail=ball_trail,
                )
            )
            processed += 1
    except Exception as exc:
        print(f"{FAIL} loop failed at frame {processed}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failures += 1
    finally:
        capture.release()
        writer_annotated.release()
        writer_topdown.release()

    elapsed = time.perf_counter() - started
    if processed == 0:
        print(f"{FAIL} no frames processed")
        failures += 1
    else:
        print(f"{PASS} processed {processed} frame(s) in {elapsed:.1f}s")
        print(f"       {total_players / processed:.2f} tracked object(s)/frame, "
              f"{len(ids_seen)} unique id(s)")
        video_seconds = processed / fps
        print(f"       {elapsed / video_seconds:.2f}x realtime at imgsz={args.imgsz} "
              f"(smoke settings, not representative)")

    # ------------------------------------------------------------ outputs
    print("\n4. Outputs")
    for label, path in (("annotated", annotated_path), ("topdown", topdown_path)):
        if path.exists() and path.stat().st_size > 0:
            print(f"{PASS} {label}: {path} ({path.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"{FAIL} {label} was not written or is empty: {path}")
            failures += 1

    print()
    print("=" * 66)
    if failures:
        print(f"SMOKE TEST FAILED: {failures} problem(s)")
    else:
        print("SMOKE TEST PASSED. The pipeline runs end to end.")
        print("Next: train the 5-class detector, then run src/pipeline.py.")
    print("=" * 66)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
