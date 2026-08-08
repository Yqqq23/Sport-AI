"""Compare tracker backends on one video with one set of weights.

Runs every backend over the same footage and prints a comparison table, so the
backend choice is made from measurements rather than from reputation.

Reuses src/track.py -- it does NOT reimplement tracking. Every backend goes
through the same run_tracker_benchmark() the pipeline uses.

Works BEFORE any custom training: with --weights yolo26n.pt it falls back to
official COCO weights and tracks the person class only, which is enough to rank
backends on identity stability. Ball / rim / jersey metrics are then unavailable
and are reported as such rather than as zeros.

    python scripts/compare_trackers.py --input clip.mp4 --weights yolo26n.pt
    python scripts/compare_trackers.py --input clip.mp4 \
        --weights runs/detect/train/basketball_baseline/weights/best.pt \
        --save-videos
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from detect import CLASS_BALL, CLASS_PLAYER, CLASS_RIM  # noqa: E402
from track import (  # noqa: E402
    BUNDLED_TRACKERS,
    TrackerConfig,
    TrackerNotAvailableError,
    probe_video,
    run_tracker_benchmark,
)

LOGGER = logging.getLogger(__name__)

#: COCO person id, used when running official weights before custom training.
COCO_PERSON_CLASS = 0


@dataclass
class ContaminationStats:
    """Whether the MOT backend is assigning ids to non-player objects.

    This exists to test the ``restrict_tracker_to_players=False`` default. The
    rim is a STATIONARY object that players repeatedly pass in front of, which
    is exactly the situation where a motion/IoU tracker can swap identities. If
    these numbers are bad, flipping that config option to True is justified.
    """

    #: NOTE ball id contamination is deliberately NOT measured here. src/track.py
    #: routes the ball through BallTracker, which replaces whatever id the MOT
    #: backend assigned, so that id never reaches this script. The rim is the
    #: meaningful probe anyway: it is stationary, always in frame, and sits
    #: exactly where players cluster.
    tracked_rim_frames: int
    distinct_nonplayer_ids: int
    #: Player id switches observed inside the rim neighbourhood.
    id_switches_near_rim: int
    #: Player id switches observed anywhere, for comparison.
    id_switches_total: int
    rim_seen_frames: int

    @property
    def rim_switch_rate(self) -> float:
        """Near-rim switches as a share of all switches. High means clustering."""
        if self.id_switches_total <= 0:
            return 0.0
        return self.id_switches_near_rim / self.id_switches_total


def analyze_contamination(frames, rim_radius_scale: float = 2.0) -> ContaminationStats:
    """Measure non-player id assignment and id churn near the rim.

    A player box counts as "near the rim" when its centre lies within
    ``rim_radius_scale`` rim-widths of the rim centre.
    """
    nonplayer_ids: set[tuple[int, int]] = set()
    tracked_rim_frames = 0
    rim_seen_frames = 0

    previous_ids: set[int] = set()
    previous_near_rim: set[int] = set()
    switches_total = 0
    switches_near_rim = 0

    for frame in frames:
        rim = max(frame.rims, key=lambda r: r.confidence) if frame.rims else None
        if rim is not None:
            rim_seen_frames += 1
            rx1, ry1, rx2, ry2 = rim.bbox_xyxy
            rim_center = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
            rim_radius = max(1.0, rx2 - rx1) * rim_radius_scale
            if rim.track_id is not None:
                tracked_rim_frames += 1
                nonplayer_ids.add((CLASS_RIM, rim.track_id))
        else:
            rim_center = None
            rim_radius = 0.0

        for extra in frame.rims:
            if extra.track_id is not None:
                nonplayer_ids.add((CLASS_RIM, extra.track_id))

        current_ids: set[int] = set()
        current_near_rim: set[int] = set()
        for player in frame.players:
            if player.track_id is None:
                continue
            current_ids.add(player.track_id)
            if rim_center is not None:
                cx, cy = player.center_xy
                distance = ((cx - rim_center[0]) ** 2 + (cy - rim_center[1]) ** 2) ** 0.5
                if distance <= rim_radius:
                    current_near_rim.add(player.track_id)

        # A "switch" here is an id appearing that was not present last frame.
        # This is a proxy, not MOTA IDSW: it also counts genuine entries into
        # frame. It is still comparable ACROSS backends on the same footage,
        # which is all this table claims to do.
        appeared = current_ids - previous_ids
        switches_total += len(appeared)
        switches_near_rim += len(appeared & current_near_rim)

        previous_ids = current_ids
        previous_near_rim = current_near_rim

    return ContaminationStats(
        tracked_rim_frames=tracked_rim_frames,
        distinct_nonplayer_ids=len(nonplayer_ids),
        id_switches_near_rim=switches_near_rim,
        id_switches_total=switches_total,
        rim_seen_frames=rim_seen_frames,
    )


def make_video_writer(path: Path, fps: float, size: tuple[int, int], fourcc: str):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark tracker backends on one clip")
    parser.add_argument("--input", type=Path, required=True, help="Input video")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolo26n.pt"),
        help="Detector weights. Defaults to official yolo26n.pt so this runs before training.",
    )
    parser.add_argument(
        "--trackers",
        nargs="*",
        default=None,
        help="Backends to test. Defaults to all six bundled ones plus the tuned custom config.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tracker_comparison"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per backend")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Write one annotated video per backend for visual comparison",
    )
    parser.add_argument("--fourcc", default="mp4v")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not args.input.exists():
        print(f"Input video not found: {args.input}")
        return 1

    custom_tracker = REPO_ROOT / "configs" / "trackers" / "bytetrack_sportcv.yaml"
    tracker_names: list[str] = args.trackers or (
        list(BUNDLED_TRACKERS) + ([str(custom_tracker)] if custom_tracker.exists() else [])
    )

    info = probe_video(args.input)
    # Detect whether these weights know the basketball classes, so ball/rim
    # columns are reported as unavailable rather than as a misleading zero.
    from ultralytics import YOLO

    probe_names = getattr(YOLO(str(args.weights)), "names", {}) or {}
    lowered = {int(k): str(v).lower() for k, v in probe_names.items()}
    basketball_model = lowered.get(CLASS_PLAYER) == "player" and lowered.get(CLASS_RIM) == "rim"

    base_config = TrackerConfig(
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        generated_tracker_dir=args.output_dir / ".generated_trackers",
    )

    print("=" * 100)
    print("TRACKER COMPARISON")
    print(f"  video    {args.input.name}  {info.width}x{info.height}  "
          f"{info.fps:.1f}fps  {info.frame_count} frames ({info.duration_sec:.1f}s)")
    print(f"  weights  {args.weights.name}")
    if basketball_model:
        print("  classes  basketball 4-class model: player/ball/rim/jersey")
    else:
        print("  classes  NOT the basketball model -> tracking the person class only.")
        print("           Identity-stability columns are still valid and comparable;")
        print("           ball / rim columns are unavailable and shown as '-'.")
    print("=" * 100)

    results = []
    for name in tracker_names:
        label = Path(name).name
        writer = {"handle": None}

        def callback(frame, frame_tracks, _label=label, _writer=writer):
            if not args.save_videos:
                return
            if _writer["handle"] is None:
                _writer["handle"] = make_video_writer(
                    args.output_dir / f"{Path(_label).stem}.mp4",
                    info.fps,
                    (frame.shape[1], frame.shape[0]),
                    args.fourcc,
                )
            _writer["handle"].write(_annotate(frame, frame_tracks))

        print(f"\n>>> {label}")
        try:
            frames, summary = run_tracker_benchmark(
                video=args.input,
                tracker_name=name,
                weights=args.weights,
                config=base_config,
                max_frames=args.max_frames,
                frame_callback=callback if args.save_videos else None,
            )
        except TrackerNotAvailableError as exc:
            print(f"    SKIPPED: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad backend must not kill the sweep
            print(f"    FAILED:  {type(exc).__name__}: {exc}")
            continue
        finally:
            if writer["handle"] is not None:
                writer["handle"].release()

        contamination = analyze_contamination(frames)
        results.append((summary, contamination))
        print(
            f"    {summary.frames_processed} frames, {summary.unique_track_ids} ids, "
            f"x{summary.realtime_ratio:.2f} realtime"
        )

    if not results:
        print("\nNo backend produced results.")
        return 1

    # ---------------------------------------------------------------- table
    print()
    print("=" * 100)
    print("IDENTITY STABILITY AND SPEED")
    print("-" * 100)
    header = (
        f"{'backend':<26}{'ids':>6}{'mean life':>11}{'median':>9}"
        f"{'players/f':>11}{'sec':>9}{'xRT':>8}{'ball f':>9}"
    )
    print(header)
    print("-" * 100)
    for summary, _ in sorted(results, key=lambda r: r[0].unique_track_ids):
        ball_cell = str(summary.frames_with_ball) if basketball_model else "-"
        print(
            f"{Path(summary.tracker_name).name:<26}"
            f"{summary.unique_track_ids:>6}"
            f"{summary.mean_track_lifetime_frames:>11.1f}"
            f"{summary.median_track_lifetime_frames:>9.1f}"
            f"{summary.mean_players_per_frame:>11.2f}"
            f"{summary.elapsed_sec:>9.1f}"
            f"{summary.realtime_ratio:>8.2f}"
            f"{ball_cell:>9}"
        )
    print("-" * 100)
    print("  ids        unique track ids. The on-court player count is fixed, so a total")
    print("             far above it means the backend is churning identities. Lower is better.")
    print("  mean life  average frames a track survives. Higher is better.")
    print("  xRT        processing time / video duration. <= 1.00 is real time.")

    # -------------------------------------------------- contamination table
    print()
    print("=" * 100)
    print("NON-PLAYER ID CONTAMINATION AND RIM-AREA CHURN")
    print("(tests the restrict_tracker_to_players=False default)")
    print("-" * 100)
    if not basketball_model:
        print("  UNAVAILABLE with non-basketball weights: there are no rim or ball classes to")
        print("  contaminate anything. Re-run with the trained 4-class model to get this table.")
    else:
        print(
            f"{'backend':<26}{'rim ids':>9}{'rim tracked f':>15}"
            f"{'sw near rim':>13}{'sw total':>10}{'near-rim share':>16}"
        )
        print("-" * 100)
        for summary, contamination in sorted(results, key=lambda r: r[0].unique_track_ids):
            print(
                f"{Path(summary.tracker_name).name:<26}"
                f"{contamination.distinct_nonplayer_ids:>9}"
                f"{contamination.tracked_rim_frames:>15}"
                f"{contamination.id_switches_near_rim:>13}"
                f"{contamination.id_switches_total:>10}"
                f"{contamination.rim_switch_rate * 100:>15.1f}%"
            )
        print("-" * 100)
        print("  rim ids         distinct track ids the backend handed to rim detections.")
        print("                  The rim is one stationary object; more than 1 means churn.")
        print("  sw near rim     new player ids appearing within 2 rim-widths of the rim.")
        print("  near-rim share  share of all id changes that happen near the rim. A high")
        print("                  share means players passing the rim disturb association,")
        print("                  which argues for restrict_tracker_to_players=True.")
        print("  NOTE  'switches' counts ids appearing that were absent last frame, so it")
        print("        also counts genuine entries into frame. It is a cross-backend")
        print("        comparison on identical footage, not an absolute MOTA IDSW.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "comparison.json"
    with report.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "video": str(args.input),
                "weights": str(args.weights),
                "basketball_model": basketball_model,
                "video_duration_sec": info.duration_sec,
                "results": [
                    {
                        "tracker": Path(s.tracker_name).name,
                        "frames_processed": s.frames_processed,
                        "unique_track_ids": s.unique_track_ids,
                        "mean_track_lifetime_frames": s.mean_track_lifetime_frames,
                        "median_track_lifetime_frames": s.median_track_lifetime_frames,
                        "mean_players_per_frame": s.mean_players_per_frame,
                        "frames_with_ball": s.frames_with_ball if basketball_model else None,
                        "elapsed_sec": s.elapsed_sec,
                        "realtime_ratio": s.realtime_ratio,
                        "distinct_nonplayer_ids": c.distinct_nonplayer_ids if basketball_model else None,
                        "rim_tracked_frames": c.tracked_rim_frames if basketball_model else None,
                        "id_switches_near_rim": c.id_switches_near_rim if basketball_model else None,
                        "id_switches_total": c.id_switches_total,
                        "rim_switch_rate": c.rim_switch_rate if basketball_model else None,
                    }
                    for s, c in results
                ],
            },
            handle,
            indent=2,
        )
    print()
    print(f"  report  {report}")
    if args.save_videos:
        print(f"  videos  {args.output_dir}  (one .mp4 per backend for visual comparison)")
    print("=" * 100)
    return 0


def _annotate(frame, frame_tracks):
    """Minimal per-backend overlay: id-coloured player boxes plus ball and rim."""
    import cv2

    out = frame.copy()
    for player in frame_tracks.players:
        x1, y1, x2, y2 = (int(v) for v in player.bbox_xyxy)
        tid = player.track_id if player.track_id is not None else -1
        # Deterministic colour per id so an id switch is visible as a colour flip.
        color = ((tid * 67) % 255, (tid * 131) % 255, (tid * 197) % 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, f"id:{tid}", (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )
    for rim in frame_tracks.rims:
        x1, y1, x2, y2 = (int(v) for v in rim.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 255), 2)
    if frame_tracks.ball is not None:
        x1, y1, x2, y2 = (int(v) for v in frame_tracks.ball.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 2)
    cv2.putText(
        out, f"frame={frame_tracks.frame_idx}", (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return out


if __name__ == "__main__":
    sys.exit(main())
