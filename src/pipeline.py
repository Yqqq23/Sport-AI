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
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

# Allow both `python src/pipeline.py` and `python -m src.pipeline`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from calib import CourtProjector, court_canvas_matrix, load_court_calibration
from motion import CourtMotionTracker
from player_trajectories import PlayerTrajectoryExporter
from config import AppConfig, load_config
from identity import Identity, PlayerIdentityRegistry, SlotTracker
from jersey import (
    GATE_OK,
    GATE_TOO_BLURRY,
    GATE_TOO_SMALL,
    JerseyNumberStabilizer,
    crop_jersey,
    gate_reason,
    make_ocr_backend,
    torso_region_from_player_box,
)


@dataclass(frozen=True)
class _TorsoBox:
    """A synthetic jersey box derived from a player box.

    Duck-types the parts of ``track.TrackedObject`` that the jersey path uses
    (``bbox_xyxy`` and ``center_xy``), so the torso fallback flows through the
    exact same ownership / gate / crop / OCR code as a real detector box. A
    separate code path would be a second place for the crop logic to drift.
    """

    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
from render import (
    BallRender,
    CourtRenderConfig,
    FrameRenderData,
    PlayerRender,
    RefereeRender,
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


def _check_output_directory(output_dir: Path) -> None:
    """Fail before inference when the output directory is not writable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / f".write_probe_{os.getpid()}"
    try:
        probe.write_bytes(b"ok")
    except OSError as exc:
        raise RuntimeError(
            f"Output directory is not writable: {output_dir} ({exc}). "
            "Check `df -h`, `quota -s`, and directory permissions."
        ) from exc
    finally:
        probe.unlink(missing_ok=True)


def _open_video_writer(path: Path, fourcc: int, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Cannot create output video: {path}. The '{path.suffix}' writer or "
            "configured codec is unavailable, or the output directory is not writable."
        )
    return writer


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
        "Calibration loaded (%s reference points): %s",
        len(calibration.point_names),
        config.calibration_json,
    )
    return projector


def run_pipeline(
    input_video: Path,
    config: AppConfig,
    tracker_override: Optional[str] = None,
    export_trajectories: bool = False,
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
    _check_output_directory(output_dir)

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
    if projector is not None:
        try:
            projector.check_frame(width, height)
        except ValueError:
            capture.release()
            raise
    tracker = Tracker(weights=config.detector.weights, config=config.tracker)
    stabilizer = JerseyNumberStabilizer(config.jersey.stabilizer)
    identities = PlayerIdentityRegistry(max_players=config.identity.max_players)
    # Court motion IDs own OCR and trajectories. Image-space slots are retained
    # only as a fallback when metric calibration is unavailable or disabled.
    slot_tracker = SlotTracker(max_players=config.identity.max_players)
    motion_tracker = None
    metric_to_canvas = court_canvas_matrix(config.canvas)
    if config.motion.enabled:
        if projector is not None and projector.calibration.homography is not None:
            motion_tracker = CourtMotionTracker(
                config.motion, config.identity.max_players,
                (config.canvas.court_width_m, config.canvas.court_height_m),
            )
            LOGGER.info("Court motion tracking: Kalman + gated Hungarian; OCR labels do not merge tracks")
        else:
            LOGGER.warning("Metric calibration unavailable; using legacy image-space slot tracking")
    ocr = make_ocr_backend(
        config.jersey.backend,
        use_gpu=config.jersey.use_gpu,
        roboflow=config.jersey.roboflow,
        local=config.jersey.local,
        parseq=config.jersey.parseq,
        perdigit=config.jersey.perdigit,
    )
    shot_analyzer = ShotAnalyzer(config.shot, fps=fps)
    recorder = ShotRecorder()

    if config.jersey.enabled and not ocr.is_available():
        LOGGER.warning(
            "Jersey OCR is enabled but no OCR backend could be loaded "
            "(configured: %s). Numbers will stay unknown and every output will "
            "fall back to track ids. Fix:  pip install paddleocr   (or easyocr), "
            "or set jersey.enabled=false to silence this.",
            config.jersey.backend,
        )

    debug_crop_dir: Optional[Path] = None
    if config.jersey.debug_crop_dir is not None:
        debug_crop_dir = Path(config.jersey.debug_crop_dir)
        debug_crop_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Dumping jersey crops to %s", debug_crop_dir)

    court_config: CourtRenderConfig = config.render.court
    court_template = draw_half_court_template(court_config)
    trails = TrailHistory(
        trail_length=court_config.trail_length,
        alpha=1.0 if motion_tracker is not None else config.render.trail_alpha,
    )
    ball_trail = make_ball_trail(court_config.trail_length)

    fourcc = cv2.VideoWriter_fourcc(*config.output.fourcc)
    annotated_path = output_dir / config.output.annotated_video_name
    topdown_path = output_dir / config.output.topdown_video_name
    writer_annotated = (
        _open_video_writer(annotated_path, fourcc, fps, (width, height))
        if config.output.write_annotated
        else None
    )
    writer_topdown = (
        _open_video_writer(
            topdown_path, fourcc, fps, (court_config.width, court_config.height)
        )
        if config.output.write_topdown
        else None
    )

    player_exporter = None
    if config.output.write_player_topdowns:
        if motion_tracker is not None:
            player_exporter = PlayerTrajectoryExporter(
                output_dir / "players", court_template, court_config, fps, fourcc, _open_video_writer
            )
        else:
            LOGGER.warning("Per-player trajectory videos require enabled metric motion tracking")

    frame_idx = 0
    ocr_calls = 0
    ocr_hits = 0
    #: Jersey boxes the DETECTOR produced. Zero across a whole run means the
    #: weights have no jersey class, which silently disabled OCR entirely.
    jersey_boxes_seen = 0
    #: Frames that fell back to a geometric torso region instead.
    torso_fallback_frames = 0
    #: Player boxes that arrived with no tracker id.
    untracked_players = 0
    # Skips are counted BY REASON. A single combined counter (the previous
    # "skipped by the size/blur gate") cannot tell you which threshold starved
    # the run, which is exactly the question you have when no number locks.
    skips = {
        GATE_TOO_SMALL: 0,
        GATE_TOO_BLURRY: 0,
        "no_owner": 0,
        "bad_crop": 0,
        "already_locked": 0,
        "call_budget": 0,
    }
    started = time.perf_counter()

    trajectory_file = None
    if export_trajectories:
        if not cv2.imwrite(str(output_dir / "court_template.png"), court_template):
            raise RuntimeError("Cannot save court template")
        trajectory_file = (output_dir / "trajectories.jsonl").open("w", encoding="utf-8")
        (output_dir / "trajectory_meta.json").write_text(json.dumps({
            "fps": fps, "width": width, "height": height,
            "calibrated": projector is not None,
            "annotated": annotated_path.name, "topdown": topdown_path.name,
        }), encoding="utf-8")

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
            jersey_boxes_seen += len(tracked.jerseys)
            untracked_players += sum(1 for p in tracked.players if p.track_id is None)

            # The mapping holds monotonic motion IDs in metric mode, or legacy
            # slot IDs otherwise. Detector/MOT IDs never override this mapping.
            if motion_tracker is not None:
                feet = np.asarray([p.ground_xy for p in tracked.players], dtype=float).reshape(-1, 2)
                if len(feet):
                    homogeneous = np.c_[feet, np.ones(len(feet))] @ projector.calibration.homography.T
                    with np.errstate(divide="ignore", invalid="ignore"):
                        positions = homogeneous[:, :2] / homogeneous[:, 2:3]
                else:
                    positions = np.empty((0, 2))
                player_slots = motion_tracker.update(positions, timestamp_sec)
                for expired in motion_tracker.expired_ids:
                    stabilizer.states.pop(expired, None)
                    identities.forget_motion(expired)
                    trails.discard(f"motion:{expired}")
            else:
                player_slots = slot_tracker.assign(
                    [p.bbox_xyxy for p in tracked.players], frame_idx
                )
            slot_of_player = {id(p): s for p, s in zip(tracked.players, player_slots)}

            if run_ocr_this_frame:
                # No jersey boxes from the detector: fall back to a geometric
                # torso region per player. Weights trained without a `jersey`
                # class otherwise make this whole loop a no-op - zero crops,
                # zero calls, zero skips - and the summary shows nothing at all
                # to explain why no numbers appeared.
                jersey_sources = tracked.jerseys
                if not jersey_sources and config.jersey.fallback_to_torso:
                    jersey_sources = [
                        _TorsoBox(
                            bbox_xyxy=tuple(
                                torso_region_from_player_box(
                                    player.bbox_xyxy,
                                    config.jersey.torso_top,
                                    config.jersey.torso_bottom,
                                    config.jersey.torso_width,
                                )
                            )
                        )
                        for player in tracked.players
                    ]
                    torso_fallback_frames += 1

                for jersey_box in jersey_sources:
                    # Jersey ownership is resolved against tracked.players ONLY.
                    # Referees (CLASS_REFEREE) are already absent from that list
                    # by the role split in src/track.py, so a jersey box drawn on
                    # a referee can never acquire an owner and never votes. That
                    # is intended: referee numbers are not player numbers, and
                    # letting one bind would corrupt a real player's vote pool.
                    owner = _nearest_player(
                        jersey_box,
                        tracked.players,
                        max_distance_ratio=config.jersey.owner_max_distance_ratio,
                    )
                    if owner is None:
                        skips["no_owner"] += 1
                        continue
                    # Keyed on the SLOT, not owner.track_id. The old check
                    # rejected every crop whenever the backend produced no ids,
                    # which is why OCR never ran a single time.
                    owner_slot = slot_of_player.get(id(owner), 0)
                    if owner_slot == 0:
                        skips["no_owner"] += 1
                        continue

                    # Once a number has LOCKED, further OCR on that track can
                    # only confirm what is already known. Checked before the
                    # crop is even cut, because with a network backend every
                    # avoided call is seconds of wall clock.
                    if config.jersey.skip_locked_tracks and stabilizer.is_locked(
                        owner_slot
                    ):
                        skips["already_locked"] += 1
                        continue

                    if 0 < config.jersey.max_ocr_calls <= ocr_calls:
                        skips["call_budget"] += 1
                        continue

                    # Padded + upscaled, NOT the raw detector box. See
                    # jersey.crop_jersey for why a tight crop is the wrong input.
                    crop = crop_jersey(frame, jersey_box.bbox_xyxy, config.jersey.gate)
                    if crop is None:
                        skips["bad_crop"] += 1
                        continue

                    reason = gate_reason(
                        list(jersey_box.bbox_xyxy), config.jersey.gate, crop
                    )
                    if reason != GATE_OK:
                        skips[reason] += 1
                        continue

                    text, confidence = ocr.recognize(crop)
                    ocr_calls += 1
                    if text is not None:
                        ocr_hits += 1
                    if debug_crop_dir is not None:
                        _dump_crop(
                            debug_crop_dir, crop, frame_idx, owner_slot, text, confidence
                        )
                    stabilizer.update(
                        track_id=owner_slot,
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
                make_player_box(slot_of_player.get(id(p)) or None, p.bbox_xyxy, p.ground_xy)
                for p in tracked.players
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
                    # Resolve through the identity registry, not the stabilizer
                    # directly: the registry is sticky, so a shot taken on a
                    # frame where OCR happened not to run is still credited to
                    # the number that track is already known to wear.
                    if motion_tracker is not None:
                        event.shooter_jersey_number = (
                            stabilizer.get_stable_number(event.shooter_track_id)
                            if stabilizer.is_locked(event.shooter_track_id) else None
                        )
                    else:
                        event.shooter_jersey_number = identities.resolve(
                            event.shooter_track_id,
                            stabilizer.get_stable_number(event.shooter_track_id),
                            frame_idx=frame_idx,
                        ).number
                if projector is not None and event.shot_position_xy is not None:
                    event.court_xy = projector.project_point(*event.shot_position_xy)
                recorder.add_event(event)

            # --- rendering -------------------------------------------------
            render_data = _build_render_data(
                tracked=tracked,
                stabilizer=stabilizer,
                identities=identities,
                projector=projector,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                slot_of_player=slot_of_player,
                motion_tracker=motion_tracker,
            )
            if trajectory_file is not None:
                trajectory_file.write(json.dumps({
                    "frame": frame_idx,
                    "players": [{
                        "key": str(p.render_key), "number": p.jersey_number,
                        "bbox": [float(v) for v in p.bbox_xyxy],
                        "court": [float(v) for v in p.court_xy]
                        if p.court_xy is not None and np.isfinite(p.court_xy).all() else None,
                    } for p in render_data.players],
                }, allow_nan=False) + "\n")
            if writer_annotated is not None:
                writer_annotated.write(
                    render_annotated_frame(
                        frame, render_data, line_width=config.render.line_width
                    )
                )
            if writer_topdown is not None or player_exporter is not None:
                topdown_data = render_data
                if motion_tracker is not None:
                    predictions = []
                    for state in motion_tracker.predictions():
                        number = (stabilizer.get_stable_number(state.track_id)
                                  if stabilizer.is_locked(state.track_id) else None)
                        identity = identities.resolve_motion(state.track_id, state.slot, number, frame_idx)
                        point = metric_to_canvas @ [*state.xy, 1]
                        predictions.append(PlayerRender(
                            track_id=state.track_id, bbox_xyxy=[], court_xy=tuple(point[:2]),
                            identity=identity, predicted=True,
                        ))
                    topdown_data = replace(render_data, players=render_data.players + predictions)
                    # Stop connecting old tails after the short prediction window.
                    for state in motion_tracker.tracks.values():
                        if timestamp_sec - state.last_observed > config.motion.predict_sec:
                            trails.discard(f"motion:{state.track_id}")
                if player_exporter is not None:
                    player_exporter.update(topdown_data, {f"motion:{k}" for k in motion_tracker.tracks})
                if writer_topdown is not None:
                    writer_topdown.write(
                        render_topdown_frame(
                            court_template=court_template,
                            data=topdown_data,
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
        if trajectory_file is not None:
            trajectory_file.close()
        if writer_annotated is not None:
            writer_annotated.release()
        if writer_topdown is not None:
            writer_topdown.release()
        if player_exporter is not None:
            player_exporter.close()

    elapsed = time.perf_counter() - started
    video_seconds = frame_idx / max(fps, 1e-6)

    try:
        csv_path, json_path = recorder.write(
            output_dir=output_dir,
            stem=config.output.shots_stem,
            meta={
                "input_video": input_video.name,
                "frames_processed": frame_idx,
                "fps": round(fps, 3),
                "tracker": tracker.backend_name,
                "reid_active": tracker.reid_active,
                "motion_tracking": "kalman_hungarian" if motion_tracker is not None else "legacy_slots",
                "weights": Path(config.detector.weights).name,
                "elapsed_sec": round(elapsed, 2),
                "realtime_ratio": round(elapsed / video_seconds, 3) if video_seconds > 0 else None,
                "ocr_backend": type(ocr).__name__,
                "ocr_calls": ocr_calls,
                "ocr_hits": ocr_hits,
                "ocr_skipped": dict(skips),
                "jersey_numbers": identities.confirmed_numbers(),
            },
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write results under {output_dir}: {exc}. "
            "The video writer warnings immediately before this usually mean the "
            "filesystem is full or your account quota was exceeded. Check `df -h`, "
            "`quota -s`, and `du -sh runs outputs`."
        ) from exc

    print(recorder.format_summary())
    print(f"  tracker            {tracker.backend_name} (reid={tracker.reid_active})")
    print(f"  motion tracking    {'kalman_hungarian (OCR-independent)' if motion_tracker is not None else 'legacy_slots'}")
    print(f"  frames processed   {frame_idx}")
    print(
        f"  elapsed            {elapsed:.1f}s for {video_seconds:.1f}s of video "
        f"(x{elapsed / video_seconds:.2f} realtime)"
        if video_seconds > 0
        else f"  elapsed            {elapsed:.1f}s"
    )
    # --- per-class detection counts -----------------------------------
    # Printed unconditionally. A class showing 0 here is the difference between
    # "the detector never proposed one" and "something downstream dropped it",
    # and previously nothing in the output distinguished those.
    from detect import BASKETBALL_CLASS_NAMES

    counts = getattr(tracker, "class_detection_counts", {})
    summary = "  ".join(
        f"{name}={counts.get(class_id, 0)}"
        for class_id, name in BASKETBALL_CLASS_NAMES.items()
    )
    print(f"  detections         {summary}")
    if counts.get(1, 0) == 0:  # CLASS_BALL
        print(
            "  BALL NOT DETECTED  The detector proposed zero ball boxes. Check the\n"
            "                     'Detector classes' line in the log: if the class\n"
            "                     names are not player/ball/rim/jersey/referee at\n"
            "                     indices 0-4, the weights use a different scheme and\n"
            "                     every class is misfiled. Otherwise lower\n"
            "                     tracker.ball_conf (currently "
            f"{config.tracker.ball_conf})."
        )

    if identities.recycled_slots or identities.overflow_tracks:
        print(
            f"  roster             max={config.identity.max_players}"
            f"  recycled={identities.recycled_slots}"
            f"  overflow={identities.overflow_tracks}"
        )

    if untracked_players:
        print(
            f"  UNTRACKED PLAYERS  {untracked_players} player box(es) arrived with no\n"
            "                     tracker id and were drawn as grey 'id:?'. The MOT\n"
            "                     backend is not assigning ids - check that\n"
            "                     tracker.persist is true and that the tracker\n"
            "                     backend loaded."
        )

    print(f"  jersey boxes       {jersey_boxes_seen} from the detector")
    if jersey_boxes_seen == 0:
        print(
            "  NO JERSEY CLASS    The detector produced zero jersey boxes across the\n"
            "                     whole run, so these weights almost certainly have no\n"
            "                     'jersey' class. This is ALSO why referees are not\n"
            "                     separated: no referee class either.\n"
            "                     The recognizer (PARSeq etc.) was never the problem -\n"
            "                     it is only ever handed crops the detector locates,\n"
            "                     so with no boxes it is simply never called.\n"
            f"                     Torso fallback used on {torso_fallback_frames} frame(s)."
        )

    hit_rate = (100.0 * ocr_hits / ocr_calls) if ocr_calls else 0.0
    total_skipped = sum(skips.values())
    print(f"  jersey backend     {type(ocr).__name__}")
    print(f"  jersey OCR         {ocr_calls} run, {ocr_hits} returned a number ({hit_rate:.0f}%)")
    print(
        f"  jersey skipped     {total_skipped}"
        f"  (too_small={skips[GATE_TOO_SMALL]}"
        f" too_blurry={skips[GATE_TOO_BLURRY]}"
        f" no_owner={skips['no_owner']}"
        f" bad_crop={skips['bad_crop']}"
        f" locked={skips['already_locked']}"
        f" budget={skips['call_budget']})"
    )
    numbers = identities.confirmed_numbers()
    print(
        f"  jersey numbers     {', '.join('#' + n for n in numbers)}"
        if numbers
        else "  jersey numbers     none locked"
    )
    # Say WHICH stage failed rather than leaving the reader to infer it from the
    # counters. These three cases have three different fixes and are easy to
    # confuse from the raw numbers alone.
    if not numbers:
        if ocr_calls == 0 and total_skipped > 0:
            print(
                "  DIAGNOSIS          every candidate was gated out. Lower "
                "jersey.gate.min_box_width / min_box_height and re-run."
            )
        elif ocr_calls > 0 and ocr_hits == 0:
            print(
                "  DIAGNOSIS          OCR ran but read nothing. Set "
                "jersey.debug_crop_dir and look at the crops: if they are "
                "legible the backend is too weak (use paddleocr), if not the "
                "numbers are below the resolution floor."
            )
        elif ocr_hits > 0:
            print(
                "  DIAGNOSIS          numbers were read but never reached the "
                "vote quorum. Lower jersey.stabilizer.slot_lock_min_votes or "
                "jersey.ocr_interval."
            )
    if projector is None:
        print("  NOTE               no calibration, so court_x/court_y are empty")
    print()

    outputs = {"shots_csv": csv_path, "shots_json": json_path}
    if player_exporter is not None:
        outputs["player_trajectories"] = player_exporter.directory
    if writer_annotated is not None:
        outputs["annotated"] = annotated_path
    if writer_topdown is not None:
        outputs["topdown"] = topdown_path
    for name, path in outputs.items():
        print(f"  {name:<18} {path}")
    return outputs


def _nearest_player(jersey_box, players, max_distance_ratio: float = 0.25):
    """Attach a jersey box to the player box that contains or is nearest its centre.

    ``players`` never contains referees: src/track.py drops CLASS_REFEREE during
    the role split. Jersey numbers therefore bind to players only, by
    construction rather than by a check here.

    The tolerance is NEW. This used to require ``distance == 0.0`` - the jersey
    centre strictly inside the player box - which silently discarded a jersey
    box sitting a few pixels proud of a tightly-drawn player box. Two
    independently-regressed boxes on a ~150 px player disagree at their edges
    routinely, so exact containment threw away usable observations and did not
    even count them as skips. The allowance scales with the player's box height
    so it means the same thing near and far from the camera.
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
    x1, y1, x2, y2 = best.bbox_xyxy
    allowance = max(0.0, float(max_distance_ratio)) * abs(y2 - y1)
    return best if distance(best) <= allowance else None


def _dump_crop(directory: Path, crop, frame_idx: int, track_id: int, text, confidence) -> None:
    """Write one OCR input to disk, with the verdict encoded in the filename.

    Sorting the directory by name groups every crop a track produced, so a
    track that never locks a number can be inspected as a filmstrip. The
    filename carries what OCR returned so "unreadable crop" and "readable crop,
    wrong answer" are distinguishable without opening anything else.
    """
    verdict = f"{text}_{confidence:.2f}" if text else "none"
    path = directory / f"f{frame_idx:06d}_t{track_id:03d}_{verdict}.png"
    try:
        cv2.imwrite(str(path), crop)
    except Exception as exc:  # never let a debug aid kill a run
        LOGGER.debug("Could not write debug crop %s: %s", path, exc)


def _build_render_data(
    tracked, stabilizer, identities, projector, frame_idx, timestamp_sec, slot_of_player,
    motion_tracker=None,
) -> FrameRenderData:
    players: list[PlayerRender] = []
    # NO DUPLICATE SLOT IN ONE FRAME. SlotTracker already assigns each box a
    # unique slot, but the identity registry MERGES on jersey number - so if two
    # different players are both misread as "#7" they collapse to one identity
    # and two markers render with the same id. Whoever claims the identity first
    # keeps it; the other falls back to its own slot for this frame.
    claimed_identities: set[str] = set()
    claimed_slots: set[int] = set()
    for player in tracked.players:
        court_xy = None
        if projector is not None:
            court_xy = projector.project_point(*player.ground_xy)
        # Everything keys on the position-assigned slot; track_id is no longer
        # trusted to exist at all.
        slot = slot_of_player.get(id(player), 0)
        number = stabilizer.get_stable_number(slot) if slot else None
        # The identity, not the tracker id, is what gets drawn and what the
        # top-down trail is keyed on. Resolving it here (once per player per
        # frame) is what makes the label read "#7" instead of "id:23".
        # frame_idx is REQUIRED for the roster cap: slot recycling picks the
        # least-recently-seen identity, so without an advancing frame counter
        # every slot looks equally fresh and nothing is ever recycled.
        if motion_tracker is not None:
            state = motion_tracker.tracks.get(slot)
            if state is None:
                # Keep the original detection visible, but no unassociated dot.
                court_xy = None
                identity = Identity(key=f"unassigned:{id(player)}", slot=0)
            else:
                number = number if stabilizer.is_locked(slot) else None
                identity = identities.resolve_motion(slot, state.slot, number, frame_idx)
                point = court_canvas_matrix(projector.canvas) @ [*state.xy, 1]
                court_xy = tuple(point[:2])
        else:
            identity = identities.resolve(slot if slot else None, number, frame_idx=frame_idx)
        if identity.key in claimed_identities or (
            identity.slot > 0 and identity.slot in claimed_slots
        ):
            # Already used by another player this frame: show this one under its
            # own slot with no number, rather than printing the same id twice.
            identity = Identity(key=f"dup:{slot}", number=None, slot=slot)
        claimed_identities.add(identity.key)
        if identity.slot > 0:
            claimed_slots.add(identity.slot)
        players.append(
            PlayerRender(
                track_id=slot,
                bbox_xyxy=list(player.bbox_xyxy),
                jersey_number=identity.number,
                conf=player.confidence,
                court_xy=court_xy,
                identity=identity,
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
        # Referees reach the annotated view and stop there. They are not
        # projected to the court, not tracked, and not written to output 3 -
        # note this list is built from tracked.referees, never merged into
        # `players` above.
        referees=[
            RefereeRender(bbox_xyxy=list(r.bbox_xyxy), conf=r.confidence)
            for r in tracked.referees
        ],
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
    parser.add_argument("--export-trajectories", action="store_true", help="Save per-frame tracks for the web application")
    parser.add_argument(
        "--jersey-backend",
        default=None,
        choices=[
            "auto", "perdigit", "parseq", "local", "roboflow",
            "paddleocr", "easyocr", "pytesseract",
        ],
        help="Override jersey.backend for this run",
    )
    parser.add_argument(
        "--jersey-weights",
        type=Path,
        default=None,
        help="Override jersey.local.weights (checkpoint from train/train_jersey.py)",
    )
    parser.add_argument(
        "--roboflow-model",
        default=None,
        help=(
            "Override jersey.roboflow.model_id, e.g. digits-x7d7z/10. "
            "Format is project-slug/version; the workspace is NOT part of it."
        ),
    )
    parser.add_argument(
        "--roboflow-url",
        default=None,
        help=(
            "Override jersey.roboflow.api_url. Use http://localhost:9001 to hit a "
            "local inference server instead of the slow hosted endpoint."
        ),
    )
    parser.add_argument(
        "--dump-jersey-crops",
        type=Path,
        default=None,
        help=(
            "Write every crop handed to OCR into this directory, named with what "
            "OCR returned. Run this FIRST when numbers are not appearing: it is "
            "the only way to tell an unreadable crop from a failing recognizer."
        ),
    )
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
    if args.jersey_backend is not None:
        config.jersey.backend = args.jersey_backend
    if args.jersey_weights is not None:
        config.jersey.local.weights = args.jersey_weights
    if args.roboflow_model is not None:
        config.jersey.roboflow.model_id = args.roboflow_model
    if args.roboflow_url is not None:
        config.jersey.roboflow.api_url = args.roboflow_url
    if args.dump_jersey_crops is not None:
        config.jersey.debug_crop_dir = args.dump_jersey_crops

    run_pipeline(input_video=args.input, config=config, tracker_override=args.tracker,
                 export_trajectories=args.export_trajectories)


if __name__ == "__main__":
    main()
