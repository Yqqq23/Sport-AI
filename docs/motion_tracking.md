# Continuous top-down player tracks

## Per-player outputs

`output.write_player_topdowns: true` (default) additionally writes:

```
players/
  motion_000001.mp4
  motion_000001.png
  motion_000002.mp4
  motion_000002.png
  index.json
```

Each video shows only one motion ID on the exact same calibrated court template
as `topdown.mp4`, at the source FPS. It starts at the track's first valid canvas
position and continues until track retirement or processing ends. Missing frames
within this interval remain on the timeline, with short predictions marked `~`
and a blank court after prediction expires. The original frame/time is shown.
`index.json` records the source start/end frame, display slot, observed jersey
labels, and observed/predicted frame counts. The PNG is the accumulated observed
path for this track, with gaps broken rather than connected through occlusions.

File numbering is the immutable **motion ID**, not the OCR jersey number or
reusable roster slot. Long occlusions may create multiple motion clips for one
physical player; clips are deliberately not merged by jersey number. Use the
index to locate tracks labelled `7` (which may itself be an OCR error).

No extra inference or recalibration is performed. Videos are encoded during the
same run; one additional encoder per active motion track increases disk/encoding
cost. Disable with `output.write_player_topdowns: false`. The feature requires
the enabled metric motion tracker and works even with `write_topdown: false`.

## Why jersey recognition caused teleports

The legacy pipeline greedily matched image-box centres, then merged identities
by OCR number. Two players read as `1` could share a trail key, even if they
were on opposite ends of the court. EMA smoothing cannot fix this: it merely
draws a slower incorrect transition between two different people.

The default metric-calibrated pipeline now uses a SORT-style online motion
tracker in **court metres**:

1. Project each detected player's bottom-edge midpoint to the floor plane.
2. Predict `[x, y, vx, vy]` and uncertainty with a constant-velocity Kalman filter.
3. Reject associations outside both a Mahalanobis uncertainty gate and a
   maximum-displacement gate (`max_speed * elapsed + position_slack`).
4. Use Hungarian global one-to-one assignment, including unmatched options.
5. Correct accepted tracks with observations and render their filtered positions.

Motion IDs increase monotonically and never recycle. Display slots remain
bounded by `identity.max_players`; they are reusable only after track expiry.
OCR evidence, jersey labels, trails, and shot attribution use the motion ID.
Numbers are labels only: even two confirmed `#1` readings never merge tracks.
Only temporally locked numbers are displayed in motion mode. The default now
continues OCR after a lock so a mistaken initial label can eventually correct.

## Occlusion policy

Confirmed tracks (at least three matched observations) are predicted for at
most **0.25 s** while unobserved; their top-down label has a `~` suffix.
Predicted players are not added to annotated detections, OCR, possession, or
shot analysis. After this prediction window the marker disappears and its tail
is cleared; association can still recover its ID for up to **1 s**.
After expiry, a new observation starts a new ID and empty OCR history/tail.

The new tracker consumes the player boxes returned by the detector/MOT layer;
it cannot recover detections that never reach that layer. Complete overlapping
players can still exchange identities, and long occlusions deliberately break
tracks instead of claiming an unreliable connection. Appearance/ReID and team
colour can be added later, but incorrect OCR is not used as an association cue.

## Configuration and execution

`configs/default.yaml` enables `motion.enabled: true`. Defaults are starting
values, not measured accuracy guarantees:

| Setting | Default | Meaning |
|---|---:|---|
| `measurement_std_m` | 0.25 | Foot-point measurement uncertainty |
| `acceleration_std_mps2` | 5.0 | Allowance for acceleration/turning |
| `max_speed_mps` | 10.0 | Association and prediction speed cap |
| `position_slack_m` | 0.6 | Measurement-error allowance in displacement gate |
| `mahalanobis_gate` | 9.21 | 99% chi-square threshold for two dimensions |
| `predict_sec` | 0.25 | Maximum displayed extrapolation |
| `max_lost_sec` | 1.0 | Association retention before retiring identity |
| `court_margin_m` | 2.0 | Accepted out-of-court detection margin |

Requires SciPy (also normally installed by Ultralytics). Run the same pipeline
command; no additional CLI switches are necessary. The startup log confirms:

```
Court motion tracking: Kalman + gated Hungarian; OCR labels do not merge tracks
```

Without a metric calibration, the pipeline logs a warning and falls back to
legacy image-space tracking; metre-based thresholds must not be applied to
uncalibrated pixels. To disable the new path explicitly, set `motion.enabled`
to false. In the new path Kalman filtering replaces the additional render EMA.

Verification: `python -m pytest tests/test_motion.py tests/test_calibration.py -q`.
Synthetic regressions cover crossing players, reordered detections, brief
occlusion, expiry, outliers, measurement jitter, and `7 -> 1 -> 7` labels through
the actual pipeline render-data assembly.
