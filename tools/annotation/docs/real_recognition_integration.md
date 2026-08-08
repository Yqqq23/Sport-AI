# Real Recognition Integration

## Overview

`jersey-active-learning` now exposes real user-facing recognition endpoints:

- `POST /videos/{video_id}/run-jersey-recognition`
- `POST /videos/{video_id}/run-jersey-recognition-async`

The normal UI no longer uses mock recognition.

## Sport-CV Components Reused

Current best-effort integration uses:

- Sport-CV-compatible Ultralytics YOLO player detector weights
- optional Sport-CV OCR adapter from `Sport-CV/0619/identity.py`

Integrated pieces:

- real player detection from Ultralytics YOLO weights
- real tracking ids from `model.track(..., persist=True)` when available
- jersey region estimation from player bbox geometry
- optional OCR via Sport-CV `RoboflowSmolVLM2Recognizer`

## What Is Real

- `player_bbox`: real detector output
- `track_id`: real tracker id when Ultralytics tracking returns ids
- `confidence`: detector confidence, or OCR confidence when OCR returns a number

## What Is Best-Effort / Not Yet Dedicated

- `jersey_bbox` is currently estimated from the detected player bbox, not from a dedicated jersey detector
- OCR depends on Sport-CV OCR dependencies and configured credential/model access
- if OCR is unavailable, `predicted_number` is returned as an empty string instead of fake data

## Required Configuration

Use `.env` in `backend/` or environment variables:

```env
REAL_RECOGNITION_ENABLED=true
SPORT_CV_ROOT=D:\25-26香港教育大学\继续卷output。。。\Sport-CV
PLAYER_DETECTOR_WEIGHTS=D:\path\to\your\player_detector_weights.pt
JERSEY_OCR_WEIGHTS=<ocr-api-key-or-access-token>
DETECTOR_CONFIDENCE=0.2
DETECTOR_IOU=0.45
RECOGNITION_FRAME_STRIDE=5
```

Notes:

- `PLAYER_DETECTOR_WEIGHTS` is required for real recognition
- `JERSEY_OCR_WEIGHTS` is optional in the current implementation; if missing, OCR is skipped and empty jersey numbers are returned

## Error Behavior

If player detector weights are missing:

- backend returns:
  - `Real recognition model weights not found. Please configure MODEL_PATH.`
- frontend shows the error and does not fall back to mock data

If OCR is unavailable:

- recognition still runs if detector weights exist
- `predicted_number` becomes `""`
- confidence falls back to detector confidence
- warnings are attached to the recognition job result/message

## Output Mapping

The recognizer produces `JerseyPrediction` rows with:

- `video_id`
- `frame_row_id`
- `frame_id`
- `track_id`
- `player_bbox`
- `jersey_bbox`
- `predicted_number`
- `confidence`

## Current RA Workflow

1. Upload a basketball video
2. Select the video
3. Click `Run Jersey Recognition`
4. Click `Load Review Candidates`
5. Review low-confidence or temporally inconsistent candidates
6. Correct bbox / jersey number
7. Save annotation
8. Click `Download Corrected Dataset`

## Current Limitations

- no dedicated jersey-region detector yet
- OCR integration depends on external Sport-CV OCR dependencies/configuration
- no real jersey-number model is bundled inside `jersey-active-learning`
- extracted frames are sampled using configured stride during recognition
