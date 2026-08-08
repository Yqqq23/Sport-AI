# Dataset Versioning Design

## Purpose

Step 13 adds reproducible, versioned multi-video dataset construction for jersey bbox detection.

It solves four problems:

1. merge annotations from multiple videos
2. avoid train/val/test leakage in video data
3. make dataset builds deterministic and reproducible
4. preserve full provenance back to database annotations

## Architecture

- source of truth: active, non-deleted `ManualAnnotation` rows in SQLite
- build service: `backend/app/services/dataset_versioning.py`
- export target: `data/datasets/versions/{version_name}/`
- metadata record: `DatasetVersion` table

## Data Flow

1. validate request ratios and source videos
2. load exportable annotations from the database
3. compute `source_data_fingerprint`
4. build canonical frame/image records keyed by `(video_id, frame_id)`
5. build split groups according to strategy
6. assign groups to train/val/test deterministically
7. export images, YOLO labels, `dataset.yaml`, `classes.txt`
8. write `manifest.json`, `metadata.json`, `split_report.json`
9. validate leakage and filesystem integrity
10. mark `DatasetVersion` as `completed` or `failed`

## Grouping Strategies

### Video Split

- group key: `video_id`
- all frames from one video stay in exactly one split
- safest default for avoiding temporal leakage

### Track Split

- conceptual grouping starts from `(video_id, track_id)`
- frame integrity has higher priority than track-only independence
- if multiple tracks share a frame, a connected component is formed so the frame and connected tracks stay together
- conflict resolutions are recorded in `split_report.json`

### Temporal Block Split

- per video, frames are sorted by `frame_id`
- contiguous frame blocks are created using `temporal_block_size`
- block key: `(video_id, block_index)`

## Split Assignment Algorithm

- compute group sizes by image count
- deterministically shuffle groups using `random_seed`
- stably sort by descending group size
- greedily assign each group to the split with the largest current image-count deficit

This keeps the result deterministic while still reacting to `random_seed` when groups have similar sizes.

## Leakage Prevention

Universal rules:

- one canonical image can appear in only one split
- one exported image filename can appear in only one split
- one source annotation can appear in only one split
- one source prediction can appear in only one split

Strategy-specific rules:

- `video`: one video cannot appear in multiple splits
- `track`: one resolved track component cannot appear in multiple splits
- `temporal_block`: one block cannot appear in multiple splits

## Fingerprint Algorithm

`source_data_fingerprint` is SHA-256 over deterministically serialized exportable source rows containing:

- `video_id`
- `frame_id`
- `annotation_id`
- `prediction_id`
- `corrected_bbox`
- `corrected_number`
- `label_status`
- `deleted_at`

If the fingerprint changes, rebuild is refused.

## Reproducibility Guarantees

Given the same:

- source database state
- selected videos
- split strategy
- ratios
- random seed
- temporal block size

the same dataset version build will produce the same split assignment.

## Manifest Provenance

`manifest.json` preserves traceability from every exported sample back to:

- `ManualAnnotation`
- `JerseyPrediction`
- `VideoFrame`
- `VideoAsset`

## Limitations

- no train/val/test balancing by jersey number or team yet
- no multi-version merge tool yet
- no combined multi-class player+jersey export yet
- no automatic model training
- track split uses connected-component conflict resolution when frames link multiple tracks
