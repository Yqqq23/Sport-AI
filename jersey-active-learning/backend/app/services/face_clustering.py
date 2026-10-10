from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_face_model = None


def _get_face_model():
    global _face_model
    if _face_model is None:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(320, 320))
        _face_model = app
    return _face_model


def _estimate_head_bbox(player_bbox: list[int], image_width: int, image_height: int) -> list[int]:
    x1, y1, x2, y2 = player_bbox
    pw = x2 - x1
    ph = y2 - y1
    head = [
        x1 + int(pw * 0.2),
        y1 + int(ph * 0.02),
        x2 - int(pw * 0.2),
        y1 + int(ph * 0.35),
    ]
    head[0] = max(0, head[0])
    head[1] = max(0, head[1])
    head[2] = min(image_width, head[2])
    head[3] = min(image_height, head[3])
    return head


def extract_face_embedding(frame: np.ndarray, player_bbox: list[int]) -> np.ndarray | None:
    app = _get_face_model()
    head = _estimate_head_bbox(player_bbox, frame.shape[1], frame.shape[0])
    head_crop = frame[head[1]:head[3], head[0]:head[2]].copy()
    if head_crop.size == 0:
        return None
    faces = app.get(head_crop)
    if not faces:
        return None
    return faces[0].normed_embedding


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))


def cluster_embeddings(
    embeddings: dict[int, np.ndarray], threshold: float = 0.4
) -> dict[int, int]:
    track_ids = list(embeddings.keys())
    if not track_ids:
        return {}
    embs = np.stack([embeddings[tid] for tid in track_ids])
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-6)
    sim = embs @ embs.T
    assigned: dict[int, int] = {}
    next_cluster = 0
    for i, tid in enumerate(track_ids):
        for j in range(i):
            if sim[i, j] >= threshold:
                assigned[tid] = assigned[track_ids[j]]
                break
        if tid not in assigned:
            assigned[tid] = next_cluster
            next_cluster += 1
    return assigned
