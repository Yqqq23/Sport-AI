"""Online court-space tracking: constant-velocity Kalman + gated Hungarian.

Jersey numbers are deliberately absent from association. IDs never recycle;
small display slots can recycle only after the previous track has expired.
All positions/noise/speed settings are in metres and seconds.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MotionConfig:
    enabled: bool = True
    measurement_std_m: float = 0.25
    acceleration_std_mps2: float = 5.0
    max_speed_mps: float = 10.0
    position_slack_m: float = 0.6
    mahalanobis_gate: float = 9.21  # 99% chi-square, 2 position dimensions
    max_lost_sec: float = 1.0
    predict_sec: float = 0.25
    court_margin_m: float = 2.0


@dataclass
class MotionTrack:
    track_id: int
    slot: int
    state: np.ndarray
    covariance: np.ndarray
    last_observed: float
    last_position: np.ndarray
    hits: int = 1

    @property
    def xy(self) -> tuple[float, float]:
        return tuple(float(v) for v in self.state[:2])


class CourtMotionTracker:
    def __init__(self, config: MotionConfig, max_players: int = 6,
                 court_size: tuple[float, float] = (15.0, 14.0)):
        self.config = config
        self.max_players = max_players
        self.court_size = np.asarray(court_size)
        for name in ("measurement_std_m", "acceleration_std_mps2", "max_speed_mps",
                     "position_slack_m", "mahalanobis_gate", "max_lost_sec"):
            if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ValueError(f"motion.{name} must be finite and positive")
        if not 0 <= config.predict_sec <= config.max_lost_sec:
            raise ValueError("motion.predict_sec must be between zero and max_lost_sec")
        if not np.isfinite(config.court_margin_m) or config.court_margin_m < 0:
            raise ValueError("motion.court_margin_m must be finite and nonnegative")
        self.tracks: dict[int, MotionTrack] = {}
        self.expired_ids: list[int] = []
        self._next_id = 1
        self._time = None
        self.rejected_pairs = 0

    def update(self, positions, timestamp: float) -> list[int]:
        """Return motion ID per observation; 0 means invalid/no available slot."""
        z = np.asarray(positions, dtype=float).reshape(-1, 2)
        if not np.isfinite(timestamp) or (self._time is not None and timestamp <= self._time):
            raise ValueError("Motion timestamps must be finite and strictly increasing")
        dt = 0.0 if self._time is None else timestamp - self._time
        self._time = timestamp
        cfg = self.config
        self.expired_ids = [k for k, t in self.tracks.items()
                            if timestamp - t.last_observed > cfg.max_lost_sec]
        for key in self.expired_ids:
            del self.tracks[key]

        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        q = g @ g.T * cfg.acceleration_std_mps2 ** 2
        r = np.eye(2) * cfg.measurement_std_m ** 2
        tracks = list(self.tracks.values())
        for track in tracks:
            track.state = f @ track.state
            track.covariance = f @ track.covariance @ f.T + q

        valid = np.isfinite(z).all(axis=1) & (z >= -cfg.court_margin_m).all(axis=1)
        valid &= (z <= self.court_size + cfg.court_margin_m).all(axis=1)
        indices = np.flatnonzero(valid)
        # Private dummy columns let every track remain unmatched. Forbidden
        # edges must not participate and steal another track's feasible match.
        costs = np.full((len(tracks), len(indices) + len(tracks)), 1e9)
        for i, track in enumerate(tracks):
            costs[i, len(indices) + i] = cfg.mahalanobis_gate + 1
            s = track.covariance[:2, :2] + r
            age = timestamp - track.last_observed
            for j, index in enumerate(indices):
                delta = z[index] - track.state[:2]
                d2 = float(delta @ np.linalg.solve(s, delta))
                distance = np.linalg.norm(z[index] - track.last_position)
                if d2 <= cfg.mahalanobis_gate and distance <= cfg.max_speed_mps * age + cfg.position_slack_m:
                    costs[i, j] = d2
                else:
                    self.rejected_pairs += 1

        result = [0] * len(z)
        if tracks:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if j >= len(indices) or costs[i, j] >= 1e9:
                    continue
                index = indices[j]
                track = tracks[i]
                s = track.covariance[:2, :2] + r
                k = np.linalg.solve(s, track.covariance[:2, :]).T
                track.state += k @ (z[index] - track.state[:2])
                # Joseph form preserves a positive-semidefinite covariance.
                a = np.eye(4)
                a[:, :2] -= k
                track.covariance = a @ track.covariance @ a.T + k @ r @ k.T
                speed = np.linalg.norm(track.state[2:])
                if speed > cfg.max_speed_mps:
                    track.state[2:] *= cfg.max_speed_mps / speed
                track.last_observed = timestamp
                track.last_position = z[index].copy()
                track.hits += 1
                result[index] = track.track_id

        for index in indices:
            if result[index]:
                continue
            used = {t.slot for t in self.tracks.values()}
            limit = self.max_players or len(used) + 1
            slot = next((s for s in range(1, limit + 1) if s not in used), None)
            if slot is None:
                continue
            key = self._next_id
            self._next_id += 1
            self.tracks[key] = MotionTrack(
                key, slot, np.r_[z[index], [0.0, 0.0]],
                np.diag([cfg.measurement_std_m ** 2] * 2 + [16.0] * 2),
                timestamp, z[index].copy(),
            )
            result[index] = key
        return result

    def predictions(self) -> list[MotionTrack]:
        """Short, confirmed-track gap fill only; never prolonged extrapolation."""
        return [t for t in self.tracks.values() if t.hits >= 3
                and 0 < self._time - t.last_observed <= self.config.predict_sec]
