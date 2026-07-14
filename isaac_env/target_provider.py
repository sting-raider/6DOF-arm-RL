"""Hardware-independent camera target estimate validation and smoothing.

The controller only needs a trustworthy object position. A detector, simulator,
recorded video, or network service can all produce :class:`TargetEstimate`
instances without changing the reach policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


Position3 = tuple[float, float, float]


@dataclass(frozen=True)
class TargetEstimate:
    """One detector result expressed in the robot/table coordinate frame."""

    position_xyz: Position3
    timestamp_s: float
    confidence: float


@dataclass(frozen=True)
class TrackedTarget:
    """Target accepted for control, or an explicit invalid result."""

    position_xyz: Position3 | None
    valid: bool
    reason: str
    source_timestamp_s: float | None


class TargetTracker:
    """Validate, smooth, and briefly hold camera target estimates.

    Invalid detector frames never overwrite the last accepted target. The last
    target may be held only until ``max_age_s``; after that the caller must stop
    target-directed motion instead of steering toward stale coordinates.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.60,
        max_age_s: float = 0.35,
        max_jump_m: float = 0.12,
        smoothing_alpha: float = 0.45,
        workspace_bounds: tuple[Position3, Position3] = (
            (0.10, -0.35, 0.75),
            (0.70, 0.35, 1.10),
        ),
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if max_age_s <= 0.0 or max_jump_m <= 0.0:
            raise ValueError("max_age_s and max_jump_m must be positive")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be within (0, 1]")
        lower, upper = workspace_bounds
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("workspace lower bounds must be below upper bounds")

        self.min_confidence = min_confidence
        self.max_age_s = max_age_s
        self.max_jump_m = max_jump_m
        self.smoothing_alpha = smoothing_alpha
        self.workspace_bounds = workspace_bounds
        self._position: Position3 | None = None
        self._source_timestamp_s: float | None = None

    def reset(self) -> None:
        self._position = None
        self._source_timestamp_s = None

    def current(self, now_s: float) -> TrackedTarget:
        """Return the last valid target only while it remains fresh."""
        if self._position is None or self._source_timestamp_s is None:
            return TrackedTarget(None, False, "no_target", None)
        if now_s - self._source_timestamp_s > self.max_age_s:
            return TrackedTarget(
                None, False, "stale_target", self._source_timestamp_s
            )
        return TrackedTarget(
            self._position, True, "held_target", self._source_timestamp_s
        )

    def update(
        self, estimate: TargetEstimate | None, *, now_s: float
    ) -> TrackedTarget:
        """Accept one estimate or safely fall back to a still-fresh target."""
        rejection = self._rejection_reason(estimate, now_s)
        if rejection is not None:
            held = self.current(now_s)
            if held.valid:
                return TrackedTarget(
                    held.position_xyz,
                    True,
                    f"held_after_{rejection}",
                    held.source_timestamp_s,
                )
            return TrackedTarget(
                None,
                False,
                rejection,
                None if estimate is None else estimate.timestamp_s,
            )

        assert estimate is not None
        measured = tuple(float(value) for value in estimate.position_xyz)
        if self._position is None:
            filtered = measured
        else:
            alpha = self.smoothing_alpha
            filtered = tuple(
                alpha * new + (1.0 - alpha) * old
                for new, old in zip(measured, self._position)
            )
        self._position = filtered  # type: ignore[assignment]
        self._source_timestamp_s = float(estimate.timestamp_s)
        return TrackedTarget(
            self._position, True, "accepted", self._source_timestamp_s
        )

    def _rejection_reason(
        self, estimate: TargetEstimate | None, now_s: float
    ) -> str | None:
        if estimate is None:
            return "missing_detection"
        values: Iterable[float] = (
            *estimate.position_xyz,
            estimate.timestamp_s,
            estimate.confidence,
            now_s,
        )
        if not all(math.isfinite(value) for value in values):
            return "nonfinite_detection"
        age_s = now_s - estimate.timestamp_s
        if age_s < -0.05:
            return "future_timestamp"
        if age_s > self.max_age_s:
            return "stale_detection"
        if estimate.confidence < self.min_confidence:
            return "low_confidence"
        lower, upper = self.workspace_bounds
        if any(
            value < lo or value > hi
            for value, lo, hi in zip(estimate.position_xyz, lower, upper)
        ):
            return "outside_workspace"
        if self._position is not None:
            jump = math.dist(estimate.position_xyz, self._position)
            if jump > self.max_jump_m:
                return "implausible_jump"
        return None
