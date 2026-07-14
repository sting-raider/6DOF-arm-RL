"""Hardware-independent aggregation helpers for evaluation diagnostics."""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


GRASP_STAGE_LABELS = ("approach", "descend", "close", "retract", "recover")


def grasp_stage_label(stage: int) -> str:
    """Return a stable label without letting diagnostics crash an evaluation."""
    if 0 <= stage < len(GRASP_STAGE_LABELS):
        return GRASP_STAGE_LABELS[stage]
    return f"unknown({stage})"


@dataclass(frozen=True)
class EpisodeEndRow:
    """Aggregated episode endings for one term/stage/attempt combination."""

    termination: str
    stage: str
    attempts_completed: int
    successful: int
    failed: int

    @property
    def total(self) -> int:
        return self.successful + self.failed


class EpisodeEndAttribution:
    """Correlate termination terms with controller state and strict success."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, int, int, bool]] = Counter()

    def record(
        self,
        terminations: Iterable[str],
        stage: int,
        attempts_completed: int,
        successful: bool,
    ) -> None:
        """Record every active term because Isaac terms may overlap."""
        for termination in terminations:
            self._counts[
                (
                    termination,
                    int(stage),
                    max(0, int(attempts_completed)),
                    bool(successful),
                )
            ] += 1

    def rows(
        self, termination_names: Iterable[str] | None = None
    ) -> list[EpisodeEndRow]:
        """Return deterministic rows, optionally restricted to selected terms."""
        allowed = None if termination_names is None else set(termination_names)
        grouped: dict[tuple[str, int, int], list[int]] = {}
        for (termination, stage, attempts, successful), count in self._counts.items():
            if allowed is not None and termination not in allowed:
                continue
            outcome_counts = grouped.setdefault((termination, stage, attempts), [0, 0])
            outcome_counts[0 if successful else 1] += count

        rows = [
            EpisodeEndRow(
                termination=termination,
                stage=grasp_stage_label(stage),
                attempts_completed=attempts,
                successful=outcomes[0],
                failed=outcomes[1],
            )
            for (termination, stage, attempts), outcomes in grouped.items()
        ]
        return sorted(
            rows,
            key=lambda row: (
                row.termination,
                GRASP_STAGE_LABELS.index(row.stage)
                if row.stage in GRASP_STAGE_LABELS
                else len(GRASP_STAGE_LABELS),
                row.attempts_completed,
            ),
        )
