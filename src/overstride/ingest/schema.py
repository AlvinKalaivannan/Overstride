"""Stage 1 input schema: what an athlete actually logs, one day at a time.

The athlete never sees the 22-feature weekly vector Stage 2 scores against —
they just log runs. See ingest/aggregate.py for the aggregation step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

VALID_ZONES = ("Z1", "Z2", "Z3", "Z4")


@dataclass
class SessionLog:
    date: date
    distance_km: float
    zone_km: dict[str, float] = field(default_factory=dict)
    rpe: int | None = None
    perceived_training_success: int | None = None
    strength_training_count: int = 0
    cross_training_hours: float = 0.0
    rest_day: bool = False

    def __post_init__(self) -> None:
        if self.distance_km < 0:
            raise ValueError(f"distance_km must be >= 0, got {self.distance_km}")

        for zone, km in self.zone_km.items():
            if zone not in VALID_ZONES:
                raise ValueError(f"zone_km key {zone!r} is not one of {VALID_ZONES}")
            if km < 0:
                raise ValueError(f"zone_km[{zone!r}] must be >= 0, got {km}")

        zone_total = sum(self.zone_km.values())
        if zone_total > self.distance_km + 1e-6:
            raise ValueError(
                f"zone_km sums to {zone_total}, which exceeds distance_km ({self.distance_km})"
            )

        if self.strength_training_count < 0:
            raise ValueError(
                f"strength_training_count must be >= 0, got {self.strength_training_count}"
            )
        if self.cross_training_hours < 0:
            raise ValueError(f"cross_training_hours must be >= 0, got {self.cross_training_hours}")

        if self.rest_day:
            if self.distance_km != 0 or self.zone_km:
                raise ValueError("rest_day=True cannot have distance_km or zone_km logged")
            if self.rpe is not None or self.perceived_training_success is not None:
                raise ValueError("rest_day=True cannot have rpe or perceived_training_success logged")
            return

        if self.distance_km > 0:
            if self.rpe is None or self.perceived_training_success is None:
                raise ValueError(
                    "rpe and perceived_training_success are required when distance_km > 0"
                )

        if self.rpe is not None and not (1 <= self.rpe <= 10):
            raise ValueError(f"rpe must be between 1 and 10, got {self.rpe}")
        if self.perceived_training_success is not None and not (
            1 <= self.perceived_training_success <= 5
        ):
            raise ValueError(
                "perceived_training_success must be between 1 and 5, "
                f"got {self.perceived_training_success}"
            )
