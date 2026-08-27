"""Configuration loading and shared paths."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "data" / "outputs"
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class Split:
    """Temporal backtest boundaries."""

    history_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date

    @property
    def horizon_days(self) -> int:
        return (self.test_end - self.test_start).days + 1


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)

    # --- convenience accessors -------------------------------------------------
    @property
    def gcp_project(self) -> str:
        return self.raw["project"]["gcp_project"]

    @property
    def dataset(self) -> str:
        return self.raw["project"]["prod_dataset"]

    @property
    def mapper(self) -> str:
        return self.raw["project"]["mapper_table"]

    @property
    def focal_program(self) -> str:
        return self.raw["programs"]["focal"]

    @property
    def all_programs(self) -> list[str]:
        return [self.focal_program, *self.raw["programs"].get("auxiliary", [])]

    @property
    def split(self) -> Split:
        s = self.raw["split"]
        return Split(
            history_start=_as_date(s["history_start"]),
            train_end=_as_date(s["train_end"]),
            test_start=_as_date(s["test_start"]),
            test_end=_as_date(s["test_end"]),
        )

    @property
    def business(self) -> dict[str, Any]:
        return self.raw["business"]

    @property
    def sizes(self) -> dict[str, Any]:
        return self.raw["sizes"]

    @property
    def seed(self) -> int:
        return int(self.raw["modeling"]["random_seed"])


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


@lru_cache(maxsize=1)
def load_config(path: Path | str = CONFIG_PATH) -> Config:
    with open(path) as fh:
        return Config(raw=yaml.safe_load(fh))


def ensure_dirs() -> None:
    for d in (RAW, PROCESSED, OUTPUTS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
