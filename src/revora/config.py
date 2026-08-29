"""Load reproducibility settings from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


@dataclass(frozen=True)
class Settings:
    seed: int
    dataset_size: int
    held_out_fraction: float
    dev_fraction: float


def load_settings() -> Settings:
    return Settings(
        seed=_int("REVORA_SEED", 42),
        dataset_size=_int("REVORA_DATASET_SIZE", 400),
        held_out_fraction=_float("REVORA_HELD_OUT_FRACTION", 0.20),
        dev_fraction=_float("REVORA_DEV_FRACTION", 0.15),
    )
