"""Load the frozen FlyVL-simple-v1 configuration (configs/sim_frozen.json)."""
from __future__ import annotations

import json
from pathlib import Path

from .sim import SimConfig
from .stimulus import EyeConfig

PATH = Path(__file__).resolve().parents[1] / "configs" / "sim_frozen.json"


def load(path: Path = PATH) -> tuple[SimConfig, EyeConfig, dict]:
    spec = json.loads(Path(path).read_text())
    eye = dict(spec["eye"])
    eye["drift_directions"] = tuple(eye["drift_directions"])
    return SimConfig(**spec["sim"]), EyeConfig(**eye), spec
