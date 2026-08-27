"""
Shared configuration for the exploration/ pipeline (run_pipeline.py and
each stageN_*.py module's `main(config)`), so one config controls all 5
stages instead of each script having its own hardcoded constants.

Load order: defaults -> JSON config file -> CLI flags (each layer
overrides the one before it). See run_pipeline.py for the CLI.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ALL_MODELS = ["naive", "prophet", "lightgbm", "sarimax", "ensemble"]
ALL_STAGES = [1, 2, 3, 4, 5]
ENSEMBLE_INPUTS = ("prophet", "lightgbm", "sarimax")


@dataclass
class PipelineConfig:
    categories: list[str] | None = None  # None = every category present in the data
    models: list[str] = field(default_factory=lambda: list(ALL_MODELS))
    test_months: int = 12
    val_months: int = 6
    forecast_horizon: int = 12  # Stage 5's genuine future-forecast length
    skip_stages: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown_models = set(self.models) - set(ALL_MODELS)
        if unknown_models:
            raise ValueError(f"unknown model(s) in config: {sorted(unknown_models)} — choose from {ALL_MODELS}")
        if not self.models:
            raise ValueError("models list cannot be empty — at least one model is needed to produce a forecast")
        if "ensemble" in self.models:
            base = [m for m in ENSEMBLE_INPUTS if m in self.models]
            if len(base) < 2:
                raise ValueError(
                    f"'ensemble' needs at least 2 of {list(ENSEMBLE_INPUTS)} also enabled in models "
                    f"(got only {base}) — a 1-model 'stack' isn't an ensemble"
                )
        unknown_stages = set(self.skip_stages) - set(ALL_STAGES)
        if unknown_stages:
            raise ValueError(f"unknown stage number(s) to skip: {sorted(unknown_stages)} — valid stages are {ALL_STAGES}")
        if self.test_months < 1 or self.val_months < 1:
            raise ValueError("test_months and val_months must both be >= 1")
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")

    def runs(self, stage: int) -> bool:
        return stage not in self.skip_stages

    @classmethod
    def load(cls, path: Path | str | None) -> PipelineConfig:
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
