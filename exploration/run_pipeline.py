"""
Single entry point for the exploration/ pipeline: runs Stages 1-5 in
order, driven by the one PipelineConfig defined below instead of each
stage's own hardcoded defaults. Every knob the config exposes — which
categories, which models, the walk-forward split sizes, the Stage 5
forecast horizon, and which stages to skip — is respected by every stage
that's actually run. This is the only place pipeline settings live;
stageN_*.py modules import PipelineConfig from here rather than defining
or duplicating any of it.

Config resolution order: the defaults below -> an optional --config JSON
file -> individual CLI flags (each layer overrides the previous one).

Usage:
    python exploration/run_pipeline.py
    python exploration/run_pipeline.py --config my_config.json
    python exploration/run_pipeline.py --categories vodka,rum --models naive,lightgbm
    python exploration/run_pipeline.py --skip 1,2 --forecast-horizon 6
    python exploration/run_pipeline.py --test-months 6 --val-months 3

Stage dependencies: each stage reads the previous stage's output from
exploration/outputs/, so skipping a stage only works if that output
already exists on disk from an earlier run. Stage 1 has no downstream
dependents (nothing reads its output), so skipping it never blocks
anything else.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ALL_MODELS = ["naive", "prophet", "lightgbm", "sarimax", "ensemble"]
ALL_STAGES = [1, 2, 3, 4, 5]
ENSEMBLE_INPUTS = ("prophet", "lightgbm", "sarimax")


@dataclass
class PipelineConfig:
    """The one settings object for the whole exploration/ pipeline. Every
    stageN_*.py module's `main(config)` reads whichever fields it needs
    from this; none of them define their own defaults for these values."""

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


STAGE_MODULES = {
    1: ("stage1_eda", "Stage 1: EDA"),
    2: ("stage2_data_prep", "Stage 2: Data prep"),
    3: ("stage3_modeling", "Stage 3: Modeling"),
    4: ("stage4_evaluation", "Stage 4: Evaluation"),
    5: ("stage5_champion_report", "Stage 5: Champion report"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="path to a JSON config file overriding the defaults above")
    parser.add_argument("--categories", type=str, default=None, help="comma-separated liquor types, e.g. vodka,rum (default: all)")
    parser.add_argument("--models", type=str, default=None, help=f"comma-separated subset of {ALL_MODELS} (default: all)")
    parser.add_argument("--test-months", type=int, default=None, help="length of the test holdout, in months")
    parser.add_argument("--val-months", type=int, default=None, help="length of the validation window, in months")
    parser.add_argument("--forecast-horizon", type=int, default=None, help="Stage 5's genuine future-forecast length, in months")
    parser.add_argument("--skip", type=str, default=None, help="comma-separated stage numbers to skip, e.g. 1,2")
    parser.add_argument("--save-config", type=str, default=None, help="write the resolved config to this path and exit, without running anything")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig.load(args.config)
    overrides = {}
    if args.categories is not None:
        overrides["categories"] = [c.strip() for c in args.categories.split(",") if c.strip()]
    if args.models is not None:
        overrides["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.test_months is not None:
        overrides["test_months"] = args.test_months
    if args.val_months is not None:
        overrides["val_months"] = args.val_months
    if args.forecast_horizon is not None:
        overrides["forecast_horizon"] = args.forecast_horizon
    if args.skip is not None:
        overrides["skip_stages"] = [int(s.strip()) for s in args.skip.split(",") if s.strip()]

    if not overrides:
        return config
    # PipelineConfig is a plain dataclass; re-validate through __post_init__ by rebuilding it.
    merged = {**asdict(config), **overrides}
    return PipelineConfig(**merged)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = resolve_config(args)

    if args.save_config:
        config.save(args.save_config)
        print(f"Wrote resolved config to {args.save_config}")
        return

    print("Resolved pipeline config:")
    print(f"  categories:       {config.categories or 'all'}")
    print(f"  models:           {config.models}")
    print(f"  test_months:      {config.test_months}")
    print(f"  val_months:       {config.val_months}")
    print(f"  forecast_horizon: {config.forecast_horizon}")
    print(f"  skip_stages:      {config.skip_stages or 'none'}")
    print()

    for stage_num in sorted(STAGE_MODULES):
        module_name, label = STAGE_MODULES[stage_num]
        if not config.runs(stage_num):
            print(f"--- {label}: SKIPPED (in skip_stages) ---\n")
            continue

        print(f"--- {label} ---")
        t0 = time.time()
        module = __import__(module_name)
        module.main(config)
        print(f"--- {label} done in {time.time() - t0:.1f}s ---\n")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
