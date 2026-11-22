"""Repeated nested temporal validation with training-refit block bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import features_available_at
from .final_results import authorize_pre_release_tuning
from .data import read_events
from .features import build_match_features, eligible_at_horizon
from .leakage import assert_leakage_safe
from .nested_validation import DEFAULT_C_GRID, repeated_nested_temporal_validate
from .reporting import write_json
from .split import DEFAULT_TIME_COLUMN
from .validation import paired_auc_delta_interval


def _write_report(path: Path, run: dict[str, Any]) -> None:
    lines = [
        "# Repeated nested temporal robustness validation",
        "",
        "Temporal cut points are shifted across repeats while every inner and outer split "
        "remains strictly forward-looking. The unshifted repeat additionally resamples "
        "contiguous training blocks and refits preprocessing plus the classifier in every "
        "bootstrap iteration.",
        "",
        f"- Input: `{run['input']}`",
        f"- Source matches: {run['matches']}",
        f"- Repeats: {run['configuration']['repeats']}",
        f"- Outer/inner splits: {run['configuration']['outer_splits']} / "
        f"{run['configuration']['inner_splits']}",
        f"- Refit bootstrap: {run['configuration']['refit_bootstrap_iterations']} "
        f"iterations per base outer fold; block size "
        f"{run['configuration']['refit_block_size']} matches",
        "- Frozen future holdout accessed: no",
        "",
        "| Model | Eligible | Repeat AUC mean ± SD | Repeat range | Selected C counts |",
        "|---|---:|---:|---:|---|",
    ]
    for name, model in run["models"].items():
        validation = model["repeated_nested_validation"]
        lines.append(
            f"| {name} | {model['eligible_matches']} | "
            f"{validation['repeat_roc_auc_mean']:.4f} ± "
            f"{validation['repeat_roc_auc_std']:.4f} | "
            f"{validation['repeat_roc_auc_min']:.4f}–"
            f"{validation['repeat_roc_auc_max']:.4f} | "
            f"{validation['selected_c_counts']} |"
        )

    lines.extend(["", "## Base-repeat block-bootstrap refit intervals", ""])
    for name, model in run["models"].items():
        base = next(
            repeat
            for repeat in model["repeated_nested_validation"]["repeat_summaries"]
            if repeat["is_base_repeat"]
        )
        lines.append(f"### {name}")
        lines.append("")
        for fold in base["nested_validation"]["folds"]:
            interval = fold["block_bootstrap_refit_auc_interval"]
            lines.append(
                f"- Outer fold {fold['outer_fold']}: {interval['estimate']:.4f} "
                f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
            )
        lines.append("")

    paired = run["paired_comparison"]
    conditional = paired["base_repeat_prediction_bootstrap_delta_interval"]
    lines.extend(
        [
            "## Paired horizon comparison",
            "",
            f"- Repeat ΔAUC mean (mid − early): {paired['repeat_delta_mean']:+.4f}",
            f"- Repeat ΔAUC SD: {paired['repeat_delta_std']:.4f}",
            f"- Repeat ΔAUC range: [{paired['repeat_delta_min']:+.4f}, "
            f"{paired['repeat_delta_max']:+.4f}]",
            f"- Base-repeat prediction-bootstrap ΔAUC: "
            f"{conditional['estimate_mid_minus_early']:+.4f} "
            f"[{conditional['lower']:+.4f}, {conditional['upper']:+.4f}]",
            "",
            "Repeat results are correlated sensitivity analyses, not independent experiments. "
            "Their standard deviation is not a confidence interval. The paired base-repeat "
            "interval conditions on fitted models; training-refit uncertainty is reported "
            "per model and outer fold above.",
            "",
            "The 20-minute results are mid-game estimates and must not be presented as "
            "10-minute early-game performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_robustness_validation(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    cohort_policy_path: str | Path | None = None,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    outer_splits: int = 3,
    inner_splits: int = 2,
    repeats: int = 5,
    max_shift_fraction: float = 0.4,
    bootstrap_iterations: int = 2_000,
    refit_bootstrap_iterations: int = 100,
    refit_block_size: int = 50,
) -> dict[str, Any]:
    """Run robustness analysis only on the development/training acquisition."""
    if refit_bootstrap_iterations < 100:
        raise ValueError("refit_bootstrap_iterations must be at least 100")
    if refit_block_size < 2:
        raise ValueError("refit_block_size must be at least 2")
    source = Path(input_path)
    if cohort_policy_path is not None:
        authorize_pre_release_tuning(
            cohort_policy_path, source, "model_selection"
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    match_features = build_match_features(read_events(source))

    models: dict[str, Any] = {}
    for horizon_minutes, key in ((10, "early_10m"), (20, "mid_20m")):
        cohort = eligible_at_horizon(match_features, horizon_minutes)
        features = features_available_at(horizon_minutes * 60_000)
        audit = assert_leakage_safe(cohort, features, horizon_minutes * 60_000)
        validation, predictions = repeated_nested_temporal_validate(
            cohort,
            features,
            c_grid=c_grid,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            repeats=repeats,
            max_shift_fraction=max_shift_fraction,
            bootstrap_iterations=bootstrap_iterations,
            refit_bootstrap_iterations=refit_bootstrap_iterations,
            refit_block_size=refit_block_size,
        )
        predictions_name = f"repeated_nested_predictions_{horizon_minutes}m.csv"
        predictions.to_csv(destination / predictions_name, index=False)
        models[key] = {
            "horizon_minutes": horizon_minutes,
            "eligible_matches": len(cohort),
            "features": features,
            "leakage_audit": audit.to_dict(),
            "repeated_nested_validation": validation,
            "predictions_path": predictions_name,
        }

    common = eligible_at_horizon(match_features, 20)
    paired_results: dict[str, dict[str, Any]] = {}
    paired_predictions: dict[str, pd.DataFrame] = {}
    for horizon_minutes, key in ((10, "early_10m"), (20, "mid_20m")):
        validation, predictions = repeated_nested_temporal_validate(
            common,
            features_available_at(horizon_minutes * 60_000),
            c_grid=c_grid,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            repeats=repeats,
            max_shift_fraction=max_shift_fraction,
            bootstrap_iterations=bootstrap_iterations,
            refit_bootstrap_iterations=refit_bootstrap_iterations,
            refit_block_size=refit_block_size,
        )
        paired_results[key] = validation
        paired_predictions[key] = predictions

    merge_keys = ["repeat", "boundary_offset", "outer_fold", "matchId"]
    early = paired_predictions["early_10m"].rename(
        columns={"win_probability": "early_10m_probability"}
    )
    mid = paired_predictions["mid_20m"].rename(
        columns={"win_probability": "mid_20m_probability"}
    )
    paired = early[
        [*merge_keys, DEFAULT_TIME_COLUMN, "win", "early_10m_probability"]
    ].merge(
        mid[[*merge_keys, "mid_20m_probability"]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(early) or len(paired) != len(mid):
        raise AssertionError("Repeated paired predictions do not align")
    repeat_comparisons: list[dict[str, Any]] = []
    for repeat, group in paired.groupby("repeat", sort=True):
        early_auc = roc_auc_score(group["win"], group["early_10m_probability"])
        mid_auc = roc_auc_score(group["win"], group["mid_20m_probability"])
        repeat_comparisons.append(
            {
                "repeat": int(repeat),
                "boundary_offset": int(group["boundary_offset"].iloc[0]),
                "matches": len(group),
                "early_10m_roc_auc": float(early_auc),
                "mid_20m_roc_auc": float(mid_auc),
                "delta_mid_minus_early": float(mid_auc - early_auc),
            }
        )
    base = paired.loc[paired["boundary_offset"].eq(0)]
    base_delta = paired_auc_delta_interval(
        base["win"].to_numpy(),
        base["early_10m_probability"].to_numpy(),
        base["mid_20m_probability"].to_numpy(),
        iterations=bootstrap_iterations,
    )
    deltas = np.asarray(
        [comparison["delta_mid_minus_early"] for comparison in repeat_comparisons]
    )
    paired_name = "repeated_paired_predictions.csv"
    paired.to_csv(destination / paired_name, index=False)

    summary: dict[str, Any] = {
        "input": source.as_posix(),
        "matches": len(match_features),
        "validation_design": "repeated_nested_temporal_with_block_bootstrap_refitting",
        "frozen_future_holdout_accessed": False,
        "configuration": {
            "c_grid": list(float(value) for value in c_grid),
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "repeats": repeats,
            "max_shift_fraction": max_shift_fraction,
            "prediction_bootstrap_iterations": bootstrap_iterations,
            "refit_bootstrap_iterations": refit_bootstrap_iterations,
            "refit_block_size": refit_block_size,
            "training_refit_bootstrap_total_fits": (
                4 * outer_splits * refit_bootstrap_iterations
            ),
        },
        "models": models,
        "paired_comparison": {
            "cohort": "matches_eligible_at_20m",
            "repeat_comparisons": repeat_comparisons,
            "repeat_delta_mean": float(deltas.mean()),
            "repeat_delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "repeat_delta_min": float(deltas.min()),
            "repeat_delta_max": float(deltas.max()),
            "base_repeat_prediction_bootstrap_delta_interval": base_delta,
            "early_10m": paired_results["early_10m"],
            "mid_20m": paired_results["mid_20m"],
            "predictions_path": paired_name,
        },
    }
    write_json(destination / "robustness_validation.json", summary)
    _write_report(destination / "robustness_validation.md", summary)
    return summary
