"""Machine-readable and human-readable run reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, run: dict[str, Any]) -> None:
    metrics = run["metrics"]
    split = run["split"]
    lines = [
        "# Reproduction report",
        "",
        f"- Input: `{run['input']}`",
        f"- Matches: {run['matches']} total; {run['eligible_matches']} eligible at horizon",
        f"- Cohort exclusions: {run['excluded_before_horizon']} ended before horizon or lacked state",
        f"- Split rows: {run['train_matches']} train / {run['test_matches']} test",
        f"- Prediction horizon: {run['horizon_minutes']} minutes",
        f"- Split: {split['strategy']} on `{split['order_column']}`",
        f"- Boundary: {split['boundary_time_utc']} ({split['timezone']})",
        f"- Train range: {split['train_range_utc']['start']} to {split['train_range_utc']['end']}",
        f"- Test range: {split['test_range_utc']['start']} to {split['test_range_utc']['end']}",
        f"- Leakage audit: {'PASS' if run['leakage_audit']['passed'] else 'FAIL'}",
        "",
        "## Holdout metrics",
        "",
        *[f"- {name}: {value:.4f}" for name, value in metrics.items()],
        "",
        "> The bundled sample is a synthetic integrity fixture. Its metrics are not scientific results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(path: Path, run: dict[str, Any]) -> None:
    lines = [
        "# Early-game vs mid-game reproduction report",
        "",
        f"- Input: `{run['input']}`",
        f"- Source matches: {run['matches']}",
        "",
        "| Model | Horizon | Eligible | Excluded | Train | Test | Leakage audit | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    split_details: list[str] = []
    for name, model in run["models"].items():
        metrics = model["metrics"]
        lines.append(
            f"| {name} | {model['horizon_minutes']} min | {model['eligible_matches']} | "
            f"{model['excluded_before_horizon']} | {model['train_matches']} | "
            f"{model['test_matches']} | "
            f"{'PASS' if model['leakage_audit']['passed'] else 'FAIL'} | "
            f"{metrics['roc_auc']:.4f} |"
        )
        split = model["split"]
        split_details.extend(
            [
                "",
                f"- `{name}` boundary: {split['boundary_time_utc']} ({split['timezone']})",
                f"- `{name}` train range: {split['train_range_utc']['start']} to {split['train_range_utc']['end']}",
                f"- `{name}` test range: {split['test_range_utc']['start']} to {split['test_range_utc']['end']}",
            ]
        )
    lines.extend(split_details)
    lines.extend(
        [
            "",
            "The 10-minute model uses only features available by 10:00. The 20-minute model is a separate mid-game model; its metrics must not be described as early-game performance.",
            "",
            "> The bundled sample is a synthetic integrity fixture. Its metrics are not scientific results.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_report(path: Path, run: dict[str, Any]) -> None:
    lines = [
        "# Real-data temporal benchmark",
        "",
        f"- Input: `{run['input']}`",
        f"- Source matches: {run['matches']}",
        f"- Holdout fraction: {run['test_fraction']:.0%}",
        "",
        "| Model | Eligible | Test | Holdout ROC-AUC | 95% bootstrap CI | Fold mean ± SD | Pooled temporal OOF AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in run["models"].items():
        interval = model["holdout_roc_auc_interval"]
        temporal_cv = model["temporal_cv"]
        pooled = temporal_cv["pooled_roc_auc_interval"]
        lines.append(
            f"| {name} | {model['eligible_matches']} | {model['test_matches']} | "
            f"{model['holdout_metrics']['roc_auc']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] | "
            f"{temporal_cv['roc_auc_mean']:.4f} ± {temporal_cv['roc_auc_std']:.4f} | "
            f"{pooled['estimate']:.4f} [{pooled['lower']:.4f}, {pooled['upper']:.4f}] |"
        )

    paired = run["paired_comparison"]
    delta = paired["delta_interval"]
    lines.extend(
        [
            "",
            "## Paired comparison on the same 20-minute-eligible cohort",
            "",
            f"- Test matches: {paired['test_matches']}",
            f"- Early 10m ROC-AUC: {paired['early_10m_roc_auc']:.4f}",
            f"- Mid 20m ROC-AUC: {paired['mid_20m_roc_auc']:.4f}",
            f"- ΔAUC (mid − early): {delta['estimate_mid_minus_early']:+.4f}",
            f"- {delta['confidence']:.0%} paired bootstrap CI: [{delta['lower']:+.4f}, {delta['upper']:+.4f}]",
            "",
        ]
    )
    if delta["lower"] > 0:
        lines.append(
            "Observed result: the mid-game model is higher on this paired holdout, and the bootstrap interval excludes zero."
        )
    elif delta["upper"] < 0:
        lines.append(
            "Observed result: the early-game model is higher on this paired holdout, and the bootstrap interval excludes zero."
        )
    else:
        lines.append(
            "Observed result: the paired bootstrap interval includes zero, so this sample does not resolve which horizon is better."
        )
    paired_temporal = run["paired_temporal_comparison"]
    temporal_delta = paired_temporal["delta_interval"]
    early_temporal = paired_temporal["early_10m_roc_auc_interval"]
    mid_temporal = paired_temporal["mid_20m_roc_auc_interval"]
    lines.extend(
        [
            "",
            "## Paired expanding-window out-of-fold comparison",
            "",
            f"- Pooled test predictions: {paired_temporal['pooled_test_matches']}",
            f"- Early 10m pooled ROC-AUC: {early_temporal['estimate']:.4f} "
            f"[{early_temporal['lower']:.4f}, {early_temporal['upper']:.4f}]",
            f"- Mid 20m pooled ROC-AUC: {mid_temporal['estimate']:.4f} "
            f"[{mid_temporal['lower']:.4f}, {mid_temporal['upper']:.4f}]",
            f"- ΔAUC (mid − early): {temporal_delta['estimate_mid_minus_early']:+.4f}",
            f"- {temporal_delta['confidence']:.0%} paired bootstrap CI: "
            f"[{temporal_delta['lower']:+.4f}, {temporal_delta['upper']:+.4f}]",
        ]
    )
    lines.extend(
        [
            "",
            "The 20-minute result is a mid-game estimate and must not be presented as 10-minute prediction performance.",
            "Metrics remain specific to this finite Challenger/Americas/patch-window sample.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_external_validation_report(path: Path, run: dict[str, Any]) -> None:
    """Write a compact report that keeps external and temporal claims distinct."""
    lines = [
        "# External-region validation",
        "",
        "The preprocessing pipeline and model are fitted only on the training cohort. "
        "The disjoint external cohort is used once for evaluation without refitting.",
        "",
        f"- Training matches: {run['training']['matches']}",
        f"- Training regions: {run['training']['regions']}",
        f"- Training patches: {run['training']['patches']}",
        f"- Training time range: {run['training']['time_range_utc']['start']} to "
        f"{run['training']['time_range_utc']['end']}",
        f"- External matches: {run['external']['matches']}",
        f"- External regions: {run['external']['regions']}",
        f"- External patches: {run['external']['patches']}",
        f"- External time range: {run['external']['time_range_utc']['start']} to "
        f"{run['external']['time_range_utc']['end']}",
        f"- Match ID overlap: {run['match_id_overlap']}",
        f"- Region overlap: {run['region_overlap']}",
        "",
        "| Model | Training eligible | External eligible | External ROC-AUC | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, model in run["models"].items():
        interval = model["external_roc_auc_interval"]
        lines.append(
            f"| {name} | {model['training_matches']} | {model['external_matches']} | "
            f"{model['external_metrics']['roc_auc']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] |"
        )
    paired = run["paired_comparison"]
    delta = paired["delta_interval"]
    lines.extend(
        [
            "",
            "## Paired external comparison",
            "",
            f"- External matches eligible at 20 minutes: {paired['external_matches']}",
            f"- Early 10m ROC-AUC: {paired['early_10m_roc_auc']:.4f}",
            f"- Mid 20m ROC-AUC: {paired['mid_20m_roc_auc']:.4f}",
            f"- ΔAUC (mid − early): {delta['estimate_mid_minus_early']:+.4f}",
            f"- {delta['confidence']:.0%} paired bootstrap CI: "
            f"[{delta['lower']:+.4f}, {delta['upper']:+.4f}]",
            "",
            "The 20-minute score is a mid-game external-validation result and must not "
            "be described as 10-minute prediction performance.",
            "This cohort changes region, calendar period, and patch together; it tests "
            "combined transfer and cannot isolate a regional effect.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shift_validation_report(path: Path, run: dict[str, Any]) -> None:
    """Report same-region and cross-region evaluations of the same fitted models."""
    lines = [
        "# Controlled shift validation",
        "",
        "Each horizon model is fitted once on the training split, then evaluated without "
        "refitting on same-region and cross-region test cohorts.",
        "",
        f"- Training: {run['training']}",
        f"- Same-region test: {run['same_region']}",
        f"- Cross-region test: {run['cross_region']}",
        f"- Shared patches: {run['controlled_support']['shared_patches']}",
        f"- Overlapping time range: "
        f"{run['controlled_support']['overlap_time_range_utc']}",
        f"- Match ID overlap: {run['match_id_overlap']}",
        "",
        "| Model | Same-region N | Same-region AUC | Cross-region N | Cross-region AUC | ΔAUC cross − same | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in run["models"].items():
        same = model["same_region_roc_auc_interval"]
        cross = model["cross_region_roc_auc_interval"]
        delta = model["cross_minus_same_auc_interval"]
        lines.append(
            f"| {name} | {model['same_region_matches']} | {same['estimate']:.4f} | "
            f"{model['cross_region_matches']} | {cross['estimate']:.4f} | "
            f"{delta['estimate_second_minus_first']:+.4f} | "
            f"[{delta['lower']:+.4f}, {delta['upper']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "The cohort comparison is unpaired because it contains different matches. "
            "Its bootstrap interval samples the two cohorts independently.",
            "The 20-minute score remains a mid-game result and is not an early-game claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_nested_validation_report(path: Path, run: dict[str, Any]) -> None:
    """Report nested temporal model selection without reusing outer test rows."""
    lines = [
        "# Nested temporal validation",
        "",
        "Hyperparameters are selected only inside each outer training window. "
        "Each untouched outer future block is used once for evaluation.",
        "",
        f"- Input: `{run['input']}`",
        f"- Source matches: {run['matches']}",
        "",
        "| Model | Eligible | Outer OOF N | Nested pooled ROC-AUC | 95% bootstrap CI | Selected C counts |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, model in run["models"].items():
        nested = model["nested_validation"]
        interval = nested["pooled_outer_roc_auc_interval"]
        lines.append(
            f"| {name} | {model['eligible_matches']} | "
            f"{nested['pooled_outer_test_matches']} | {interval['estimate']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] | "
            f"{nested['selected_c_counts']} |"
        )

    paired = run["paired_comparison"]
    early = paired["early_10m"]["pooled_outer_roc_auc_interval"]
    mid = paired["mid_20m"]["pooled_outer_roc_auc_interval"]
    delta = paired["delta_interval"]
    lines.extend(
        [
            "",
            "## Paired nested comparison on the 20-minute-eligible cohort",
            "",
            f"- Pooled outer test matches: {paired['pooled_outer_test_matches']}",
            f"- Early 10m ROC-AUC: {early['estimate']:.4f} "
            f"[{early['lower']:.4f}, {early['upper']:.4f}]",
            f"- Mid 20m ROC-AUC: {mid['estimate']:.4f} "
            f"[{mid['lower']:.4f}, {mid['upper']:.4f}]",
            f"- ΔAUC (mid − early): {delta['estimate_mid_minus_early']:+.4f}",
            f"- {delta['confidence']:.0%} paired bootstrap CI: "
            f"[{delta['lower']:+.4f}, {delta['upper']:+.4f}]",
            "",
            "The 20-minute result is a mid-game nested-validation estimate and must not "
            "be presented as 10-minute performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

