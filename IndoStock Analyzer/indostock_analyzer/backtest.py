from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean, stdev
from typing import Any


class BacktestDataError(ValueError):
    pass


@dataclass(frozen=True)
class BacktestPolicy:
    minimum_calibration_observations: int = 100
    minimum_holdout_observations: int = 100
    minimum_holdout_periods: int = 12
    minimum_symbols_per_period: int = 3
    minimum_calibration_bin_size: int = 20


SCORE_BINS = ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0))
TICKER_RE = re.compile(r"^[A-Z]{4}$")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BacktestDataError(f"cannot read JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BacktestDataError(f"invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise BacktestDataError(f"{path} must contain an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BacktestDataError(f"cannot read JSONL file: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BacktestDataError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise BacktestDataError(f"record at {path}:{line_number} must be an object")
        records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BacktestDataError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def validate_manifest(manifest: Mapping[str, Any], observations_hash: str | None = None) -> dict[str, Any]:
    required_strings = (
        "name", "version", "license", "source_url", "content_sha256", "indicator_version",
        "benchmark_id", "split_strategy", "calibration_end_at", "holdout_start_at",
        "return_type",
    )
    for key in required_strings:
        if not isinstance(manifest.get(key), str) or not str(manifest[key]).strip():
            raise BacktestDataError(f"manifest needs non-empty {key}")
    if observations_hash is not None and manifest["content_sha256"] != observations_hash:
        raise BacktestDataError("manifest content_sha256 does not match observations")
    if manifest["split_strategy"] != "temporal_holdout_with_embargo":
        raise BacktestDataError("split_strategy must be temporal_holdout_with_embargo")
    if manifest["return_type"] != "total_return":
        raise BacktestDataError("return_type must be total_return")
    for key in ("point_in_time_universe", "includes_inactive_symbols", "corporate_action_adjusted"):
        if manifest.get(key) is not True:
            raise BacktestDataError(f"manifest requires {key}=true")
    calibration_end = _timestamp(manifest["calibration_end_at"], "calibration_end_at")
    holdout_start = _timestamp(manifest["holdout_start_at"], "holdout_start_at")
    if calibration_end >= holdout_start:
        raise BacktestDataError("holdout_start_at must be after calibration_end_at")
    periods_per_year = _positive_number(manifest.get("periods_per_year"), "periods_per_year")
    top_fraction = _positive_number(manifest.get("top_fraction"), "top_fraction")
    if top_fraction > 1:
        raise BacktestDataError("top_fraction cannot exceed 1")
    variants = manifest.get("variants_tried")
    if isinstance(variants, bool) or not isinstance(variants, int) or variants < 1:
        raise BacktestDataError("variants_tried must be a positive integer")
    costs = manifest.get("costs_bps")
    if not isinstance(costs, Mapping):
        raise BacktestDataError("manifest needs costs_bps")
    normalized_costs = {}
    for key in ("commission", "slippage", "taxes"):
        value = _nonnegative_number(costs.get(key), f"costs_bps.{key}")
        normalized_costs[key] = value
    if sum(normalized_costs.values()) > 1_000:
        raise BacktestDataError("one-way costs cannot exceed 1000 bps")
    return {
        **dict(manifest),
        "calibration_end_at": calibration_end,
        "holdout_start_at": holdout_start,
        "periods_per_year": periods_per_year,
        "top_fraction": top_fraction,
        "costs_bps": normalized_costs,
    }


def validate_observations(
    records: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, datetime]] = set()
    result = {"calibration": [], "holdout": []}
    cohort_windows: dict[datetime, tuple[datetime, datetime]] = {}
    for index, record in enumerate(records, 1):
        record_id = _required_string(record, "id", index)
        if record_id in seen_ids:
            raise BacktestDataError(f"duplicate observation id: {record_id}")
        seen_ids.add(record_id)
        ticker = _required_string(record, "ticker", index).upper()
        if not TICKER_RE.fullmatch(ticker):
            raise BacktestDataError(f"observation {record_id} has invalid IDX ticker")
        signal_at = _timestamp(record.get("signal_at"), f"{record_id}.signal_at")
        outcome_start = _timestamp(record.get("outcome_start"), f"{record_id}.outcome_start")
        outcome_end = _timestamp(record.get("outcome_end"), f"{record_id}.outcome_end")
        if not signal_at < outcome_start < outcome_end:
            raise BacktestDataError(f"observation {record_id} has look-ahead/invalid outcome timing")
        key = (ticker, signal_at)
        if key in seen_keys:
            raise BacktestDataError(f"duplicate ticker/signal observation: {ticker} {signal_at.isoformat()}")
        seen_keys.add(key)
        inputs = record.get("inputs_available_at")
        if not isinstance(inputs, Mapping) or not inputs:
            raise BacktestDataError(f"observation {record_id} needs inputs_available_at")
        for input_name, value in inputs.items():
            available_at = _timestamp(value, f"{record_id}.inputs_available_at.{input_name}")
            if available_at > signal_at:
                raise BacktestDataError(f"observation {record_id} uses {input_name} after signal_at")
        universe_as_of = _timestamp(record.get("universe_as_of"), f"{record_id}.universe_as_of")
        if universe_as_of > signal_at:
            raise BacktestDataError(f"observation {record_id} uses a future universe snapshot")
        if record.get("eligible_at_signal") is not True:
            raise BacktestDataError(f"observation {record_id} was not eligible at signal time")
        snapshot_id = _required_string(record, "universe_snapshot_id", index)
        score = _finite_number(record.get("score"), f"{record_id}.score")
        if not 1 <= score <= 5:
            raise BacktestDataError(f"observation {record_id} score must be within 1..5")
        forward_return = _finite_number(
            record.get("forward_total_return"), f"{record_id}.forward_total_return"
        )
        benchmark_return = _finite_number(
            record.get("benchmark_total_return"), f"{record_id}.benchmark_total_return"
        )
        if forward_return <= -1 or benchmark_return <= -1:
            raise BacktestDataError(f"observation {record_id} has impossible simple return")
        window = (outcome_start, outcome_end)
        if signal_at in cohort_windows and cohort_windows[signal_at] != window:
            raise BacktestDataError(f"cohort {signal_at.isoformat()} has inconsistent outcome windows")
        cohort_windows[signal_at] = window
        if signal_at <= manifest["calibration_end_at"]:
            split = "calibration"
            if outcome_end >= manifest["holdout_start_at"]:
                raise BacktestDataError(f"observation {record_id} crosses the holdout embargo")
        elif signal_at >= manifest["holdout_start_at"]:
            split = "holdout"
        else:
            raise BacktestDataError(f"observation {record_id} lies inside the embargo gap")
        result[split].append(
            {
                "id": record_id,
                "ticker": ticker,
                "signal_at": signal_at,
                "outcome_start": outcome_start,
                "outcome_end": outcome_end,
                "score": score,
                "forward_total_return": forward_return,
                "benchmark_total_return": benchmark_return,
                "universe_snapshot_id": snapshot_id,
            }
        )
    if not result["calibration"] or not result["holdout"]:
        raise BacktestDataError("observations need both calibration and holdout records")
    windows = sorted(set(cohort_windows.values()))
    for previous, current in zip(windows, windows[1:]):
        if current[0] < previous[1]:
            raise BacktestDataError("outcome cohorts overlap; pre-aggregate non-overlapping rebalances")
    return result


def build_backtest_report(
    split_records: Mapping[str, list[dict[str, Any]]],
    manifest: Mapping[str, Any],
    report_id: str,
    policy: BacktestPolicy = BacktestPolicy(),
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not report_id.strip():
        raise BacktestDataError("report_id cannot be empty")
    calibration = split_records["calibration"]
    holdout = split_records["holdout"]
    calibration_table = _calibration_table(calibration)
    global_probability = (sum(_outperformed(item) for item in calibration) + 1) / (len(calibration) + 2)
    probabilities = []
    outcomes = []
    for item in holdout:
        bin_name = _score_bin(item["score"])
        probability = calibration_table[bin_name]["probability_outperforming_benchmark"]
        probabilities.append(probability if probability is not None else global_probability)
        outcomes.append(_outperformed(item))
    portfolio = _portfolio_series(holdout, manifest)
    period_excess = [item["net_excess_return"] for item in portfolio]
    holdout_dates = {item["signal_at"] for item in holdout}
    per_date_counts = defaultdict(int)
    for item in holdout:
        per_date_counts[item["signal_at"]] += 1
    checks = {
        "minimum_calibration_observations": len(calibration) >= policy.minimum_calibration_observations,
        "minimum_holdout_observations": len(holdout) >= policy.minimum_holdout_observations,
        "minimum_holdout_periods": len(holdout_dates) >= policy.minimum_holdout_periods,
        "minimum_symbols_per_period": min(per_date_counts.values()) >= policy.minimum_symbols_per_period,
        "minimum_calibration_bin_size": all(
            item["observations"] >= policy.minimum_calibration_bin_size
            for item in calibration_table.values()
        ),
    }
    return {
        "schema_version": 1,
        "report_type": "backtest",
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_not_investment_validated",
        "disclaimer": "Historical simulation only; not financial advice or evidence of future performance.",
        "evidence": dict(evidence) if evidence is not None else {
            "level": "generated_from_bound_inputs",
            "independently_verified": False,
            "input_sha256": {},
        },
        "dataset": _serializable_manifest(manifest),
        "evaluation_policy": asdict(policy),
        "policy_checks": {"passed": all(checks.values()), "checks": checks},
        "sample": {
            "calibration_observations": len(calibration),
            "holdout_observations": len(holdout),
            "holdout_periods": len(holdout_dates),
            "variants_tried": manifest["variants_tried"],
        },
        "economic_calibration": {
            "definition": "Probability that total return exceeds the configured benchmark",
            "calibration_bins": calibration_table,
            "holdout_brier_score": mean((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)),
            "holdout_ece": _ece(probabilities, outcomes),
        },
        "association": {
            "mean_cross_sectional_spearman": _mean_information_coefficient(holdout),
            "score_buckets": _holdout_score_buckets(holdout),
        },
        "portfolio_simulation": _portfolio_metrics(
            portfolio, float(manifest["periods_per_year"]), period_excess
        ),
        "periods": portfolio,
        "limitations": [
            "Historical results do not establish future performance.",
            "Normal metrics can be unstable in small or regime-dependent samples.",
            "The simulator uses equal weights and the configured cost assumptions.",
            "Multiple-testing risk increases with variants_tried.",
        ],
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _calibration_table(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = {f"{low:.0f}-{high:.0f}": [] for low, high in SCORE_BINS}
    for item in records:
        groups[_score_bin(item["score"])].append(item)
    table = {}
    for name, items in groups.items():
        table[name] = {
            "observations": len(items),
            "probability_outperforming_benchmark": (
                (sum(_outperformed(item) for item in items) + 1) / (len(items) + 2)
                if items else None
            ),
            "mean_excess_return": (
                mean(item["forward_total_return"] - item["benchmark_total_return"] for item in items)
                if items else None
            ),
        }
    return table


def _portfolio_series(records: list[dict[str, Any]], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    cohorts: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        cohorts[item["signal_at"]].append(item)
    previous_weights: dict[str, float] = {}
    one_way_cost = sum(manifest["costs_bps"].values()) / 10_000
    periods = []
    for signal_at, items in sorted(cohorts.items()):
        count = max(1, math.ceil(len(items) * manifest["top_fraction"]))
        selected = sorted(items, key=lambda item: (-item["score"], item["ticker"]))[:count]
        weights = {item["ticker"]: 1 / count for item in selected}
        tickers = set(previous_weights) | set(weights)
        turnover = 0.5 * sum(abs(weights.get(ticker, 0) - previous_weights.get(ticker, 0)) for ticker in tickers)
        if not previous_weights:
            turnover = 1.0
        cost = turnover * one_way_cost
        gross_return = mean(item["forward_total_return"] for item in selected)
        benchmark_return = mean(item["benchmark_total_return"] for item in selected)
        net_return = gross_return - cost
        if net_return <= -1:
            raise BacktestDataError("modeled cost produces an impossible net simple return")
        periods.append(
            {
                "signal_at": signal_at.isoformat(),
                "outcome_start": selected[0]["outcome_start"].isoformat(),
                "outcome_end": selected[0]["outcome_end"].isoformat(),
                "selected_tickers": sorted(weights),
                "turnover": turnover,
                "cost": cost,
                "gross_total_return": gross_return,
                "net_total_return": net_return,
                "benchmark_total_return": benchmark_return,
                "net_excess_return": net_return - benchmark_return,
            }
        )
        previous_weights = weights
    return periods


def _portfolio_metrics(periods: list[dict[str, Any]], periods_per_year: float, excess: list[float]) -> dict[str, Any]:
    net = [item["net_total_return"] for item in periods]
    benchmark = [item["benchmark_total_return"] for item in periods]
    cumulative_net = _compound(net)
    cumulative_benchmark = _compound(benchmark)
    volatility = stdev(excess) if len(excess) > 1 else 0.0
    information_ratio = mean(excess) / volatility * math.sqrt(periods_per_year) if volatility else None
    return {
        "periods": len(periods),
        "cumulative_net_total_return": cumulative_net,
        "cumulative_benchmark_total_return": cumulative_benchmark,
        "cumulative_net_excess_return": cumulative_net - cumulative_benchmark,
        "annualized_net_total_return": (1 + cumulative_net) ** (periods_per_year / len(net)) - 1,
        "annualized_benchmark_total_return": (1 + cumulative_benchmark) ** (periods_per_year / len(benchmark)) - 1,
        "annualized_information_ratio": information_ratio,
        "maximum_drawdown": _max_drawdown(net),
        "mean_period_excess_return": mean(excess),
        "positive_excess_period_rate": mean(value > 0 for value in excess),
        "mean_turnover": mean(item["turnover"] for item in periods),
        "total_modeled_cost": sum(item["cost"] for item in periods),
        "block_bootstrap_95pct_mean_excess": _block_bootstrap_interval(excess),
    }


def _mean_information_coefficient(records: list[dict[str, Any]]) -> float | None:
    cohorts: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        cohorts[item["signal_at"]].append(item)
    values = []
    for items in cohorts.values():
        if len(items) >= 3:
            score_ranks = _ranks([item["score"] for item in items])
            return_ranks = _ranks([item["forward_total_return"] - item["benchmark_total_return"] for item in items])
            correlation = _correlation(score_ranks, return_ranks)
            if correlation is not None:
                values.append(correlation)
    return mean(values) if values else None


def _holdout_score_buckets(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for low, high in SCORE_BINS:
        items = [item for item in records if _score_bin(item["score"]) == f"{low:.0f}-{high:.0f}"]
        result[f"{low:.0f}-{high:.0f}"] = {
            "observations": len(items),
            "mean_excess_return": mean(
                item["forward_total_return"] - item["benchmark_total_return"] for item in items
            ) if items else None,
        }
    return result


def _score_bin(score: float) -> str:
    if score < 2:
        return "1-2"
    if score < 3:
        return "2-3"
    if score < 4:
        return "3-4"
    return "4-5"


def _outperformed(item: Mapping[str, Any]) -> int:
    return int(item["forward_total_return"] > item["benchmark_total_return"])


def _ece(probabilities: list[float], outcomes: list[int], bins: int = 10) -> float:
    buckets = [[] for _ in range(bins)]
    for probability, outcome in zip(probabilities, outcomes):
        buckets[min(int(probability * bins), bins - 1)].append((probability, outcome))
    return sum(
        len(bucket) / len(probabilities)
        * abs(mean(item[0] for item in bucket) - mean(item[1] for item in bucket))
        for bucket in buckets if bucket
    )


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2 + 1
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _compound(returns: list[float]) -> float:
    wealth = 1.0
    for value in returns:
        wealth *= 1 + value
    return wealth - 1


def _max_drawdown(returns: list[float]) -> float:
    wealth = peak = 1.0
    maximum = 0.0
    for value in returns:
        wealth *= 1 + value
        peak = max(peak, wealth)
        maximum = min(maximum, wealth / peak - 1)
    return maximum


def _block_bootstrap_interval(values: list[float], samples: int = 1000) -> list[float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(20261122)
    block = max(1, round(math.sqrt(len(values))))
    estimates = []
    for _ in range(samples):
        resampled = []
        while len(resampled) < len(values):
            start = generator.randrange(len(values))
            resampled.extend(values[(start + offset) % len(values)] for offset in range(block))
        estimates.append(mean(resampled[:len(values)]))
    estimates.sort()
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975) - 1]]


def _serializable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in manifest.items()
    }


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise BacktestDataError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BacktestDataError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise BacktestDataError(f"{name} must include a timezone")
    return parsed


def _required_string(record: Mapping[str, Any], key: str, index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BacktestDataError(f"record {index} needs non-empty {key}")
    return value.strip()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise BacktestDataError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BacktestDataError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise BacktestDataError(f"{name} must be finite")
    return number


def _positive_number(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise BacktestDataError(f"{name} must be positive")
    return number


def _nonnegative_number(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise BacktestDataError(f"{name} cannot be negative")
    return number
