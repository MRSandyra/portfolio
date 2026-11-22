import hashlib
import json

import pytest

from indostock_analyzer.backtest import (
    BacktestDataError,
    BacktestPolicy,
    build_backtest_report,
    validate_manifest,
    validate_observations,
)
from indostock_analyzer.backtest_cli import main as backtest_cli_main
from indostock_analyzer.artifacts import validate_artifact, verify_checksum
from indostock_analyzer.artifact_cli import main as audit_cli_main
from indostock_analyzer.evidence import EvidenceVerificationError, verify_backtest_evidence


def manifest(validate=True, **updates):
    value = {
        "name": "fixture",
        "version": "1",
        "license": "CC-BY-4.0",
        "source_url": "https://example.test/dataset",
        "content_sha256": "abc123",
        "indicator_version": "2.3.0-fixture",
        "benchmark_id": "fixture-total-return-index",
        "split_strategy": "temporal_holdout_with_embargo",
        "calibration_end_at": "2024-01-10T00:00:00+07:00",
        "holdout_start_at": "2024-02-01T00:00:00+07:00",
        "point_in_time_universe": True,
        "includes_inactive_symbols": True,
        "corporate_action_adjusted": True,
        "return_type": "total_return",
        "periods_per_year": 12,
        "top_fraction": 0.2,
        "variants_tried": 1,
        "costs_bps": {"commission": 10, "slippage": 5, "taxes": 0},
    }
    value.update(updates)
    return validate_manifest(value) if validate else value


def observation(record_id, ticker, signal, start, end, score, stock_return, benchmark=0.01):
    return {
        "id": record_id,
        "ticker": ticker,
        "signal_at": signal,
        "outcome_start": start,
        "outcome_end": end,
        "inputs_available_at": {
            "news": signal,
            "fundamental": signal,
            "technical": signal,
        },
        "universe_as_of": signal,
        "universe_snapshot_id": f"universe-{signal[:10]}",
        "eligible_at_signal": True,
        "score": score,
        "forward_total_return": stock_return,
        "benchmark_total_return": benchmark,
    }


def records():
    calibration = [
        observation("c1", "AAAA", "2024-01-01T00:00:00+07:00", "2024-01-02T00:00:00+07:00", "2024-01-05T00:00:00+07:00", 1.5, -0.03),
        observation("c2", "BBBB", "2024-01-01T00:00:00+07:00", "2024-01-02T00:00:00+07:00", "2024-01-05T00:00:00+07:00", 2.5, 0.00),
        observation("c3", "CCCC", "2024-01-01T00:00:00+07:00", "2024-01-02T00:00:00+07:00", "2024-01-05T00:00:00+07:00", 3.5, 0.02),
        observation("c4", "DDDD", "2024-01-01T00:00:00+07:00", "2024-01-02T00:00:00+07:00", "2024-01-05T00:00:00+07:00", 4.5, 0.05),
    ]
    holdout = [
        observation("h1", "AAAA", "2024-02-01T00:00:00+07:00", "2024-02-02T00:00:00+07:00", "2024-02-05T00:00:00+07:00", 2.0, -0.02),
        observation("h2", "BBBB", "2024-02-01T00:00:00+07:00", "2024-02-02T00:00:00+07:00", "2024-02-05T00:00:00+07:00", 3.0, 0.02),
        observation("h3", "CCCC", "2024-02-01T00:00:00+07:00", "2024-02-02T00:00:00+07:00", "2024-02-05T00:00:00+07:00", 5.0, 0.06),
    ]
    return calibration + holdout


def small_policy():
    return BacktestPolicy(4, 3, 1, 3, 1)


def test_manifest_requires_point_in_time_survivorship_and_total_returns():
    for key in ("point_in_time_universe", "includes_inactive_symbols", "corporate_action_adjusted"):
        raw = manifest(validate=False)
        raw[key] = False
        with pytest.raises(BacktestDataError, match=key):
            validate_manifest(raw)
    with pytest.raises(BacktestDataError, match="total_return"):
        validate_manifest(manifest(validate=False, return_type="price_return"))


def test_future_input_is_rejected_as_lookahead():
    values = records()
    values[0]["inputs_available_at"]["fundamental"] = "2024-01-03T00:00:00+07:00"

    with pytest.raises(BacktestDataError, match="after signal_at"):
        validate_observations(values, manifest())


def test_future_universe_and_ineligible_symbol_are_rejected():
    values = records()
    values[0]["universe_as_of"] = "2024-01-02T00:00:00+07:00"
    with pytest.raises(BacktestDataError, match="future universe"):
        validate_observations(values, manifest())
    values = records()
    values[0]["eligible_at_signal"] = False
    with pytest.raises(BacktestDataError, match="not eligible"):
        validate_observations(values, manifest())


def test_score_return_and_embargo_boundaries_are_enforced():
    values = records()
    values[0]["score"] = 5.01
    with pytest.raises(BacktestDataError, match="within 1..5"):
        validate_observations(values, manifest())
    values = records()
    values[0]["forward_total_return"] = -1
    with pytest.raises(BacktestDataError, match="impossible simple return"):
        validate_observations(values, manifest())
    values = records()
    values[0]["outcome_end"] = "2024-02-02T00:00:00+07:00"
    with pytest.raises(BacktestDataError, match="crosses the holdout embargo"):
        validate_observations(values, manifest())


def test_report_uses_only_calibration_probabilities_and_holdout_performance():
    configured = manifest()
    split = validate_observations(records(), configured)

    report = build_backtest_report(split, configured, "fixture-report", small_policy())

    assert report["status"] == "research_only_not_investment_validated"
    assert report["policy_checks"]["passed"] is True
    assert report["sample"]["calibration_observations"] == 4
    assert report["sample"]["holdout_observations"] == 3
    assert report["association"]["mean_cross_sectional_spearman"] == pytest.approx(1.0)
    assert report["periods"][0]["selected_tickers"] == ["CCCC"]
    assert report["economic_calibration"]["calibration_bins"]["4-5"]["observations"] == 1


def test_transaction_cost_is_deducted_from_simulation():
    with_cost = manifest()
    without_cost = manifest(costs_bps={"commission": 0, "slippage": 0, "taxes": 0})
    split = validate_observations(records(), with_cost)

    costly = build_backtest_report(split, with_cost, "costly", small_policy())
    free = build_backtest_report(split, without_cost, "free", small_policy())

    assert costly["periods"][0]["cost"] == pytest.approx(0.0015)
    assert costly["portfolio_simulation"]["cumulative_net_total_return"] < free["portfolio_simulation"]["cumulative_net_total_return"]


def test_overlapping_outcome_cohorts_are_rejected():
    values = records()
    values.append(
        observation("h4", "DDDD", "2024-02-03T00:00:00+07:00", "2024-02-04T00:00:00+07:00", "2024-02-07T00:00:00+07:00", 4.0, 0.03)
    )
    with pytest.raises(BacktestDataError, match="overlap"):
        validate_observations(values, manifest())


def test_cli_verifies_hash_and_writes_research_only_report(tmp_path):
    observations_path = tmp_path / "observations.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "report.json"
    observations_path.write_text(
        "\n".join(json.dumps(item) for item in records()) + "\n",
        encoding="utf-8",
    )
    content_hash = hashlib.sha256(observations_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest(validate=False, content_sha256=content_hash)),
        encoding="utf-8",
    )

    exit_code = backtest_cli_main([
        "--observations", str(observations_path),
        "--manifest", str(manifest_path),
        "--report-id", "fixture-cli",
        "--output", str(output_path),
    ])

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "research_only_not_investment_validated"
    assert report["policy_checks"]["passed"] is False
    assert report["evidence"]["level"] == "generated_from_bound_inputs"
    assert set(report["evidence"]["input_sha256"]) == {"observations", "manifest"}
    assert validate_artifact(report) == "backtest"
    assert verify_checksum(output_path) is True
    verification = verify_backtest_evidence(
        report,
        observations_path=observations_path,
        manifest_path=manifest_path,
    )
    assert verification.report_id == "fixture-cli"
    assert "economic_calibration" in verification.recomputed_sections
    assert audit_cli_main([
        "verify-backtest-evidence",
        str(output_path),
        "--observations", str(observations_path),
        "--manifest", str(manifest_path),
    ]) == 0

    forged = dict(report)
    forged["association"] = {
        **report["association"],
        "mean_cross_sectional_spearman": -0.75,
    }
    assert validate_artifact(forged) == "backtest"
    with pytest.raises(EvidenceVerificationError, match="mean_cross_sectional_spearman"):
        verify_backtest_evidence(
            forged,
            observations_path=observations_path,
            manifest_path=manifest_path,
        )


def test_cli_rejects_observation_hash_mismatch(tmp_path):
    observations_path = tmp_path / "observations.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "report.json"
    observations_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest(validate=False, content_sha256="wrong")),
        encoding="utf-8",
    )

    exit_code = backtest_cli_main([
        "--observations", str(observations_path),
        "--manifest", str(manifest_path),
        "--report-id", "fixture-cli",
        "--output", str(output_path),
    ])

    assert exit_code == 2
    assert output_path.exists() is False


def test_cli_rejects_checksum_valid_but_semantically_future_input(tmp_path):
    observations_path = tmp_path / "observations.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "report.json"
    values = records()
    values[0]["inputs_available_at"]["fundamental"] = "2024-01-03T00:00:00+07:00"
    observations_path.write_text(
        "\n".join(json.dumps(item) for item in values) + "\n",
        encoding="utf-8",
    )
    content_hash = hashlib.sha256(observations_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest(validate=False, content_sha256=content_hash)),
        encoding="utf-8",
    )

    exit_code = backtest_cli_main([
        "--observations", str(observations_path),
        "--manifest", str(manifest_path),
        "--report-id", "future-input",
        "--output", str(output_path),
    ])

    assert exit_code == 2
    assert output_path.exists() is False
