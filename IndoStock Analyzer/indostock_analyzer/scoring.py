from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import CompositeIndicator, ScoreResult, SentimentResult


COMPONENT_WEIGHTS = {"sentiment": 0.2, "fundamental": 0.5, "technical": 0.3}


def clamp_score(value: float) -> float:
    return max(1.0, min(5.0, float(value)))


def exploratory_label(score: float | None) -> str:
    if score is None:
        return "unavailable"
    if score >= 4.2:
        return "upper exploratory range"
    if score >= 3.7:
        return "positive exploratory range"
    if score >= 3.0:
        return "mixed exploratory range"
    if score >= 2.3:
        return "negative exploratory range"
    return "lower exploratory range"


def score_sentiments(results: Sequence[SentimentResult]) -> ScoreResult:
    polarities = [item.polarity for item in results if item.supported and item.polarity is not None]
    if not polarities:
        return ScoreResult(None, "unavailable", ("No supported sentiment observations",), 0)
    average = sum(polarities) / len(polarities)
    score = clamp_score(3.0 + average * 2.0)
    return ScoreResult(score, exploratory_label(score), (f"Mean polarity {average:+.3f}",), len(polarities))


def score_fundamental(data: Mapping[str, object] | None) -> ScoreResult:
    if not data:
        return ScoreResult(None, "unavailable", ("Fundamental data unavailable",), 0)
    quality = data.get("data_quality")
    if isinstance(quality, Mapping) and quality.get("stale") is True:
        return ScoreResult(None, "unavailable", ("Fundamental data is stale",), 0)
    if isinstance(quality, Mapping) and quality.get("usable_for_scoring") is False:
        return ScoreResult(None, "unavailable", ("Fundamental freshness is unverified",), 0)
    score = 3.0
    reasons: list[str] = []
    available = 0

    pe = _number(data.get("pe_ratio"))
    if pe is not None:
        available += 1
        if pe < 10:
            score += 1
            reasons.append(f"P/E below heuristic threshold ({pe:.1f}x)")
        elif pe > 25:
            score -= 1
            reasons.append(f"P/E above heuristic threshold ({pe:.1f}x)")

    pb = _number(data.get("pb_ratio"))
    if pb is not None:
        available += 1
        if pb < 1:
            score += 1
            reasons.append(f"P/B below heuristic threshold ({pb:.2f}x)")
        elif pb > 3:
            score -= 1
            reasons.append(f"P/B above heuristic threshold ({pb:.2f}x)")

    roe = _number(data.get("roe"))
    if roe is not None:
        available += 1
        if roe > 0.15:
            score += 1
            reasons.append(f"ROE above heuristic threshold ({roe:.1%})")
        elif roe < 0.05:
            score -= 1
            reasons.append(f"ROE below heuristic threshold ({roe:.1%})")

    debt = _number(data.get("debt_to_equity"))
    if debt is not None:
        available += 1
        if debt < 0.5:
            score += 0.5
            reasons.append(f"Debt/equity below heuristic threshold ({debt:.2f})")
        elif debt > 2:
            score -= 0.5
            reasons.append(f"Debt/equity above heuristic threshold ({debt:.2f})")

    growth = _number(data.get("earnings_growth"))
    if growth is not None:
        available += 1
        if growth > 0.1:
            score += 1
            reasons.append(f"Earnings growth above heuristic threshold ({growth:.1%})")
        elif growth < -0.05:
            score -= 1
            reasons.append(f"Earnings growth below heuristic threshold ({growth:.1%})")

    if available == 0:
        return ScoreResult(None, "unavailable", ("No scorable fundamental metrics",), 0)
    bounded = clamp_score(score)
    return ScoreResult(bounded, exploratory_label(bounded), tuple(reasons), available)


def score_technical(data: Mapping[str, object] | None) -> ScoreResult:
    if not data:
        return ScoreResult(None, "unavailable", ("Technical data unavailable",), 0)
    quality = data.get("data_quality")
    if isinstance(quality, Mapping) and quality.get("stale") is True:
        return ScoreResult(None, "unavailable", ("Technical data is stale",), 0)
    if isinstance(quality, Mapping) and quality.get("usable_for_scoring") is False:
        return ScoreResult(None, "unavailable", ("Technical freshness is unverified",), 0)
    score = 3.0
    reasons: list[str] = []
    available = 0

    trend = data.get("ma_trend")
    if trend in {"BULLISH", "BEARISH", "NEUTRAL"}:
        available += 1
        if trend == "BULLISH":
            score += 1
            reasons.append("Short moving average is above long moving average")
        elif trend == "BEARISH":
            score -= 1
            reasons.append("Short moving average is below long moving average")

    rsi = _number(data.get("rsi"))
    if rsi is not None:
        available += 1
        if rsi < 30:
            score += 1
            reasons.append(f"RSI below heuristic threshold ({rsi:.1f})")
        elif rsi > 70:
            score -= 1
            reasons.append(f"RSI above heuristic threshold ({rsi:.1f})")

    macd = data.get("macd_signal")
    if macd in {"BULLISH", "BEARISH", "NEUTRAL"}:
        available += 1
        if macd == "BULLISH":
            score += 0.5
            reasons.append("MACD histogram is positive")
        elif macd == "BEARISH":
            score -= 0.5
            reasons.append("MACD histogram is negative")

    position = _number(data.get("price_position"))
    if position is not None:
        available += 1
        if position < 0.2:
            score += 0.5
            reasons.append("Price is near the observed range low")
        elif position > 0.8:
            score -= 0.5
            reasons.append("Price is near the observed range high")

    volume_surge = data.get("volume_surge")
    price_change = _number(data.get("price_change_1d"))
    if isinstance(volume_surge, bool) and price_change is not None:
        available += 1
        if volume_surge:
            score += 0.5 if price_change > 0 else -0.5
            reasons.append("Latest volume exceeds 1.5x the observed mean")

    if available == 0:
        return ScoreResult(None, "unavailable", ("No scorable technical metrics",), 0)
    bounded = clamp_score(score)
    return ScoreResult(bounded, exploratory_label(bounded), tuple(reasons), available)


def combine_scores(components: Mapping[str, float | None]) -> CompositeIndicator:
    numerator = 0.0
    coverage = 0.0
    normalized: dict[str, float | None] = {}
    for name, weight in COMPONENT_WEIGHTS.items():
        value = components.get(name)
        if value is None:
            normalized[name] = None
            continue
        bounded = clamp_score(value)
        normalized[name] = bounded
        numerator += bounded * weight
        coverage += weight
    score = clamp_score(numerator / coverage) if coverage else None
    return CompositeIndicator(score, exploratory_label(score), round(coverage, 4), normalized)


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
