"""Validate untrusted ranker JSON. Never executes a payment action."""

from __future__ import annotations

from revora.rankers.schema import DELAY_ALLOWLIST, Recommendation, RecommendationError
from revora.schemas import RecoveryAction


def parse_recommendation(
    raw: object,
    eligible: tuple[RecoveryAction, ...] | list[RecoveryAction],
) -> Recommendation:
    eligible_set = set(eligible)
    if not eligible_set:
        raise RecommendationError("No eligible actions")
    if not isinstance(raw, dict):
        raise RecommendationError("Recommendation must be an object")

    if raw.get("abstain") is True:
        stop = RecoveryAction.STOP_AND_ESCALATE
        if stop not in eligible_set:
            raise RecommendationError("Abstain requires STOP_AND_ESCALATE to be eligible")
        delay = raw.get("suggested_delay_hours", 0)
        if delay not in DELAY_ALLOWLIST:
            raise RecommendationError("Invalid suggested_delay_hours")
        return Recommendation(
            recommended_action=stop,
            ranked_actions=(stop,),
            suggested_delay_hours=int(delay),
            abstain=True,
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            reason_codes=tuple(str(x) for x in raw.get("reason_codes", ()) or ()),
            rationale=str(raw.get("rationale", "abstain"))[:240],
        )

    try:
        rec_action = RecoveryAction(str(raw.get("recommended_action")))
    except ValueError as exc:
        raise RecommendationError("Unknown recommended_action") from exc
    if rec_action not in eligible_set:
        raise RecommendationError("recommended_action is not C2-eligible")

    ranked_raw = raw.get("ranked_actions", [rec_action.value])
    if not isinstance(ranked_raw, list) or not ranked_raw:
        raise RecommendationError("ranked_actions must be a non-empty list")
    ranked: list[RecoveryAction] = []
    for item in ranked_raw:
        try:
            action = RecoveryAction(str(item))
        except ValueError as exc:
            raise RecommendationError("Unknown action in ranked_actions") from exc
        if action not in eligible_set:
            raise RecommendationError("ranked_actions contains an ineligible action")
        if action not in ranked:
            ranked.append(action)
    if rec_action not in ranked:
        ranked.insert(0, rec_action)

    delay = raw.get("suggested_delay_hours", 0)
    if delay not in DELAY_ALLOWLIST:
        raise RecommendationError("Invalid suggested_delay_hours")

    conf = raw.get("confidence", 0.0)
    try:
        confidence = float(conf)
    except (TypeError, ValueError) as exc:
        raise RecommendationError("Invalid confidence") from exc
    if not 0.0 <= confidence <= 1.0:
        raise RecommendationError("confidence must be in [0, 1]")

    rationale = str(raw.get("rationale", ""))[:240]
    codes = tuple(str(x) for x in raw.get("reason_codes", ()) or ())
    return Recommendation(
        recommended_action=rec_action,
        ranked_actions=tuple(ranked),
        suggested_delay_hours=int(delay),
        abstain=False,
        confidence=confidence,
        reason_codes=codes,
        rationale=rationale,
    )
