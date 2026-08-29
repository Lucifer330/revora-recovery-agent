"""Structured ranker recommendation (LLM-compatible). No execution authority."""

from __future__ import annotations

from dataclasses import dataclass, field

from revora.schemas import RecoveryAction

DELAY_ALLOWLIST: frozenset[int] = frozenset({0, 1, 2, 6, 12, 24})


@dataclass(frozen=True)
class Recommendation:
    recommended_action: RecoveryAction
    ranked_actions: tuple[RecoveryAction, ...]
    suggested_delay_hours: int
    abstain: bool
    confidence: float
    reason_codes: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "recommended_action": self.recommended_action.value,
            "ranked_actions": [a.value for a in self.ranked_actions],
            "suggested_delay_hours": self.suggested_delay_hours,
            "abstain": self.abstain,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "rationale": self.rationale,
        }


class RecommendationError(ValueError):
    """Untrusted ranker output failed schema or eligibility checks."""
