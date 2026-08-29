"""Deterministic skip table. The skip decision is never made by the ranker/AI."""

from __future__ import annotations

from revora.schemas import FailureCategory

# NOT USED: skip ranker, use heuristic fallback.
# OPTIONAL: frozen AI adapter may run.
SKIP_AI_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.TRANSIENT_NETWORK,
        FailureCategory.REPEATED_FAILURE,
        FailureCategory.UNKNOWN,
    }
)


def should_skip_ai(category: FailureCategory) -> bool:
    return category in SKIP_AI_CATEGORIES
