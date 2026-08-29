"""Curated recovery-policy knowledge base with deterministic lookup.

This is not RAG and not a vector index. Lookup is an exact match on
diagnosed failure category (plus optional method-type filter).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from revora.schemas import (
    CandidateAction,
    FailureCategory,
    PaymentMethodType,
    RecoveryAction,
)

_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "data" / "recovery_policies.json"


@dataclass(frozen=True)
class KnowledgeEntry:
    id: str
    failure_categories: tuple[FailureCategory, ...]
    payment_methods: tuple[PaymentMethodType, ...] | None
    candidate_actions: tuple[RecoveryAction, ...]
    max_retries: int
    cooldown_hours: float
    rationale: str
    stop_after_recovery_attempts: int


def load_knowledge(path: Path | None = None) -> list[KnowledgeEntry]:
    raw = json.loads((path or _KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    entries: list[KnowledgeEntry] = []
    for item in raw["policies"]:
        methods = item.get("payment_methods")
        entries.append(
            KnowledgeEntry(
                id=item["id"],
                failure_categories=tuple(
                    FailureCategory(c) for c in item["failure_categories"]
                ),
                payment_methods=(
                    tuple(PaymentMethodType(m) for m in methods) if methods else None
                ),
                candidate_actions=tuple(
                    RecoveryAction(a) for a in item["candidate_actions"]
                ),
                max_retries=int(item["max_retries"]),
                cooldown_hours=float(item["cooldown_hours"]),
                rationale=item["rationale"],
                stop_after_recovery_attempts=int(item["stop_after_recovery_attempts"]),
            )
        )
    return entries


@lru_cache(maxsize=1)
def default_knowledge() -> tuple[KnowledgeEntry, ...]:
    return tuple(load_knowledge())


def lookup_policies(
    category: FailureCategory,
    method: PaymentMethodType,
    knowledge: tuple[KnowledgeEntry, ...] | None = None,
) -> list[KnowledgeEntry]:
    kb = knowledge or default_knowledge()
    matched = [
        e
        for e in kb
        if category in e.failure_categories
        and (e.payment_methods is None or method in e.payment_methods)
    ]
    matched.sort(key=lambda e: e.id)
    return matched


def candidate_actions_for(
    category: FailureCategory,
    method: PaymentMethodType,
    knowledge: tuple[KnowledgeEntry, ...] | None = None,
) -> tuple[CandidateAction, ...]:
    seen: set[RecoveryAction] = set()
    out: list[CandidateAction] = []
    for entry in lookup_policies(category, method, knowledge):
        for action in entry.candidate_actions:
            if action in seen:
                continue
            seen.add(action)
            out.append(
                CandidateAction(
                    action=action,
                    rationale=entry.rationale,
                    knowledge_id=entry.id,
                )
            )
    if not out:
        out.append(
            CandidateAction(
                action=RecoveryAction.STOP_AND_ESCALATE,
                rationale="No curated policy matched; default to stop.",
                knowledge_id="fallback_stop",
            )
        )
    return tuple(out)
