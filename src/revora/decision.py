"""Revora decision engine: recommend, then pass through the policy gate.

Pipeline:
  diagnosis -> knowledge lookup -> candidates -> recoverability score
  -> rank by heuristic EV -> policy gate -> allowed action or reject.
"""

from __future__ import annotations

from revora.diagnosis import diagnose
from revora.knowledge import candidate_actions_for
from revora.policy import PolicyContext, first_allowed
from revora.schemas import FailedPayment, RecoveryAction, SystemDecision
from revora.scoring import rank_candidates, score_payment


SYSTEM_ID = "revora"
SYSTEM_ID_C2 = "revora_c2"


def decide(payment: FailedPayment, ctx: PolicyContext) -> SystemDecision:
    return _decide(payment, ctx, system_id=SYSTEM_ID, c2_ledger=False)


def decide_c2(payment: FailedPayment, ctx: PolicyContext) -> SystemDecision:
    """Revora C2: same ranking as v1, mechanism-aware stop ledger. Not AI."""
    return _decide(payment, ctx, system_id=SYSTEM_ID_C2, c2_ledger=True)


def _decide(
    payment: FailedPayment,
    ctx: PolicyContext,
    *,
    system_id: str,
    c2_ledger: bool,
) -> SystemDecision:
    diagnosis = diagnose(payment)
    candidates = candidate_actions_for(diagnosis.category, payment.payment_method)
    score = score_payment(payment, diagnosis.category)
    ranked = rank_candidates(
        payment,
        diagnosis.category,
        score,
        tuple(c.action for c in candidates),
    )
    notes = tuple(f"{action.value}: {note}" for action, _ev, note in ranked)
    ranked_actions = [action for action, _ev, _note in ranked]
    if RecoveryAction.STOP_AND_ESCALATE not in ranked_actions:
        ranked_actions.append(RecoveryAction.STOP_AND_ESCALATE)

    policy = first_allowed(
        payment,
        diagnosis.category,
        ranked_actions,
        ctx,
        revora_rules=True,
        c2_ledger=c2_ledger,
    )
    return SystemDecision(
        system_id=system_id,
        action=policy.action,
        allowed=policy.allowed,
        policy=policy,
        diagnosis=diagnosis,
        candidates=candidates,
        score=score,
        ranking_notes=notes,
    )
