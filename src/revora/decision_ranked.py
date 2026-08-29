"""C2 + ranker arms. Does not change Revora v1 or C2 decide_c2 scoring path.

Pipeline:
  diagnose → candidates → C2 eligibility → skip table (AI arm)
  → ranker → FINAL C2+safety gate → (orchestrator executes)
"""

from __future__ import annotations

from revora.decision import decide_c2
from revora.diagnosis import diagnose
from revora.knowledge import candidate_actions_for
from revora.policy import PolicyContext, c2_eligible_actions, first_allowed
from revora.rankers.frozen_ai import FrozenAIRanker
from revora.rankers.heuristic import HeuristicRanker
from revora.rankers.schema import Recommendation, RecommendationError
from revora.rankers.skip import should_skip_ai
from revora.rankers.validate import parse_recommendation
from revora.schemas import FailedPayment, RecoveryAction, SystemDecision
from revora.scoring import score_payment

SYSTEM_ID_HEURISTIC = "revora_c2_heuristic"
SYSTEM_ID_AI = "revora_c2_ai"

_HEURISTIC = HeuristicRanker()
_FROZEN_AI = FrozenAIRanker()


def _candidate_actions(payment: FailedPayment, category) -> tuple[RecoveryAction, ...]:
    kb = tuple(c.action for c in candidate_actions_for(category, payment.payment_method))
    if RecoveryAction.STOP_AND_ESCALATE not in kb:
        kb = kb + (RecoveryAction.STOP_AND_ESCALATE,)
    return kb


def _gated(
    payment: FailedPayment,
    ctx: PolicyContext,
    ranked: list[RecoveryAction],
    *,
    system_id: str,
    ranker_id: str,
    ai_status: str,
    notes: tuple[str, ...],
    agreed_h: bool | None,
    agreed_c2: bool | None,
) -> SystemDecision:
    diagnosis = diagnose(payment)
    candidates = candidate_actions_for(diagnosis.category, payment.payment_method)
    score = score_payment(payment, diagnosis.category)
    policy = first_allowed(
        payment,
        diagnosis.category,
        ranked,
        ctx,
        revora_rules=True,
        c2_ledger=True,
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
        ranker_id=ranker_id,
        ai_status=ai_status,
        agreed_with_heuristic=agreed_h,
        agreed_with_c2=agreed_c2,
    )


def decide_c2_heuristic(payment: FailedPayment, ctx: PolicyContext) -> SystemDecision:
    diagnosis = diagnose(payment)
    pool = _candidate_actions(payment, diagnosis.category)
    eligible = c2_eligible_actions(payment, diagnosis.category, pool, ctx)
    rec = _HEURISTIC.rank(payment, diagnosis.category, eligible, ctx)
    notes = (
        "C2_HEURISTIC",
        rec.rationale,
        f"eligible={[a.value for a in eligible]}",
    )
    return _gated(
        payment,
        ctx,
        list(rec.ranked_actions),
        system_id=SYSTEM_ID_HEURISTIC,
        ranker_id=_HEURISTIC.ranker_id,
        ai_status="n/a",
        notes=notes,
        agreed_h=None,
        agreed_c2=None,
    )


def decide_c2_ai(payment: FailedPayment, ctx: PolicyContext) -> SystemDecision:
    diagnosis = diagnose(payment)
    pool = _candidate_actions(payment, diagnosis.category)
    eligible = c2_eligible_actions(payment, diagnosis.category, pool, ctx)
    heuristic_rec = _HEURISTIC.rank(payment, diagnosis.category, eligible, ctx)
    c2_choice = decide_c2(payment, ctx).action

    ai_status = "used"
    ranker_id = _FROZEN_AI.ranker_id
    rec: Recommendation = heuristic_rec

    if should_skip_ai(diagnosis.category):
        ai_status = "skipped"
        ranker_id = _HEURISTIC.ranker_id
        rec = heuristic_rec
    else:
        try:
            rec = _FROZEN_AI.rank(payment, diagnosis.category, eligible, ctx)
            if rec.abstain:
                ai_status = "fallback"
                rec = heuristic_rec
                ranker_id = _HEURISTIC.ranker_id
        except RecommendationError:
            ai_status = "invalid"
            rec = heuristic_rec
            ranker_id = _HEURISTIC.ranker_id

    notes = (
        f"C2_AI status={ai_status}",
        rec.rationale,
        f"eligible={[a.value for a in eligible]}",
    )
    return _gated(
        payment,
        ctx,
        list(rec.ranked_actions),
        system_id=SYSTEM_ID_AI,
        ranker_id=ranker_id,
        ai_status=ai_status,
        notes=notes,
        agreed_h=rec.recommended_action == heuristic_rec.recommended_action,
        agreed_c2=rec.recommended_action == c2_choice,
    )


def apply_untrusted_ai_payload(
    payment: FailedPayment,
    ctx: PolicyContext,
    raw: object,
) -> SystemDecision:
    """Test helper: validate untrusted output then gate. Fallback on error."""
    diagnosis = diagnose(payment)
    pool = _candidate_actions(payment, diagnosis.category)
    eligible = c2_eligible_actions(payment, diagnosis.category, pool, ctx)
    heuristic_rec = _HEURISTIC.rank(payment, diagnosis.category, eligible, ctx)
    try:
        rec = parse_recommendation(raw, eligible)
        status = "used"
        if rec.abstain:
            rec = heuristic_rec
            status = "fallback"
    except RecommendationError:
        rec = heuristic_rec
        status = "invalid"
    return _gated(
        payment,
        ctx,
        list(rec.ranked_actions),
        system_id=SYSTEM_ID_AI,
        ranker_id="untrusted",
        ai_status=status,
        notes=(f"untrusted status={status}",),
        agreed_h=rec.recommended_action == heuristic_rec.recommended_action,
        agreed_c2=None,
    )
