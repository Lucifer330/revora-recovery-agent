"""Single-payment decision trace for demo (held-out ids, not synthetic stories)."""

from __future__ import annotations

from revora.dataset import generate_dataset
from revora.decision import decide, decide_c2
from revora.decision_ranked import (
    _HEURISTIC,
    _FROZEN_AI,
    _candidate_actions,
    decide_c2_ai,
)
from revora.diagnosis import diagnose
from revora.orchestrator import run_cycle
from revora.policy import (
    PolicyContext,
    c2_eligible_actions,
    evaluate_revora_c2_policy,
    family_attempts_for_c2,
)
from revora.rankers.skip import should_skip_ai
from revora.schemas import FailedPayment, RecoveryAction, Split
from revora.simulator import RecoverySimulator


def find_payment(payment_id: str, *, size: int = 400, seed: int = 42) -> FailedPayment:
    ds = generate_dataset(size=size, seed=seed)
    for p in ds.payments:
        if p.payment_id == payment_id:
            return p
    raise SystemExit(f"Unknown payment_id {payment_id} in dataset seed={seed} size={size}")


def render_trace(payment: FailedPayment) -> str:
    ctx = PolicyContext(0, 999.0, None, False, ())
    d = diagnose(payment)
    pool = _candidate_actions(payment, d.category)
    eligible = c2_eligible_actions(payment, d.category, pool, ctx)
    skip = should_skip_ai(d.category)
    skip_why = {
        True: {
            "UNKNOWN": "undiagnosed cause: skip AI (uncertainty), use heuristic",
            "TRANSIENT_NETWORK": "no residual decision: skip AI",
            "REPEATED_FAILURE": "exhaustion: skip AI, prefer stop/escalate",
        }.get(d.category.value, "skip table"),
        False: "optional category: frozen AI adapter may rank eligible actions",
    }[skip]
    hrec = _HEURISTIC.rank(payment, d.category, eligible, ctx)
    if skip:
        rec = hrec
        router = "SKIPPED"
    else:
        rec = _FROZEN_AI.rank(payment, d.category, eligible, ctx)
        router = "INVOKED (frozen adapter, no live LLM)"
    gate = evaluate_revora_c2_policy(payment, d.category, rec.recommended_action, ctx)
    v1 = decide(payment, ctx)
    c2 = decide_c2(payment, ctx)
    ai = decide_c2_ai(payment, ctx)
    sim = RecoverySimulator()
    _dec, audit = run_cycle(payment, decide_c2_ai, sim)
    families = (
        RecoveryAction.RETRY_LATER,
        RecoveryAction.PAYMENT_METHOD_UPDATE,
        RecoveryAction.RECOVERY_NUDGE,
        RecoveryAction.ALTERNATIVE_PATH,
    )
    ledger_lines = [
        f"    {a.value}: this_episode={dict(ctx.attempts_by_family).get(a.value, 0)} "
        f"c2_family_total={family_attempts_for_c2(payment, a, ctx)}"
        + ("  [dataset previous_recovery_attempts counted here]" if a == RecoveryAction.RETRY_LATER else "")
        for a in families
    ]
    lines = [
        "======== REVORA PAYMENT TRACE ========",
        f"payment_id={payment.payment_id}  event_id={payment.event_id}  split={payment.split.value}",
        "",
        "1. Incoming event + diagnosed cause",
        f"   failure_code={payment.failure_code}  method={payment.payment_method.value}",
        f"   amount_paise={payment.amount_paise}  attempt_count={payment.attempt_count}",
        f"   hours_since_failure={payment.hours_since_failure}",
        f"   diagnosis={d.category.value}  reasons={list(d.reasons)}",
        "",
        "2. Ledger (mechanism-aware, not pooled)",
        f"   unlabeled previous_recovery_attempts={payment.previous_recovery_attempts} (charged to RETRY_LATER family only)",
        *ledger_lines,
        "",
        "3. Policy-eligible action set (C2 eligibility, before safety)",
        f"   { [a.value for a in eligible] }",
        "",
        "4. Router (skip table is not chosen by the ranker)",
        f"   AI: {router}",
        f"   why: {skip_why}",
        f"   decide_c2_ai.ai_status={ai.ai_status}",
        "",
        "5. Ranked / chosen action + reasoning",
        f"   v1: {v1.action.value}",
        f"   C2: {c2.action.value}",
        f"   heuristic top: {hrec.recommended_action.value}  {hrec.rationale}",
        f"   ranker used top: {rec.recommended_action.value}  codes={list(rec.reason_codes)}",
        f"   ranked: {[a.value for a in rec.ranked_actions]}",
        "",
        "6. Safety + C2 gate on ranker top pick",
        f"   allowed={gate.allowed}  rule_id={gate.rule_id}",
        f"   reason={gate.reason}",
        f"   final decide_c2_ai action={ai.action.value}  allowed={ai.allowed}",
        "",
        "7. Verified outcome + audit",
        f"   executed={audit.executed}  verified_state={audit.verified_state}",
        f"   recorded_recovered={audit.recorded_recovered}  (execution success is not recovery)",
        f"   audit: payment_id={audit.payment_id} event_id={audit.event_id} "
        f"system={audit.system_id} action={audit.action.value} ai_status={audit.ai_status}",
        "======================================",
    ]
    return "\n".join(lines)
