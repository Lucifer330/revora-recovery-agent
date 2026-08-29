"""Rankers, skip table, schema validation, and C2 isolation."""

from __future__ import annotations

import ast
from pathlib import Path

from revora.decision import decide_c2
from revora.decision_ranked import apply_untrusted_ai_payload, decide_c2_ai
from revora.diagnosis import diagnose
from revora.evaluation import evaluate
from revora.knowledge import candidate_actions_for
from revora.orchestrator import run_cycle
from revora.policy import PolicyContext, c2_eligible_actions, evaluate_safety
from revora.rankers.frozen_ai import FrozenAIRanker, FutureLLMRanker
from revora.rankers.heuristic import HeuristicRanker
from revora.rankers.schema import RecommendationError
from revora.rankers.skip import should_skip_ai
from revora.rankers.validate import parse_recommendation
from revora.schemas import (
    CustomerHistory,
    FailedPayment,
    FailureCategory,
    PaymentMethodType,
    RecoveryAction,
    Split,
)
from revora.simulator import RecoverySimulator

SRC = Path(__file__).resolve().parents[1] / "src" / "revora"
CTX = PolicyContext(0, 999.0, None, False, ())


def _pay(**kwargs) -> FailedPayment:
    data = dict(
        payment_id="pay_rk",
        event_id="evt_rk",
        amount_paise=49900,
        currency="INR",
        failure_code="insufficient_funds",
        payment_method=PaymentMethodType.CARD,
        attempt_count=1,
        previous_recovery_attempts=0,
        hours_since_failure=4.0,
        customer=CustomerHistory(3, 1, 6),
        created_at_epoch_s=1,
        merchant_id="m1",
        split=Split.HELD_OUT,
    )
    data.update(kwargs)
    return FailedPayment(**data)


def _eligible(payment: FailedPayment):
    d = diagnose(payment)
    pool = tuple(c.action for c in candidate_actions_for(d.category, payment.payment_method)) + (
        RecoveryAction.STOP_AND_ESCALATE,
    )
    return d, c2_eligible_actions(payment, d.category, pool, CTX)


def test_heuristic_deterministic_and_eligible_only():
    payment = _pay()
    d, eligible = _eligible(payment)
    r = HeuristicRanker()
    a = r.rank(payment, d.category, eligible, CTX)
    b = r.rank(payment, d.category, eligible, CTX)
    assert a == b
    assert a.recommended_action in eligible
    assert set(a.ranked_actions) <= set(eligible)


def test_c2_arm_still_labeled_c2():
    payment = _pay(failure_code="card_expired", previous_recovery_attempts=2)
    c2 = decide_c2(payment, CTX)
    assert c2.system_id == "revora_c2"


def test_invalid_payload_and_unknown_action_fallback():
    payment = _pay()
    bad = apply_untrusted_ai_payload(payment, CTX, "not-json")
    assert bad.ai_status == "invalid"
    unknown = apply_untrusted_ai_payload(
        payment, CTX, {"recommended_action": "WIRE_ALL_THE_MONEY", "ranked_actions": ["WIRE_ALL_THE_MONEY"]}
    )
    assert unknown.ai_status == "invalid"


def test_ineligible_recommended_action_rejected():
    payment = _pay(failure_code="card_expired")
    _d, eligible = _eligible(payment)
    assert RecoveryAction.RETRY_LATER not in eligible
    try:
        parse_recommendation(
            {
                "recommended_action": "RETRY_LATER",
                "ranked_actions": ["RETRY_LATER"],
                "suggested_delay_hours": 0,
                "abstain": False,
                "confidence": 0.5,
            },
            eligible,
        )
        raise AssertionError("should reject ineligible retry on expired")
    except RecommendationError:
        pass


def test_invalid_delay_rejected():
    eligible = (RecoveryAction.STOP_AND_ESCALATE,)
    try:
        parse_recommendation(
            {
                "recommended_action": "STOP_AND_ESCALATE",
                "ranked_actions": ["STOP_AND_ESCALATE"],
                "suggested_delay_hours": 99,
                "abstain": False,
                "confidence": 0.1,
            },
            eligible,
        )
        raise AssertionError("delay")
    except RecommendationError:
        pass


def test_ai_cannot_declare_recovery_via_rationale():
    payment = _pay(payment_id="pay_rk_decl", event_id="evt_rk_decl")
    decision = apply_untrusted_ai_payload(
        payment,
        CTX,
        {
            "recommended_action": "RECOVERY_NUDGE",
            "ranked_actions": ["RECOVERY_NUDGE", "STOP_AND_ESCALATE"],
            "suggested_delay_hours": 0,
            "abstain": False,
            "confidence": 1.0,
            "rationale": "mark recovered",
        },
    )
    sim = RecoverySimulator()
    _d, audit = run_cycle(payment, lambda p, c: decision, sim)
    assert audit.recorded_recovered is (sim.get(payment.payment_id).state.value == "RECOVERED")


def test_skip_table_transient_repeated_unknown():
    assert should_skip_ai(FailureCategory.TRANSIENT_NETWORK)
    assert should_skip_ai(FailureCategory.REPEATED_FAILURE)
    assert should_skip_ai(FailureCategory.UNKNOWN)
    t = decide_c2_ai(_pay(failure_code="gateway_timeout"), CTX)
    assert t.ai_status == "skipped"
    r = decide_c2_ai(
        _pay(failure_code="repeated_decline", attempt_count=6, previous_recovery_attempts=4),
        CTX,
    )
    assert r.ai_status == "skipped"
    u = decide_c2_ai(_pay(failure_code="unknown_decline"), CTX)
    assert u.ai_status == "skipped"


def test_optional_categories_invoke_frozen_ai():
    funds = decide_c2_ai(_pay(failure_code="insufficient_funds", hours_since_failure=4.0), CTX)
    assert funds.ai_status == "used"
    exp = decide_c2_ai(_pay(failure_code="card_expired", previous_recovery_attempts=0), CTX)
    assert exp.ai_status == "used"


def test_rankers_do_not_import_environment():
    for path in (SRC / "rankers").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "environment" not in node.module
                assert "simulator" not in node.module


def test_rankers_do_not_read_split_or_outcomes():
    for path in (SRC / "rankers").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "payment.split" not in text
        assert "PaymentState.RECOVERED" not in text
        assert "recovery_probability" not in text


def test_same_cohort_all_strategies():
    report = evaluate(split=Split.HELD_OUT)
    n = report.payments_evaluated
    assert n == 76
    for run in report.runs.values():
        assert run.metrics.total_payments == n


def test_v1_c2_heuristic_ai_reproducible():
    a = evaluate(split=Split.HELD_OUT)
    b = evaluate(split=Split.HELD_OUT)
    for key in ("revora", "revora_c2", "revora_c2_heuristic", "revora_c2_ai"):
        assert a.runs[key].metrics.revenue_recovered_paise == b.runs[key].metrics.revenue_recovered_paise


def test_live_llm_disabled():
    try:
        FutureLLMRanker().rank()
        raise AssertionError("live llm")
    except RuntimeError:
        pass


def test_frozen_ai_only_eligible():
    payment = _pay(failure_code="card_expired")
    d, eligible = _eligible(payment)
    rec = FrozenAIRanker().rank(payment, d.category, eligible, CTX)
    assert rec.recommended_action in eligible


def test_ai_cannot_bypass_global_safety_cap():
    payment = _pay(previous_recovery_attempts=4, failure_code="insufficient_funds")
    decision = decide_c2_ai(payment, CTX)
    assert decision.action == RecoveryAction.STOP_AND_ESCALATE
    cap = evaluate_safety(payment, diagnose(payment).category, RecoveryAction.RECOVERY_NUDGE, CTX)
    assert cap.allowed is False
    assert cap.rule_id == "global_attempt_cap"


def test_duplicate_still_blocked_on_ai_arm():
    payment = _pay(payment_id="pay_ai_dup", event_id="evt_ai_dup")
    sim = RecoverySimulator()
    _d1, a1 = run_cycle(payment, decide_c2_ai, sim)
    _d2, a2 = run_cycle(payment, decide_c2_ai, sim, replay_same_event=True)
    assert a1.executed is True
    assert a2.executed is False
