"""Reproducible Failure Lab scenarios.

Each scenario asserts safe control-plane behavior, not recovery volume.
"""

from __future__ import annotations

from dataclasses import dataclass

from revora.baselines import decide_simple_retry
from revora.decision import decide as decide_revora
from revora.diagnosis import DiagnosisError, diagnose
from revora.orchestrator import policy_context, run_cycle
from revora.policy import PolicyContext, evaluate_policy
from revora.schemas import (
    CustomerHistory,
    FailedPayment,
    FailureCategory,
    PaymentMethodType,
    PaymentState,
    RecoveryAction,
    Split,
)
from revora.simulator import InjectedFault, RecoverySimulator


@dataclass(frozen=True)
class LabResult:
    name: str
    passed: bool
    detail: str


def _payment(**overrides: object) -> FailedPayment:
    base = dict(
        payment_id="pay_lab_001",
        event_id="evt_lab_001",
        amount_paise=49900,
        currency="INR",
        failure_code="gateway_timeout",
        payment_method=PaymentMethodType.CARD,
        attempt_count=1,
        previous_recovery_attempts=0,
        hours_since_failure=1.0,
        customer=CustomerHistory(5, 1, 12),
        created_at_epoch_s=1_700_000_000,
        merchant_id="merch_lab",
        split=Split.DEV,
    )
    base.update(overrides)
    return FailedPayment(**base)  # type: ignore[arg-type]


def scenario_duplicate_webhook() -> LabResult:
    payment = _payment()
    sim = RecoverySimulator()
    _d1, a1 = run_cycle(payment, decide_revora, sim)
    _d2, a2 = run_cycle(payment, decide_revora, sim, replay_same_event=True)
    executed_twice = a1.executed and a2.executed
    passed = a1.executed and not a2.executed and "Duplicate" in (a2.policy_reason + (a2.execution_error or ""))
    return LabResult(
        "duplicate_webhook",
        passed and not executed_twice,
        f"first_executed={a1.executed} second_executed={a2.executed} reason={a2.policy_reason}",
    )


def scenario_recovery_api_timeout() -> LabResult:
    payment = _payment()
    sim = RecoverySimulator()
    _d, audit = run_cycle(
        payment,
        decide_revora,
        sim,
        fault=InjectedFault(timeout_on_execute=True),
    )
    state = sim.get(payment.payment_id).state
    passed = (
        audit.execution_timed_out
        and not audit.recorded_recovered
        and state != PaymentState.RECOVERED
    )
    return LabResult(
        "recovery_api_timeout",
        passed,
        f"timed_out={audit.execution_timed_out} recovered={audit.recorded_recovered} state={state.value}",
    )


def scenario_verification_timeout() -> LabResult:
    payment = _payment()
    sim = RecoverySimulator()
    _d, audit = run_cycle(
        payment,
        decide_revora,
        sim,
        fault=InjectedFault(timeout_on_verify=True),
    )
    passed = (
        not audit.recorded_recovered
        and audit.verified_state == PaymentState.UNKNOWN
    )
    return LabResult(
        "verification_timeout",
        passed,
        f"recovered={audit.recorded_recovered} verified_state={audit.verified_state.value if audit.verified_state else None}",
    )


def scenario_invalid_recovery_action() -> LabResult:
    payment = _payment()
    sim = RecoverySimulator()
    ctx = policy_context(sim, payment)
    # Force an action string that the gate rejects via invalid path in simulator.
    _d, audit = run_cycle(
        payment,
        decide_simple_retry,
        sim,
        fault=InjectedFault(force_invalid_action=True),
    )
    _ = ctx
    passed = not audit.executed and not audit.recorded_recovered
    return LabResult(
        "invalid_recovery_action",
        passed,
        f"executed={audit.executed} error={audit.execution_error}",
    )


def scenario_max_retry_reached() -> LabResult:
    payment = _payment(
        failure_code="gateway_timeout",
        previous_recovery_attempts=4,
        attempt_count=6,
    )
    diagnosis = diagnose(payment)
    decision = evaluate_policy(
        payment,
        diagnosis.category,
        RecoveryAction.RETRY_LATER,
        PolicyContext(0, 999.0, None, False),
    )
    passed = (not decision.allowed) and decision.rule_id in {
        "retry_limit",
        "stopping_rule",
        "global_attempt_cap",
        "forbidden_pair",
    }
    # Repeated-failure override may also forbid retry.
    return LabResult(
        "max_retry_reached",
        passed,
        f"allowed={decision.allowed} rule={decision.rule_id} reason={decision.reason} category={diagnosis.category.value}",
    )


def scenario_missing_invalid_context() -> LabResult:
    missing_ok = False
    try:
        diagnose(None)
    except DiagnosisError:
        missing_ok = True
    invalid = _payment(amount_paise=0)
    invalid_ok = False
    try:
        diagnose(invalid)
    except DiagnosisError:
        invalid_ok = True
    gate = evaluate_policy(
        None,
        FailureCategory.UNKNOWN,
        RecoveryAction.RETRY_LATER,
        PolicyContext(0, 0, None, False),
    )
    gate_ok = not gate.allowed and gate.rule_id == "require_context"
    passed = missing_ok and invalid_ok and gate_ok
    return LabResult(
        "missing_invalid_payment_context",
        passed,
        f"diagnose_none={missing_ok} diagnose_zero_amount={invalid_ok} policy={gate.rule_id}",
    )


SCENARIOS = (
    scenario_duplicate_webhook,
    scenario_recovery_api_timeout,
    scenario_verification_timeout,
    scenario_invalid_recovery_action,
    scenario_max_retry_reached,
    scenario_missing_invalid_context,
)


def run_failure_lab() -> list[LabResult]:
    return [fn() for fn in SCENARIOS]


def render_lab(results: list[LabResult]) -> str:
    lines = ["======== REVORA FAILURE LAB ========", ""]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"[{mark}] {r.name}: {r.detail}")
    lines.append("")
    lines.append(f"{sum(1 for r in results if r.passed)}/{len(results)} scenarios passed")
    return "\n".join(lines)
