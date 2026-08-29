"""Orchestrate recovery: decide → safety/policy → execute → verify → audit."""

from __future__ import annotations

from collections.abc import Callable

from revora.actions import AUTOMATED_ACTIONS
from revora.environment import resolution_hours
from revora.policy import SHARED_RETRY_COOLDOWN_HOURS, PolicyContext
from revora.schemas import (
    AuditRecord,
    FailedPayment,
    PaymentState,
    RecoveryAction,
    SystemDecision,
)
from revora.simulator import InjectedFault, RecoverySimulator
from revora.verification import verify_outcome

DecideFn = Callable[[FailedPayment, PolicyContext], SystemDecision]


def policy_context(simulator: RecoverySimulator, payment: FailedPayment) -> PolicyContext:
    sim = simulator.seed_failed(payment)
    return PolicyContext(
        recovery_attempts_this_run=sim.recovery_attempts,
        hours_since_last_recovery=sim.hours_since_last_recovery,
        last_action=sim.last_action,
        event_already_processed=payment.event_id in sim.processed_event_ids,
        attempts_by_family=tuple(sorted(sim.attempts_by_family.items())),
    )


def _wait_out_cooldown(
    payment: FailedPayment,
    decide: DecideFn,
    simulator: RecoverySimulator,
) -> FailedPayment:
    """If the strategy is blocked only by cooldown, advance the clock and retry later.

    Waiting is charged equally to every strategy. It does not skip the action.
    """
    ctx = policy_context(simulator, payment)
    peek = decide(payment, ctx)
    if peek.policy.rule_id != "cooldown":
        return payment
    sim = simulator.get(payment.payment_id)
    wait = SHARED_RETRY_COOLDOWN_HOURS - sim.hours_since_last_recovery
    if wait <= 0:
        return payment
    simulator.advance_clock(payment.payment_id, wait)
    return payment.evolve(hours_since_failure=payment.hours_since_failure + wait)


def run_cycle(
    payment: FailedPayment,
    decide: DecideFn,
    simulator: RecoverySimulator,
    *,
    fault: InjectedFault | None = None,
    replay_same_event: bool = False,
) -> tuple[SystemDecision, AuditRecord]:
    ctx = policy_context(simulator, payment)
    decision = decide(payment, ctx)

    duplicate_before = ctx.event_already_processed
    executed = False
    timed_out = False
    exec_error: str | None = None
    recorded_recovered = False
    verified_state: PaymentState | None = None
    notes = decision.policy.reason

    if not decision.allowed:
        exec_error = decision.policy.reason
        notes = decision.policy.reason
    else:
        result = simulator.execute(
            payment,
            decision.diagnosis.category,
            decision.action,
            allowed=decision.allowed,
            fault=fault,
        )
        executed = result.accepted
        timed_out = result.timed_out
        exec_error = result.error
        verification = verify_outcome(
            simulator,
            payment.payment_id,
            execution_accepted=result.accepted,
            fault=fault,
        )
        verified_state = verification.verified_state
        recorded_recovered = verification.recovered
        notes = verification.reason

        if result.accepted and verification.verified_state != PaymentState.RECOVERED:
            recorded_recovered = False

    audit = AuditRecord(
        system_id=decision.system_id,
        payment_id=payment.payment_id,
        event_id=payment.event_id,
        action=decision.action,
        allowed=decision.allowed,
        policy_reason=decision.policy.reason,
        executed=executed,
        execution_timed_out=timed_out,
        execution_error=exec_error,
        verified_state=verified_state,
        recorded_recovered=recorded_recovered,
        duplicate_suppressed=duplicate_before or replay_same_event and not executed,
        notes=notes,
        ranker_id=decision.ranker_id,
        ai_status=decision.ai_status,
        agreed_with_heuristic=decision.agreed_with_heuristic,
        agreed_with_c2=decision.agreed_with_c2,
    )
    return decision, audit


def run_until_terminal(
    payment: FailedPayment,
    decide: DecideFn,
    simulator: RecoverySimulator,
    *,
    max_cycles: int = 4,
) -> list[AuditRecord]:
    """Original webhook, then follow-ups with new event IDs and real clock time."""
    records: list[AuditRecord] = []
    current = payment
    for cycle in range(max_cycles):
        current = _wait_out_cooldown(current, decide, simulator)
        decision, audit = run_cycle(current, decide, simulator)
        records.append(audit)
        sim = simulator.get(payment.payment_id)
        if sim.state in {PaymentState.RECOVERED, PaymentState.ESCALATED}:
            break
        if decision.action == RecoveryAction.STOP_AND_ESCALATE:
            break
        if not audit.executed:
            break
        if decision.action not in AUTOMATED_ACTIONS:
            break
        hours = resolution_hours(decision.action)
        simulator.advance_clock(payment.payment_id, hours)
        current = current.evolve(
            event_id=f"{payment.event_id}_followup_{cycle + 1}",
            attempt_count=current.attempt_count + 1,
            previous_recovery_attempts=payment.previous_recovery_attempts + sim.recovery_attempts,
            hours_since_failure=current.hours_since_failure + hours,
        )
    return records
