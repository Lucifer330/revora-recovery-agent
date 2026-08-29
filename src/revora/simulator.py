"""Recovery execution simulator with explicit payment states.

Decision engines must not import this module or ``revora.environment``.
Outcomes come from the independent environment + a strategy-free hash draw.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from revora.environment import recovery_probability
from revora.schemas import (
    ExecutionResult,
    FailedPayment,
    FailureCategory,
    PaymentState,
    RecoveryAction,
    SimulatedPayment,
)


def _unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


@dataclass
class InjectedFault:
    """Optional Failure Lab fault. Empty in the main evaluation."""

    timeout_on_execute: bool = False
    timeout_on_verify: bool = False
    force_invalid_action: bool = False


class RecoverySimulator:
    def __init__(self) -> None:
        self._payments: dict[str, SimulatedPayment] = {}

    def seed_failed(self, payment: FailedPayment) -> SimulatedPayment:
        existing = self._payments.get(payment.payment_id)
        if existing is None:
            existing = SimulatedPayment(
                payment_id=payment.payment_id,
                amount_paise=payment.amount_paise,
                state=PaymentState.FAILED,
                recovery_attempts=0,
            )
            self._payments[payment.payment_id] = existing
        return existing

    def get(self, payment_id: str) -> SimulatedPayment:
        if payment_id not in self._payments:
            raise KeyError(f"Unknown simulated payment {payment_id}")
        return self._payments[payment_id]

    def event_processed(self, payment: FailedPayment) -> bool:
        sim = self._payments.get(payment.payment_id)
        if sim is None:
            return False
        return payment.event_id in sim.processed_event_ids

    def advance_clock(self, payment_id: str, hours: float) -> None:
        sim = self.get(payment_id)
        sim.hours_since_last_recovery += max(0.0, hours)

    def execute(
        self,
        payment: FailedPayment,
        category: FailureCategory,
        action: RecoveryAction,
        *,
        allowed: bool,
        fault: InjectedFault | None = None,
    ) -> ExecutionResult:
        fault = fault or InjectedFault()
        sim = self.seed_failed(payment)

        if fault.force_invalid_action:
            return ExecutionResult(
                accepted=False,
                timed_out=False,
                error="Invalid recovery action",
                action=action,
                payment_id=payment.payment_id,
                event_id=payment.event_id,
            )

        if not allowed:
            return ExecutionResult(
                accepted=False,
                timed_out=False,
                error="Policy rejected action; execution skipped",
                action=action,
                payment_id=payment.payment_id,
                event_id=payment.event_id,
            )

        if payment.event_id in sim.processed_event_ids:
            return ExecutionResult(
                accepted=False,
                timed_out=False,
                error="Duplicate event_id; execution skipped",
                action=action,
                payment_id=payment.payment_id,
                event_id=payment.event_id,
            )

        if fault.timeout_on_execute:
            sim.last_execution_ok = False
            sim.state = PaymentState.UNKNOWN if sim.state == PaymentState.FAILED else sim.state
            return ExecutionResult(
                accepted=False,
                timed_out=True,
                error="Recovery API timeout",
                action=action,
                payment_id=payment.payment_id,
                event_id=payment.event_id,
            )

        sim.processed_event_ids.add(payment.event_id)
        sim.last_action = action
        sim.last_execution_ok = True
        sim.hours_since_last_recovery = 0.0

        if action == RecoveryAction.STOP_AND_ESCALATE:
            sim.state = PaymentState.ESCALATED
            return ExecutionResult(
                accepted=True,
                timed_out=False,
                error=None,
                action=action,
                payment_id=payment.payment_id,
                event_id=payment.event_id,
            )

        sim.recovery_attempts += 1
        sim.attempts_by_family[action.value] = sim.attempts_by_family.get(action.value, 0) + 1
        sim.state = PaymentState.RECOVERY_ATTEMPTED

        threshold = recovery_probability(
            payment, category, action, sim.recovery_attempts
        )
        draw = _unit_interval(
            f"outcome:{payment.payment_id}:{action.value}:{sim.recovery_attempts}"
        )
        if draw < threshold:
            sim.state = PaymentState.RECOVERED

        return ExecutionResult(
            accepted=True,
            timed_out=False,
            error=None,
            action=action,
            payment_id=payment.payment_id,
            event_id=payment.event_id,
        )
