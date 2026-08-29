"""Outcome verification.

Never treat execution acceptance as recovery. Recovery is recorded only
when the simulated payment state is RECOVERED.
"""

from __future__ import annotations

from revora.schemas import PaymentState, VerificationResult
from revora.simulator import InjectedFault, RecoverySimulator


def verify_outcome(
    simulator: RecoverySimulator,
    payment_id: str,
    *,
    execution_accepted: bool,
    fault: InjectedFault | None = None,
) -> VerificationResult:
    fault = fault or InjectedFault()
    if fault.timeout_on_verify:
        return VerificationResult(
            verified_state=PaymentState.UNKNOWN,
            recovered=False,
            reason="Verification timeout; recovery not recorded",
        )

    sim = simulator.get(payment_id)
    state = sim.state
    recovered = state == PaymentState.RECOVERED
    if execution_accepted and not recovered:
        reason = (
            f"Execution accepted but verified state is {state.value}; "
            "not counted as recovered"
        )
    elif recovered:
        reason = "Verified state is RECOVERED"
    else:
        reason = f"Verified state is {state.value}"
    return VerificationResult(verified_state=state, recovered=recovered, reason=reason)
