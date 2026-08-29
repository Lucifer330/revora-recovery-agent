"""Bounded recovery actions. These are labels only — execution is simulated."""

from revora.schemas import RecoveryAction

ACTION_DESCRIPTIONS: dict[RecoveryAction, str] = {
    RecoveryAction.RETRY_LATER: (
        "Schedule a bounded automatic retry of the same instrument after cooldown."
    ),
    RecoveryAction.PAYMENT_METHOD_UPDATE: (
        "Ask the customer to replace an expired, revoked, or invalid instrument."
    ),
    RecoveryAction.RECOVERY_NUDGE: (
        "Send a time-bounded reminder (e.g. after expected fund replenishment)."
    ),
    RecoveryAction.ALTERNATIVE_PATH: (
        "Offer a different rail or checkout path (e.g. UPI instead of card)."
    ),
    RecoveryAction.STOP_AND_ESCALATE: (
        "Stop automated recovery and escalate to a human or ticket queue."
    ),
}

AUTOMATED_ACTIONS = frozenset(
    {
        RecoveryAction.RETRY_LATER,
        RecoveryAction.PAYMENT_METHOD_UPDATE,
        RecoveryAction.RECOVERY_NUDGE,
        RecoveryAction.ALTERNATIVE_PATH,
    }
)
