"""Domain schemas for failed payments, recovery actions, and audit records.

Amounts are stored as integer paise (1 INR = 100 paise) to avoid float money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureCategory(str, Enum):
    TRANSIENT_NETWORK = "TRANSIENT_NETWORK"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    EXPIRED_PAYMENT_METHOD = "EXPIRED_PAYMENT_METHOD"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    REPEATED_FAILURE = "REPEATED_FAILURE"
    UNKNOWN = "UNKNOWN"


class PaymentMethodType(str, Enum):
    CARD = "CARD"
    UPI = "UPI"
    NETBANKING = "NETBANKING"
    WALLET = "WALLET"


class PaymentState(str, Enum):
    PENDING = "PENDING"
    FAILED = "FAILED"
    RECOVERY_ATTEMPTED = "RECOVERY_ATTEMPTED"
    RECOVERED = "RECOVERED"
    ESCALATED = "ESCALATED"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(str, Enum):
    RETRY_LATER = "RETRY_LATER"
    PAYMENT_METHOD_UPDATE = "PAYMENT_METHOD_UPDATE"
    RECOVERY_NUDGE = "RECOVERY_NUDGE"
    ALTERNATIVE_PATH = "ALTERNATIVE_PATH"
    STOP_AND_ESCALATE = "STOP_AND_ESCALATE"


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    HELD_OUT = "held_out"


@dataclass(frozen=True)
class CustomerHistory:
    successful_payments: int
    failed_payments: int
    days_since_last_success: int | None

    @property
    def observed_success_rate(self) -> float:
        total = self.successful_payments + self.failed_payments
        if total <= 0:
            return 0.5
        return self.successful_payments / total


@dataclass(frozen=True)
class FailedPayment:
    """A clearly synthetic failed-payment event.

    ``event_id`` is the webhook/idempotency key. Replaying the same
    ``event_id`` must not trigger another recovery action.
    """

    payment_id: str
    event_id: str
    amount_paise: int
    currency: str
    failure_code: str
    payment_method: PaymentMethodType
    attempt_count: int
    previous_recovery_attempts: int
    hours_since_failure: float
    customer: CustomerHistory
    created_at_epoch_s: int
    merchant_id: str
    split: Split = Split.TRAIN

    def with_split(self, split: Split) -> FailedPayment:
        return self.evolve(split=split)

    def evolve(self, **changes: Any) -> FailedPayment:
        data = {
            "payment_id": self.payment_id,
            "event_id": self.event_id,
            "amount_paise": self.amount_paise,
            "currency": self.currency,
            "failure_code": self.failure_code,
            "payment_method": self.payment_method,
            "attempt_count": self.attempt_count,
            "previous_recovery_attempts": self.previous_recovery_attempts,
            "hours_since_failure": self.hours_since_failure,
            "customer": self.customer,
            "created_at_epoch_s": self.created_at_epoch_s,
            "merchant_id": self.merchant_id,
            "split": self.split,
        }
        data.update(changes)
        return FailedPayment(**data)


@dataclass(frozen=True)
class Diagnosis:
    category: FailureCategory
    failure_code: str
    reasons: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    raw_value: float
    weight: float
    weighted_value: float
    note: str


@dataclass(frozen=True)
class RecoverabilityScore:
    """Transparent heuristic score in [0, 1]. Not a trained model."""

    value: float
    components: tuple[ScoreComponent, ...]
    formula: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "formula": self.formula,
            "components": [
                {
                    "name": c.name,
                    "raw_value": c.raw_value,
                    "weight": c.weight,
                    "weighted_value": c.weighted_value,
                    "note": c.note,
                }
                for c in self.components
            ],
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class CandidateAction:
    action: RecoveryAction
    rationale: str
    knowledge_id: str


@dataclass(frozen=True)
class PolicyDecision:
    action: RecoveryAction
    allowed: bool
    reason: str
    rule_id: str


@dataclass(frozen=True)
class SystemDecision:
    """Output of a recovery system before (or instead of) execution."""

    system_id: str
    action: RecoveryAction
    allowed: bool
    policy: PolicyDecision
    diagnosis: Diagnosis
    candidates: tuple[CandidateAction, ...]
    score: RecoverabilityScore | None
    ranking_notes: tuple[str, ...] = ()
    ranker_id: str = ""
    ai_status: str = "n/a"
    agreed_with_heuristic: bool | None = None
    agreed_with_c2: bool | None = None


@dataclass
class SimulatedPayment:
    """Mutable payment record inside the simulator. Decision engines never see this."""

    payment_id: str
    amount_paise: int
    state: PaymentState
    recovery_attempts: int = 0
    last_action: RecoveryAction | None = None
    last_execution_ok: bool = False
    hours_since_last_recovery: float = 999.0
    processed_event_ids: set[str] = field(default_factory=set)
    claimed_recovered_without_verify: bool = False
    attempts_by_family: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    timed_out: bool
    error: str | None
    action: RecoveryAction
    payment_id: str
    event_id: str


@dataclass(frozen=True)
class VerificationResult:
    verified_state: PaymentState
    recovered: bool
    reason: str


@dataclass(frozen=True)
class AuditRecord:
    system_id: str
    payment_id: str
    event_id: str
    action: RecoveryAction
    allowed: bool
    policy_reason: str
    executed: bool
    execution_timed_out: bool
    execution_error: str | None
    verified_state: PaymentState | None
    recorded_recovered: bool
    duplicate_suppressed: bool
    notes: str
    ranker_id: str = ""
    ai_status: str = "n/a"
    agreed_with_heuristic: bool | None = None
    agreed_with_c2: bool | None = None


@dataclass(frozen=True)
class SystemMetrics:
    system_id: str
    total_payments: int
    revenue_at_risk_paise: int
    recovered_payments: int
    revenue_recovered_paise: int
    recovery_rate: float
    unnecessary_interventions: int
    average_recovery_attempts: float
    policy_violations: int
    duplicate_actions: int
    verification_failures: int
    actions_executed: int
    actions_rejected: int
    escalated_payments: int
