"""Reproducible synthetic failed-payment dataset.

Every call with the same seed and size yields the same payments and splits.
Splits are assigned from a hash of payment_id so generators stay stable if
size increases. Decision engines must not read ``split`` or simulator state.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from revora.schemas import (
    CustomerHistory,
    FailedPayment,
    PaymentMethodType,
    Split,
)

FAILURE_CODES = (
    "gateway_timeout",
    "network_error",
    "issuer_unavailable",
    "insufficient_funds",
    "bank_decline_funds",
    "card_expired",
    "invalid_card",
    "token_revoked",
    "authentication_failed",
    "3ds_failed",
    "upi_pin_failed",
    "repeated_decline",
    "max_attempts_issuer",
    "unknown_decline",
    "do_not_honor",
)

# Target mix (approximate). Not used at decision time.
CODE_WEIGHTS = (
    12, 10, 8,  # transient
    14, 10,  # funds
    8, 6, 6,  # expired/invalid
    7, 6, 5,  # auth
    4, 3,  # repeated
    4, 4,  # unknown
)

METHODS = (
    PaymentMethodType.CARD,
    PaymentMethodType.UPI,
    PaymentMethodType.NETBANKING,
    PaymentMethodType.WALLET,
)

# Common INR ticket sizes in paise.
AMOUNT_TABLE_PAISE = (
    9900,
    19900,
    49900,
    99900,
    149900,
    249900,
    499900,
    999900,
    2499900,
    4999900,
)


def _stable_split(payment_id: str, held_out: float, dev: float) -> Split:
    digest = hashlib.sha256(f"split:{payment_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < held_out:
        return Split.HELD_OUT
    if bucket < held_out + dev:
        return Split.DEV
    return Split.TRAIN


@dataclass(frozen=True)
class Dataset:
    seed: int
    payments: tuple[FailedPayment, ...]

    def by_split(self, split: Split) -> tuple[FailedPayment, ...]:
        return tuple(p for p in self.payments if p.split == split)


def generate_dataset(
    size: int = 400,
    seed: int = 42,
    held_out_fraction: float = 0.20,
    dev_fraction: float = 0.15,
) -> Dataset:
    if size < 1:
        raise ValueError("size must be >= 1")
    if held_out_fraction + dev_fraction >= 1:
        raise ValueError("train fraction must be positive")

    rng = random.Random(seed)
    payments: list[FailedPayment] = []

    for i in range(size):
        payment_id = f"pay_syn_{seed}_{i:05d}"
        event_id = f"evt_syn_{seed}_{i:05d}"
        code = rng.choices(FAILURE_CODES, weights=CODE_WEIGHTS, k=1)[0]
        method = rng.choice(METHODS)
        amount = rng.choice(AMOUNT_TABLE_PAISE)
        if rng.random() < 0.08:
            amount += rng.randrange(0, 10000)

        # Repeated-looking codes already carry higher attempts.
        if code in {"repeated_decline", "max_attempts_issuer"}:
            attempt_count = rng.randint(5, 8)
            prev_recovery = rng.randint(2, 5)
        else:
            attempt_count = rng.randint(1, 4)
            prev_recovery = rng.randint(0, 2)

        hours = round(rng.uniform(0.1, 96.0), 2)
        successes = rng.randint(0, 20)
        failures = rng.randint(0, 8)
        days_last = None if successes == 0 else rng.randint(1, 180)

        payment = FailedPayment(
            payment_id=payment_id,
            event_id=event_id,
            amount_paise=amount,
            currency="INR",
            failure_code=code,
            payment_method=method,
            attempt_count=attempt_count,
            previous_recovery_attempts=prev_recovery,
            hours_since_failure=hours,
            customer=CustomerHistory(
                successful_payments=successes,
                failed_payments=failures,
                days_since_last_success=days_last,
            ),
            created_at_epoch_s=1_700_000_000 + i * 60,
            merchant_id=f"merch_syn_{rng.randint(1, 12):02d}",
        )
        payments.append(payment.with_split(_stable_split(payment_id, held_out_fraction, dev_fraction)))

    return Dataset(seed=seed, payments=tuple(payments))
