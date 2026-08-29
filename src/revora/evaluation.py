"""Held-out evaluation of Simple Retry, Static Rules, and Revora.

Metrics are computed from simulator states and audit logs. Nothing is hardcoded.
The held-out split is not used to set scoring weights or policy rules.
The same metric function is applied to every strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from revora.actions import AUTOMATED_ACTIONS
from revora.baselines import decide_simple_retry, decide_static_rules
from revora.config import Settings, load_settings
from revora.dataset import Dataset, generate_dataset
from revora.decision import decide as decide_revora
from revora.decision import decide_c2 as decide_revora_c2
from revora.decision_ranked import decide_c2_ai, decide_c2_heuristic
from revora.diagnosis import diagnose
from revora.orchestrator import DecideFn, run_cycle, run_until_terminal
from revora.schemas import (
    AuditRecord,
    FailedPayment,
    FailureCategory,
    PaymentState,
    Split,
    SystemMetrics,
)
from revora.simulator import RecoverySimulator

SYSTEMS: tuple[tuple[str, DecideFn], ...] = (
    ("simple_retry", decide_simple_retry),
    ("static_rules", decide_static_rules),
    ("revora", decide_revora),
    ("revora_c2", decide_revora_c2),
    ("revora_c2_heuristic", decide_c2_heuristic),
    ("revora_c2_ai", decide_c2_ai),
)


def format_inr(paise: int) -> str:
    rupees = paise / 100.0
    return f"₹{rupees:,.2f}"


@dataclass(frozen=True)
class CategorySlice:
    category: str
    n: int
    recovered: int
    recovery_rate: float
    revenue_at_risk_paise: int
    revenue_recovered_paise: int


@dataclass(frozen=True)
class RankerStats:
    payments: int
    used: int
    skipped: int
    fallback: int
    invalid: int
    agreed_heuristic: int
    disagreed_heuristic: int
    agreed_c2: int
    disagreed_c2: int
    by_category_used: tuple[tuple[str, int, int], ...]  # category, used, n


@dataclass(frozen=True)
class SystemRun:
    metrics: SystemMetrics
    audits: tuple[AuditRecord, ...]
    final_states: dict[str, PaymentState]
    by_category: tuple[CategorySlice, ...]
    ranker_stats: RankerStats | None = None


@dataclass(frozen=True)
class EvaluationReport:
    seed: int
    dataset_size: int
    split: Split
    payments_evaluated: int
    revenue_at_risk_paise: int
    runs: dict[str, SystemRun]

    def incremental_revenue_paise(self, system_id: str, baseline_id: str) -> int:
        return (
            self.runs[system_id].metrics.revenue_recovered_paise
            - self.runs[baseline_id].metrics.revenue_recovered_paise
        )


def _metrics(
    system_id: str,
    payments: tuple[FailedPayment, ...],
    audits: list[AuditRecord],
    simulator: RecoverySimulator,
) -> SystemMetrics:
    total = len(payments)
    at_risk = sum(p.amount_paise for p in payments)
    recovered_ids = {
        p.payment_id
        for p in payments
        if simulator.get(p.payment_id).state == PaymentState.RECOVERED
    }
    recovered_payments = len(recovered_ids)
    revenue_recovered = sum(p.amount_paise for p in payments if p.payment_id in recovered_ids)
    recovery_rate = recovered_payments / total if total else 0.0

    attempts = [simulator.get(p.payment_id).recovery_attempts for p in payments]
    avg_attempts = sum(attempts) / total if total else 0.0

    escalated = sum(
        1 for p in payments if simulator.get(p.payment_id).state == PaymentState.ESCALATED
    )

    unnecessary = 0
    for p in payments:
        if p.payment_id in recovered_ids:
            continue
        unnecessary += sum(
            1
            for a in audits
            if a.payment_id == p.payment_id
            and a.executed
            and a.action in AUTOMATED_ACTIONS
        )

    policy_violations = sum(1 for a in audits if a.executed and not a.allowed)

    executed_by_event: dict[tuple[str, str], int] = {}
    for a in audits:
        if a.executed:
            key = (a.payment_id, a.event_id)
            executed_by_event[key] = executed_by_event.get(key, 0) + 1
    duplicate_actions = sum(n - 1 for n in executed_by_event.values() if n > 1)

    verification_failures = 0
    for a in audits:
        if a.recorded_recovered and a.verified_state != PaymentState.RECOVERED:
            verification_failures += 1
        if a.recorded_recovered:
            sim = simulator.get(a.payment_id)
            if sim.state != PaymentState.RECOVERED:
                verification_failures += 1

    return SystemMetrics(
        system_id=system_id,
        total_payments=total,
        revenue_at_risk_paise=at_risk,
        recovered_payments=recovered_payments,
        revenue_recovered_paise=revenue_recovered,
        recovery_rate=recovery_rate,
        unnecessary_interventions=unnecessary,
        average_recovery_attempts=avg_attempts,
        policy_violations=policy_violations,
        duplicate_actions=duplicate_actions,
        verification_failures=verification_failures,
        actions_executed=sum(1 for a in audits if a.executed),
        actions_rejected=sum(1 for a in audits if not a.allowed),
        escalated_payments=escalated,
    )


def _category_slices(
    payments: tuple[FailedPayment, ...],
    simulator: RecoverySimulator,
) -> tuple[CategorySlice, ...]:
    buckets: dict[str, list[FailedPayment]] = {c.value: [] for c in FailureCategory}
    for p in payments:
        buckets[diagnose(p).category.value].append(p)
    slices: list[CategorySlice] = []
    for name in sorted(buckets):
        group = buckets[name]
        if not group:
            continue
        recovered = [p for p in group if simulator.get(p.payment_id).state == PaymentState.RECOVERED]
        at_risk = sum(p.amount_paise for p in group)
        won = sum(p.amount_paise for p in recovered)
        slices.append(
            CategorySlice(
                category=name,
                n=len(group),
                recovered=len(recovered),
                recovery_rate=len(recovered) / len(group),
                revenue_at_risk_paise=at_risk,
                revenue_recovered_paise=won,
            )
        )
    return tuple(slices)


def _first_audits(payments: tuple[FailedPayment, ...], audits: list[AuditRecord]) -> list[AuditRecord]:
    first: dict[str, AuditRecord] = {}
    for a in audits:
        if a.payment_id not in first:
            first[a.payment_id] = a
    return [first[p.payment_id] for p in payments if p.payment_id in first]


def _ranker_stats(
    payments: tuple[FailedPayment, ...],
    audits: list[AuditRecord],
) -> RankerStats | None:
    firsts = _first_audits(payments, audits)
    if not firsts or all(a.ai_status == "n/a" for a in firsts):
        return None
    used = sum(1 for a in firsts if a.ai_status == "used")
    skipped = sum(1 for a in firsts if a.ai_status == "skipped")
    fallback = sum(1 for a in firsts if a.ai_status == "fallback")
    invalid = sum(1 for a in firsts if a.ai_status == "invalid")
    ah = sum(1 for a in firsts if a.agreed_with_heuristic is True)
    dh = sum(1 for a in firsts if a.agreed_with_heuristic is False)
    ac = sum(1 for a in firsts if a.agreed_with_c2 is True)
    dc = sum(1 for a in firsts if a.agreed_with_c2 is False)
    by_cat: list[tuple[str, int, int]] = []
    buckets: dict[str, list[FailedPayment]] = {}
    audit_by_id = {a.payment_id: a for a in firsts}
    for p in payments:
        cat = diagnose(p).category.value
        buckets.setdefault(cat, []).append(p)
    for cat in sorted(buckets):
        group = buckets[cat]
        u = sum(1 for p in group if audit_by_id[p.payment_id].ai_status == "used")
        by_cat.append((cat, u, len(group)))
    return RankerStats(
        payments=len(firsts),
        used=used,
        skipped=skipped,
        fallback=fallback,
        invalid=invalid,
        agreed_heuristic=ah,
        disagreed_heuristic=dh,
        agreed_c2=ac,
        disagreed_c2=dc,
        by_category_used=tuple(by_cat),
    )


def evaluate_system(payments: tuple[FailedPayment, ...], system_id: str, decide: DecideFn) -> SystemRun:
    simulator = RecoverySimulator()
    audits: list[AuditRecord] = []
    for payment in payments:
        audits.extend(run_until_terminal(payment, decide, simulator))
        _decision, replay_audit = run_cycle(
            payment, decide, simulator, replay_same_event=True
        )
        audits.append(replay_audit)

    metrics = _metrics(system_id, payments, audits, simulator)
    finals = {p.payment_id: simulator.get(p.payment_id).state for p in payments}
    return SystemRun(
        metrics=metrics,
        audits=tuple(audits),
        final_states=finals,
        by_category=_category_slices(payments, simulator),
        ranker_stats=_ranker_stats(payments, audits),
    )


def evaluate(
    dataset: Dataset | None = None,
    settings: Settings | None = None,
    split: Split = Split.HELD_OUT,
) -> EvaluationReport:
    settings = settings or load_settings()
    dataset = dataset or generate_dataset(
        size=settings.dataset_size,
        seed=settings.seed,
        held_out_fraction=settings.held_out_fraction,
        dev_fraction=settings.dev_fraction,
    )
    payments = dataset.by_split(split)
    runs = {
        system_id: evaluate_system(payments, system_id, decide)
        for system_id, decide in SYSTEMS
    }
    at_risk = sum(p.amount_paise for p in payments)
    return EvaluationReport(
        seed=dataset.seed,
        dataset_size=len(dataset.payments),
        split=split,
        payments_evaluated=len(payments),
        revenue_at_risk_paise=at_risk,
        runs=runs,
    )


def render_report(report: EvaluationReport) -> str:
    order = (
        "simple_retry",
        "static_rules",
        "revora",
        "revora_c2",
        "revora_c2_heuristic",
        "revora_c2_ai",
    )
    titles = {
        "simple_retry": "Simple Retry",
        "static_rules": "Static Rules",
        "revora": "Revora v1 (pooled recovery history)",
        "revora_c2": "Revora C2 (POLICY: mechanism-aware ledger, not AI)",
        "revora_c2_heuristic": "C2 + Heuristic ranker",
        "revora_c2_ai": "C2 + Frozen AI adapter (no live LLM)",
    }

    def block(title: str, m: SystemMetrics) -> str:
        return (
            f"{title}\n"
            f"Recovery rate: {m.recovery_rate:.2%}\n"
            f"Revenue recovered: {format_inr(m.revenue_recovered_paise)}\n"
            f"Recovered payments: {m.recovered_payments}\n"
            f"Unnecessary interventions: {m.unnecessary_interventions}\n"
            f"Average recovery attempts: {m.average_recovery_attempts:.3f}\n"
            f"Policy violations: {m.policy_violations}\n"
            f"Duplicate actions: {m.duplicate_actions}\n"
            f"Verification failures: {m.verification_failures}\n"
        )

    lines = [
        "========================================",
        "REVORA RECOVERY EVALUATION",
        "========================================",
        "",
        f"Split: {report.split.value} (seed={report.seed}, dataset_n={report.dataset_size})",
        f"Payments evaluated: {report.payments_evaluated}",
        "",
        f"Revenue at risk: {format_inr(report.revenue_at_risk_paise)}",
        "",
    ]
    for sid in order:
        lines.append(block(titles[sid], report.runs[sid].metrics))

    inc_simple = report.incremental_revenue_paise("revora", "simple_retry")
    inc_static = report.incremental_revenue_paise("revora", "static_rules")
    inc_c2_simple = report.incremental_revenue_paise("revora_c2", "simple_retry")
    inc_c2_static = report.incremental_revenue_paise("revora_c2", "static_rules")
    inc_c2_v1 = report.incremental_revenue_paise("revora_c2", "revora")
    lines.extend(
        [
            "Revora v1 incremental vs Simple Retry:",
            f"{format_inr(inc_simple)}",
            "",
            "Revora v1 incremental vs Static Rules:",
            f"{format_inr(inc_static)}",
            "",
            "Revora C2 incremental vs Simple Retry:",
            f"{format_inr(inc_c2_simple)}",
            "",
            "Revora C2 incremental vs Static Rules:",
            f"{format_inr(inc_c2_static)}",
            "",
            "Revora C2 incremental vs Revora v1:",
            f"{format_inr(inc_c2_v1)}",
            "",
            "Note: C2 is a POLICY CHANGE (per-mechanism attempt ledger), not an AI improvement.",
            "",
            f"C2 + Heuristic incremental vs C2: {format_inr(report.incremental_revenue_paise('revora_c2_heuristic', 'revora_c2'))}",
            f"C2 + Frozen AI incremental vs C2: {format_inr(report.incremental_revenue_paise('revora_c2_ai', 'revora_c2'))}",
            f"C2 + Frozen AI incremental vs C2 + Heuristic: {format_inr(report.incremental_revenue_paise('revora_c2_ai', 'revora_c2_heuristic'))}",
            "",
            "Per-category recovery (held-out)",
            "----------------------------------------",
        ]
    )
    for slice_row in report.runs["revora"].by_category:
        cat = slice_row.category
        parts = [f"{cat} n={slice_row.n}"]
        for sid in order:
            row = next(s for s in report.runs[sid].by_category if s.category == cat)
            parts.append(
                f"{titles[sid]}={row.recovery_rate:.0%} ({row.recovered}/{row.n}) "
                f"{format_inr(row.revenue_recovered_paise)}"
            )
        lines.append(" | ".join(parts))

    ai_run = report.runs["revora_c2_ai"]
    stats = ai_run.ranker_stats
    if stats is not None:
        n = stats.payments or 1
        lines.extend(
            [
                "",
                "Frozen AI adapter usage (first decision per payment)",
                "----------------------------------------",
                f"Invocation (used): {stats.used}/{stats.payments} ({stats.used / n:.1%})",
                f"Skip table: {stats.skipped}/{stats.payments} ({stats.skipped / n:.1%})",
                f"Fallback: {stats.fallback}/{stats.payments}",
                f"Invalid output: {stats.invalid}/{stats.payments}",
                f"Agree with heuristic: {stats.agreed_heuristic}  disagree: {stats.disagreed_heuristic}",
                f"Agree with C2: {stats.agreed_c2}  disagree: {stats.disagreed_c2}",
                "Per-category AI used/n:",
            ]
        )
        for cat, used, total in stats.by_category_used:
            lines.append(f"  {cat}: {used}/{total}")

    lines.extend(["", "========================================"])
    return "\n".join(lines)
