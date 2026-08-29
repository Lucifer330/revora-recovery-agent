# Revora

Adaptive Revenue Recovery Control Plane — **milestone 1** (core experiment).

Revora decides whether a failed payment is worth recovering, recommends a **bounded** action, and will only execute that action if a **deterministic policy engine** allows it. Execution is simulated. Recovery is recorded only after **outcome verification** of the simulated payment state.

This is a synthetic, reproducible experiment for the Razorpay AI Buildathon 2026. It is **not** a production Razorpay integration, not a trained model, and not an autonomous financial agent.

## What this milestone is

A working control plane you can evaluate in one command:

```bash
python -m revora evaluate
```

Three systems run on the **same held-out** synthetic dataset:

| ID | System | How it chooses |
| --- | --- | --- |
| A | Simple Retry | Always recommends `RETRY_LATER` |
| B | Static Rules | Fixed category → action map, no score |
| C | Revora | Diagnose → knowledge lookup → candidates → explainable score → **policy gate** |

Recommendations never execute directly:

```
AI / heuristic recommendation
        ↓
Deterministic policy gate
        ↓
   allowed? ──yes──► execute (simulator) ──► verify state ──► audit
        │
        no → reject / explain
```

## What this milestone is not

- Machine learning
- RAG / vector search
- LLM agent
- Production Razorpay API
- A product dashboard

Those are later milestones, and only if they are actually implemented and measured.

## Architecture (milestone 1)

```
src/revora/
  dataset.py        reproducible synthetic failed payments + train/dev/held-out split
  diagnosis.py      failure_code + exhaustion context → FailureCategory
  knowledge.py      exact lookup into data/recovery_policies.json (Revora only)
  scoring.py        documented weighted heuristic (not ML)
  policy.py         shared safety vs Revora-only eligibility
  decision.py       Revora pipeline
  baselines.py      Simple Retry + Static Rules (shared safety only)
  environment.py    independent synthetic outcome model (not used at decision time)
  simulator.py      payment states; calls environment for P(recover | context, action)
  verification.py   recovery iff verified state == RECOVERED
  orchestrator.py   decide → execute → verify → audit; real clock / cooldown wait
  evaluation.py     same metrics for all three strategies
  failure_lab.py    six reproducible unsafe-path scenarios
  cli.py            evaluate | failure-lab
```

**Split rule:** `train` / `dev` / `held-out` are assigned from a hash of `payment_id`. Scoring weights and policy rules are constants. The held-out set is used only to **report** metrics. Environment outcomes are not features.

**Fair evaluation:** Baselines execute their own recommended actions whenever shared safety allows. Revora’s knowledge-base allow-list is not applied to A or B. The outcome environment does not take a strategy id. Run `python -m revora evaluate` for current held-out numbers; do not treat them as Razorpay production performance.

## Recoverability score (heuristic)

```
score = clip(
    0.40 * category_prior
  + 0.15 * attempt_factor
  + 0.15 * prior_recovery_factor
  + 0.15 * customer_success_rate
  + 0.10 * recency_factor
  + 0.05 * amount_factor
, 0, 1)
```

| Feature | Meaning |
| --- | --- |
| category_prior | A priori recoverability of the diagnosed class |
| attempt_factor | More original attempts → lower |
| prior_recovery_factor | More prior recovery tries → lower |
| customer_success_rate | Historical successes / (successes + failures) |
| recency_factor | Decays with hours since failure |
| amount_factor | Weak, non-linear ticket-size prior |

The score **ranks** candidate actions via heuristic expected value (`score * action_fit * amount`). The policy engine can still reject the top action.

**Assumptions:** weights were chosen up front, not fit on held-out data; category priors are operational intuition, not simulator leakage.

**Limitations:** not calibrated to live traffic; linear mix misses interactions; not a probability from a fitted model.

## Bounded actions

- `RETRY_LATER`
- `PAYMENT_METHOD_UPDATE`
- `RECOVERY_NUDGE`
- `ALTERNATIVE_PATH`
- `STOP_AND_ESCALATE`

## Simulator states

`PENDING` · `FAILED` · `RECOVERY_ATTEMPTED` · `RECOVERED` · `ESCALATED` · `UNKNOWN`

An execution request succeeding does **not** mark revenue recovered.

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

On macOS/Linux: `source .venv/bin/activate` and `cp .env.example .env`.

`.env` only sets `REVORA_SEED` and dataset size. There are no API keys.

## Commands

```bash
python -m revora evaluate
python -m revora evaluate --seed 42 --size 400 --split held_out
python -m revora failure-lab
python -m revora trace pay_syn_42_00007
python -m revora trace pay_syn_42_00151
pytest
```

Same seed + size ⇒ same report. That is a test, not a promise in prose.

## Failure Lab

Reproducible scenarios:

1. Duplicate webhook (same `event_id`)
2. Recovery API timeout
3. Verification timeout
4. Invalid recovery action
5. Maximum retry reached
6. Missing / invalid payment context

## Evaluation metrics

Computed from audit logs and verified states (never invented):

- payments evaluated, revenue at risk
- recovered payments, revenue recovered, recovery rate
- incremental revenue vs A and vs B
- unnecessary interventions, average recovery attempts
- policy violations, duplicate actions, verification failures

## Design choices

- One Python package, no microservices.
- Amounts are integer **paise**; reports print INR.
- Idempotency key = webhook `event_id`; processed IDs are stored on the simulated payment.
- Follow-up recovery cycles use **new** event IDs (retry scheduler), not duplicate webhooks.

## Ranking impact (not a revenue claim)

Frozen held-out evaluation (seed=42, n=400, 76 payments). C2 policy arm vs ranking arms:

| Arm | Unnecessary interventions | Avg recovery attempts | Revenue recovered |
| --- | --- | --- | --- |
| Revora C2 (policy only) | 60 | 1.329 | ₹319,967 |
| C2 + Heuristic | 44 | 1.105 | ₹319,768 |
| C2 + Frozen AI adapter | 43 | 1.079 | ₹319,769 |

**Ranking impact:** ranking (heuristic or frozen AI) reduces unnecessary intervention attempts by about 27–28% (60 → 44 heuristic, 60 → 43 frozen AI) at a revenue difference of ₹198–₹199 on ₹319,967 (about 0.06%, noise-level at n=76).

This is an **intervention-precision** result. It is **not** a claim that AI or the heuristic improves recovered revenue. On this cohort, C2 remains slightly ahead of both ranking arms on revenue (C2 ₹319,967 vs ranking ₹319,768–₹319,769). Static Rules remains highest on revenue (₹322,872).

## License

Private / hackathon use unless you add a license later.
