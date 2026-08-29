# Decisions

## UNKNOWN: 12% (v1/C2) vs 0% (C2 + Heuristic / C2 + Frozen AI)

This is **not** a C2 ledger bug. On the frozen held-out set (seed=42, n=400, 76 held-out), **UNKNOWN recovery is 12% for both Revora v1 and Revora C2** (1/8). It is **0% for C2 + Heuristic and C2 + Frozen AI** (0/8). Static Rules is 25% (2/8) on the same eight payments.

### What the code does

UNKNOWN knowledge (`src/revora/data/recovery_policies.json`, `unknown_failure`) allows `RECOVERY_NUDGE` and `STOP_AND_ESCALATE`, with `stop_after_recovery_attempts: 2`.

**v1** (`evaluate_revora_eligibility`): pooled `previous_recovery_attempts + attempts_this_run >= 2` blocks **nudge**. STOP remains allowed. Scoring still ranks nudge above STOP when nudge is eligible (`expected_value_paise` forces STOP EV = 0 in `scoring.py`).

**C2** (`evaluate_revora_eligibility_c2` / `family_attempts_for_c2`): unlabeled history is charged only to the `RETRY_LATER` family. Nudge starts at 0 this episode, so **C2 still allows nudge** when v1 would already have stopped (e.g. `pay_syn_42_00011`, `prev_rec=2`). C2 is *more* permissive on UNKNOWN, not less.

Traced held-out examples at first decision (`PolicyContext` empty episode):

| payment_id | prev_rec | v1 nudge eligible | C2 nudge eligible | v1 action | C2 action | Heuristic / Frozen AI |
| --- | --- | --- | --- | --- | --- | --- |
| `pay_syn_42_00007` | 0 | yes | yes | NUDGE | NUDGE | STOP (skip) |
| `pay_syn_42_00011` | 2 | **no** | **yes** | STOP | NUDGE | STOP (skip) |
| `pay_syn_42_00173` | 1 | yes | yes | NUDGE | NUDGE | STOP (skip) |
| `pay_syn_42_00178` | 2 | **no** | **yes** | STOP | NUDGE | STOP (skip) |
| `pay_syn_42_00205` | 2 | **no** | **yes** | STOP | NUDGE | STOP (skip) |

**C2 + Heuristic** (`rankers/heuristic.py` `_PRIORITY[UNKNOWN]`): order is `STOP_AND_ESCALATE`, then `RECOVERY_NUDGE`. STOP is always C2-eligible, so the heuristic **always chooses STOP** on UNKNOWN. That is ranking, not eligibility.

**C2 + Frozen AI**: skip table (`rankers/skip.py`) lists UNKNOWN as skip-AI. Skipped payments use the heuristic. Same STOP.

### Why this is left unchanged

Undiagnosed issuer declines have no reliable mechanism. Preferring STOP when the cause is UNKNOWN is a deliberate conservative ranker/skip policy (AI is not used; the heuristic does not spend a nudge “just in case”). C2’s ledger still *permits* a nudge; v1/C2 scoring still *takes* it when eligible. Changing C2 eligibility to “fix” 0% would be the wrong layer. Changing the heuristic to prefer nudge would be an UNKNOWN recovery retune, which is out of scope for this investigation.

No code behavior was changed for this finding.
