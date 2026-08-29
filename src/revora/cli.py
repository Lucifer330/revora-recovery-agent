"""CLI: one command for evaluation, optional Failure Lab."""

from __future__ import annotations

import argparse
import sys

from revora.config import load_settings
from revora.dataset import generate_dataset
from revora.evaluation import evaluate, render_report
from revora.failure_lab import render_lab, run_failure_lab
from revora.schemas import Split


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except OSError:
            pass

    parser = argparse.ArgumentParser(prog="revora", description="Revora recovery control plane (milestone 1)")
    sub = parser.add_subparsers(dest="command")

    eval_p = sub.add_parser("evaluate", help="Run held-out evaluation of three systems")
    eval_p.add_argument("--split", choices=[s.value for s in Split], default=Split.HELD_OUT.value)
    eval_p.add_argument("--size", type=int, default=None)
    eval_p.add_argument("--seed", type=int, default=None)

    sub.add_parser("failure-lab", help="Run reproducible Failure Lab scenarios")

    tr = sub.add_parser("trace", help="Print a 7-step decision trace for one payment_id")
    tr.add_argument("payment_id", help="e.g. pay_syn_42_00007 (held-out UNKNOWN) or pay_syn_42_00151 (expired, prev=2)")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--size", type=int, default=400)

    args = parser.parse_args(argv)
    command = args.command or "evaluate"

    if command == "failure-lab":
        results = run_failure_lab()
        print(render_lab(results))
        return 0 if all(r.passed for r in results) else 1

    if command == "trace":
        from revora.trace import find_payment, render_trace

        payment = find_payment(args.payment_id, size=args.size, seed=args.seed)
        print(render_trace(payment))
        return 0

    settings = load_settings()
    size = getattr(args, "size", None) or settings.dataset_size
    seed = getattr(args, "seed", None) or settings.seed
    dataset = generate_dataset(
        size=size,
        seed=seed,
        held_out_fraction=settings.held_out_fraction,
        dev_fraction=settings.dev_fraction,
    )
    split = Split(getattr(args, "split", None) or Split.HELD_OUT.value)
    report = evaluate(dataset=dataset, settings=settings, split=split)
    print(render_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
