"""Command line: ``act-pilot <command> --config <file>``.

Commands:

* ``estimate``: counts the calls per boundary and writes ``cost_estimate.md``; no call made;
* ``run-act``: the estimate, then the ACT runs (``--folds`` and ``--seeds`` restrict them);
* ``evaluate``: the answers on the outer folds, the baselines, the CSVs and the report;
* ``validity-template``: the questions to be rated in ``content_validity.csv``;
* ``all``: ``run-act``, ``evaluate`` and ``validity-template`` in sequence.

``--max-calls`` stops before any call when the estimate is above the given number, and
``--dry-run`` stops after the estimate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import load_config
from .costs import format_costs
from .pipeline import (
    BudgetExceeded,
    check_budget,
    clients,
    estimate,
    evaluate_stage,
    load_inputs,
    run_act_stage,
    validity_template,
    write_estimate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="act-pilot", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "run-act", "evaluate", "validity-template", "all"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="the YAML configuration")
        p.add_argument("--boundary", action="append", help="a boundary id; repeatable")
        if name in ("run-act", "evaluate", "all"):
            p.add_argument("--max-calls", type=int, default=None,
                           help="stop before calling if the estimate exceeds this number")
            p.add_argument("--dry-run", action="store_true", help="stop after the estimate")
        if name == "run-act":
            p.add_argument("--folds", type=int, nargs="+", default=None)
            p.add_argument("--seeds", type=int, nargs="+", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    inputs = load_inputs(config, args.boundary)
    costs = estimate(config, inputs)
    path = write_estimate(config, costs)
    if args.command == "estimate":
        print(format_costs(costs))
        print(f"\nWritten to {path}")
        return 0
    if args.command == "validity-template":
        print(f"Written to {validity_template(config, inputs)}")
        return 0
    print(format_costs(costs))
    try:
        check_budget(costs, args.max_calls)
    except BudgetExceeded as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    answerer, optimizer = clients(config)
    if args.command in ("run-act", "all"):
        run_act_stage(
            config, inputs, answerer, optimizer,
            folds=getattr(args, "folds", None), seeds=getattr(args, "seeds", None),
        )
    if args.command in ("evaluate", "all"):
        evaluate_stage(config, inputs, answerer, costs=costs, extra_stats_clients=[optimizer])
        print(f"Report: {config.output_dir / 'report.md'}")
    if args.command == "all":
        print(f"Template: {validity_template(config, inputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
