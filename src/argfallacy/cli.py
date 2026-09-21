"""Command line: ``argfallacy annotations update``, ``argfallacy run plan|execute CONFIG``,
``argfallacy parse RUN_ID``."""

from __future__ import annotations

import argparse
import sys

from .env import load_env


def _cmd_annotations_update(args: argparse.Namespace) -> int:
    import os

    from .annotations import ANNOTATIONS_FILE, update
    from .schemes import SchemeError

    workbook = args.workbook or os.environ.get("WORKBOOK_PATH")
    if not workbook:
        raise SystemExit("no workbook: set WORKBOOK_PATH in .env or pass --workbook")
    try:
        counts = update(workbook)
    except SchemeError as error:
        raise SystemExit(f"{error}\nannotations.csv non e' stato toccato.") from None
    for sheet, (before, after) in counts.items():
        print(f"{sheet}: {before} righe prima, {after} adesso")
    print(f"scritto {ANNOTATIONS_FILE}")
    return 0


def _cmd_run_plan(args: argparse.Namespace) -> int:
    from .client import ResponseCache, RunConfig, load_items, plan, select_items, summarise_plan

    config = RunConfig.load(args.config)
    items = select_items(config, load_items(args.items))
    calls = plan(config, items)
    with ResponseCache() as cache:
        report = summarise_plan(calls, cache)
    print(f"configurazione : {config.name}")
    print(f"item           : {len(items)}")
    print(f"chiamate       : {report['calls']}")
    print(f"gia' in cache  : {report['cached']}")
    print(f"da fare        : {report['missing']}")
    print()
    print(f"{'modello':<18}{'totali':>8}{'cache':>8}{'da fare':>9}{'conc.':>7}{'minuti':>9}")
    print("-" * 59)
    for name, entry in sorted(report["per_model"].items()):
        print(f"{name:<18}{entry['total']:>8}{entry['cached']:>8}{entry['missing']:>9}"
              f"{entry['max_concurrency']:>7}{entry['estimated_minutes']:>9}")
    print()
    print("Nessuna chiamata fatta: `plan` non chiama niente.")
    return 0


def _cmd_run_execute(args: argparse.Namespace) -> int:
    from .client import RunConfig, execute, load_items, select_items

    config = RunConfig.load(args.config)
    items = select_items(config, load_items(args.items))
    manifest = execute(config, items, run_id=args.run_id)
    print(f"run_id          : {manifest['run_id']}")
    print(f"chiamate         : previste {manifest['calls_planned']}, "
          f"eseguite {manifest['calls_executed']}, "
          f"da cache {manifest['calls_from_cache']}, "
          f"fallite {manifest['calls_failed']}")
    return 1 if manifest["calls_failed"] else 0


def _cmd_parse(args: argparse.Namespace) -> int:
    from .client.run import default_runs_dir
    from .parse import parse_run

    written = parse_run(default_runs_dir() / args.run_id)
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argfallacy")
    sub = parser.add_subparsers(dest="group", required=True)

    annotations = sub.add_parser("annotations", help="data/annotations.csv").add_subparsers(
        dest="command", required=True
    )
    update_cmd = annotations.add_parser(
        "update", help="replace the rows of the 'Annotazione Dumitru' sheets with the workbook's"
    )
    update_cmd.add_argument("--workbook", default=None, help="default: WORKBOOK_PATH")
    update_cmd.set_defaults(func=_cmd_annotations_update)
    run = sub.add_parser("run", help="ask the models").add_subparsers(
        dest="command", required=True
    )
    for name, help_text, func in (
        ("plan", "count the calls without making any", _cmd_run_plan),
        ("execute", "make the missing calls and write the run", _cmd_run_execute),
    ):
        parser_for = run.add_parser(name, help=help_text)
        parser_for.add_argument("config", help="a YAML run configuration")
        parser_for.add_argument(
            "--items", default=None, help="an items table (default: data/items.csv)"
        )
        if name == "execute":
            parser_for.add_argument("--run-id", default=None, help="resume this run")
        parser_for.set_defaults(func=func)

    parse_cmd = sub.add_parser("parse", help="turn a run's raw.jsonl into tables")
    parse_cmd.add_argument("run_id")
    parse_cmd.set_defaults(func=_cmd_parse)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
