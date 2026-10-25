"""Command line: ``argfallacy annotations update|samples``, ``argfallacy run plan|execute CONFIG``,
``argfallacy parse RUN_ID``, ``argfallacy report run RUN_ID`` (and its earlier name
``argfallacy pilot report RUN_ID``), ``argfallacy score prior``, ``argfallacy evaluate CONFIG``."""

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
        counts = update(workbook, skip=args.skip)
    except SchemeError as error:
        raise SystemExit(f"{error}\nannotations.csv was not touched.") from None
    for sheet, (before, after) in counts.items():
        print(f"{sheet}: {before} rows before, {after} now")
    print(f"wrote {ANNOTATIONS_FILE}")
    return 0


def _cmd_annotations_samples(args: argparse.Namespace) -> int:
    from .annotation_samples import SampleConfig, write_samples
    from .schemes import SchemeError

    try:
        manifest = write_samples(SampleConfig.load(args.config), args.out)
    except SchemeError as error:
        raise SystemExit(f"{error}\nnothing was written.") from None
    for sample, counts in manifest["counts"].items():
        print(f"{sample}: {sum(counts.values())} items, {counts}")
    print(f"rounds of the direct sample: {manifest['rounds']}")
    return 0


def _entries(config, names: list[str] | None) -> list[str]:
    """The entries named with ``--model``, or all of them; an unknown name stops here."""
    from .client import select_models
    from .schemes import SchemeError

    try:
        return select_models(config, names)
    except SchemeError as error:
        raise SystemExit(str(error)) from None


def _cmd_run_plan(args: argparse.Namespace) -> int:
    from .client import ResponseCache, RunConfig, load_items, plan, select_items, summarise_plan
    from .schemes import SchemeError

    config = RunConfig.load(args.config)
    entries = _entries(config, args.model)
    items = select_items(config, load_items(args.items))
    try:
        calls = plan(config, items, only=entries)
    except SchemeError as error:
        raise SystemExit(str(error)) from None
    with ResponseCache() as cache:
        report = summarise_plan(calls, cache)
    print(f"configuration  : {config.name}")
    print(f"items          : {len(items)}")
    print(f"calls          : {report['calls']}")
    if config.stages == ["ssoft"] and config.history:
        print("planning       : first calls only; later levels are planned during the run")
    print(f"already cached : {report['cached']}")
    print(f"to do          : {report['missing']}")
    print()
    print(f"{'model':<18}{'total':>8}{'cache':>8}{'to do':>9}{'conc.':>7}{'minutes':>9}")
    print("-" * 59)
    for name, entry in sorted(report["per_model"].items()):
        print(f"{name:<18}{entry['total']:>8}{entry['cached']:>8}{entry['missing']:>9}"
              f"{entry['max_concurrency']:>7}{entry['estimated_minutes']:>9}")
    print()
    print("No call made: `plan` calls nothing.")
    return 0


def _cmd_run_execute(args: argparse.Namespace) -> int:
    from .client import RunConfig, execute, load_items, select_items
    from .schemes import SchemeError

    config = RunConfig.load(args.config)
    entries = _entries(config, args.model)
    items = select_items(config, load_items(args.items))
    try:
        manifest = execute(config, items, run_id=args.run_id, only=entries)
    except SchemeError as error:
        raise SystemExit(str(error)) from None
    print(f"run_id          : {manifest['run_id']}")
    print(f"calls           : planned {manifest['calls_planned']}, "
          f"executed {manifest['calls_executed']}, "
          f"from cache {manifest['calls_from_cache']}, "
          f"failed {manifest['calls_failed']}")
    per_model = manifest["calls_per_model"]
    if args.model:
        for name in entries:
            counts = per_model[name]
            print(f"  {name:<14}: planned {counts['planned']}, executed {counts['executed']}, "
                  f"from cache {counts['from_cache']}, failed {counts['failed']}")
    # the exit code speaks for the entries this invocation ran
    return 1 if any(per_model[name]["failed"] for name in entries) else 0


def _cmd_parse(args: argparse.Namespace) -> int:
    from .client.run import default_runs_dir
    from .parse import parse_run

    written = parse_run(default_runs_dir() / args.run_id)
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_report_format(args: argparse.Namespace) -> int:
    from .client.run import default_runs_dir
    from .format_report import check_format
    from .schemes import SchemeError

    try:
        checks, shares = check_format(default_runs_dir() / args.run_id, args.model)
    except (SchemeError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    for entry, group in checks.groupby("model", sort=False):
        print(f"\n{entry}")
        print(group.drop(columns="model").to_string(index=False))
        own = shares[shares["model"] == entry]
        if not own.empty:
            print("\nOutside-arc answer shares (informational):")
            print(own.drop(columns="model").to_string(index=False))
    return int(checks.empty or checks["failed"].gt(0).any())


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .client.run import default_runs_dir
    from .eval.evaluate import EvaluationConfig, evaluate
    from .schemes import SchemeError

    try:
        out = evaluate(EvaluationConfig.load(args.config), runs_dir=default_runs_dir(),
                       progress=lambda step: print(f"  {step}", flush=True))
    except SchemeError as error:
        raise SystemExit(str(error)) from None
    print(f"  {out / ('summary.md' if (out / 'summary.md').is_file() else 'summary.html')}")
    return 0


def _cmd_report_cqs(args: argparse.Namespace) -> int:
    from .client.run import default_runs_dir
    from .eval.cq import write_cq_report

    out = write_cq_report(default_runs_dir() / args.run_id, args.out)
    print(f"  {out / 'cq_report.md'}")
    return 0


def _cmd_report_summary(args: argparse.Namespace) -> int:
    from .eval.html_report import write_summary

    print(write_summary(args.evaluation, args.judge_audit, args.revised_cqs))
    return 0


def _cmd_report_run(args: argparse.Namespace) -> int:
    from .client.run import default_runs_dir
    from .eval.run_report import PILOT_REPORT_NAME, REPORT_NAME, write_report

    # `pilot report` is the earlier name: it writes the report under the earlier file name
    pilot = args.group == "pilot"
    name = PILOT_REPORT_NAME if pilot else REPORT_NAME
    run_dir = default_runs_dir() / args.run_id
    try:
        path = write_report(run_dir, name=name, title="Pilot report" if pilot else "Run report",
                            read_raw=not args.no_raw)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    print(f"  {name}: {path}")
    return 0


def _cmd_score_prior(args: argparse.Namespace) -> int:
    import os
    from pathlib import Path

    from .client.run import default_runs_dir
    from .eval import score_prior
    from .schemes import SchemeError

    prior_dir = os.environ.get("PRIOR_RUNS_DIR")
    if not prior_dir:
        raise SystemExit("no prior runs: set PRIOR_RUNS_DIR in .env")
    out = Path(args.out) if args.out else default_runs_dir() / "scores" / "prior"
    try:
        written = score_prior(prior_dir, out)
    except SchemeError as error:
        raise SystemExit(str(error)) from None
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
    update_cmd.add_argument("--skip", action="append", default=[], metavar="SHEET",
                            help="a sheet not to read, its rows stay; repeatable")
    update_cmd.set_defaults(func=_cmd_annotations_update)
    samples_cmd = annotations.add_parser(
        "samples", help="draw the annotation samples and write the direct-verdict sheet"
    )
    samples_cmd.add_argument("config", help="a YAML sampling configuration")
    samples_cmd.add_argument("--out", required=True, help="a folder outside the repository")
    samples_cmd.set_defaults(func=_cmd_annotations_samples)
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
        parser_for.add_argument(
            "--model", action="append", default=None, metavar="NAME",
            help="only this entry of the configuration; can be repeated",
        )
        if name == "execute":
            parser_for.add_argument(
                "--run-id", default=None, help="resume this run, or add entries to it"
            )
        parser_for.set_defaults(func=func)

    parse_cmd = sub.add_parser("parse", help="turn a run's raw.jsonl into tables")
    parse_cmd.add_argument("run_id")
    parse_cmd.set_defaults(func=_cmd_parse)

    for group, group_help, command, help_text in (
        ("report", "the report of a run", "run",
         "write run_report.md from a parsed run (answers.csv, summary.csv)"),
        ("pilot", "earlier name of `report run`", "report",
         "the same report, written as pilot_report.md"),
    ):
        commands = sub.add_parser(group, help=group_help).add_subparsers(
            dest="command", required=True
        )
        report_cmd = commands.add_parser(command, help=help_text)
        report_cmd.add_argument("run_id")
        report_cmd.add_argument(
            "--no-raw", action="store_true",
            help="skip the token counts, which read raw.jsonl one line at a time",
        )
        report_cmd.set_defaults(func=_cmd_report_run)
        if group == "report":
            format_cmd = commands.add_parser("format", help="check verdict response formats")
            format_cmd.add_argument("run_id")
            format_cmd.add_argument("--model", help="restrict checks to one manifest entry")
            format_cmd.set_defaults(func=_cmd_report_format)
            cqs_cmd = commands.add_parser("cqs", help="CQ diagnostics and human agreement")
            cqs_cmd.add_argument("run_id")
            cqs_cmd.add_argument("--out", required=True, help="an evaluation output folder")
            cqs_cmd.set_defaults(func=_cmd_report_cqs)
            summary_cmd = commands.add_parser(
                "summary", help="visual HTML of saved evaluation tables")
            summary_cmd.add_argument("evaluation", help="an evaluation output folder")
            summary_cmd.add_argument("--judge-audit", help="optional four-judge audit folder")
            summary_cmd.add_argument("--revised-cqs", help="optional revised-CQ diagnostics folder")
            summary_cmd.set_defaults(func=_cmd_report_summary)

    score = sub.add_parser("score", help="compute the metrics").add_subparsers(
        dest="command", required=True
    )
    prior_cmd = score.add_parser(
        "prior", help="score the prior runs under PRIOR_RUNS_DIR, compat and canonical"
    )
    prior_cmd.add_argument("--out", default=None, help="default: runs/scores/prior/")
    prior_cmd.set_defaults(func=_cmd_score_prior)

    evaluate_cmd = sub.add_parser(
        "evaluate", help="the aggregators against A0, written into evaluations/<name>/"
    )
    evaluate_cmd.add_argument("config", help="a YAML evaluation configuration")
    evaluate_cmd.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
