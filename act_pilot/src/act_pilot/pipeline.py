"""The stages of the pilot: estimate, ACT, evaluation, content-validity template.

Output layout under ``output_dir``::

    calls.jsonl                       one line per LLM request (cache hits included)
    cache/                            the reply cache (unless cache_dir is set elsewhere)
    cost_estimate.md                  the estimate written before the ACT stage
    act/<B>_act_candidates.csv        every candidate question of every ACT run
    act/<B>_act_selected.csv          the selected question of each (outer fold, seed)
    evaluation/<B>_items.csv          one row per item and candidate or baseline
    evaluation/<B>_questions.csv      one row per candidate or baseline
    evaluation/<B>_sources.csv        one row per candidate or baseline and source
    content_validity_template.csv     the questions to rate by hand
    report.md                         the report
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .acceptance import load_content_validity, write_validity_template
from .act import ActRun, run_act
from .baselines import source_only, tfidf_cart, tfidf_rulefit
from .config import PilotConfig, load_manual_rewrites
from .costs import BoundaryCost, estimate_boundary, format_costs
from .data import LabeledItem, boundary_items, load_folds, load_items
from .evaluate import BoundaryEvaluation, build_candidates, predict, summarise_boundary
from .llm import ChatBackend, LLMClient, make_clients, total_stats
from .report import (
    act_candidate_rows,
    act_selected_rows,
    read_act_selected,
    render_report,
    write_item_csv,
    write_question_csv,
    write_rows,
    write_source_csv,
)
from .splits import fold_splits


class BudgetExceeded(RuntimeError):
    """Raised when the estimated number of calls is above the limit given by the user."""


@dataclass
class Inputs:
    items: dict[str, list[LabeledItem]]
    manual: dict[str, list[str]]


def act_dir(config: PilotConfig) -> Path:
    return config.output_dir / "act"


def evaluation_dir(config: PilotConfig) -> Path:
    return config.output_dir / "evaluation"


def load_inputs(config: PilotConfig, boundary_ids: Sequence[str] | None = None) -> Inputs:
    """The items of each selected boundary, with their folds, and the manual rewrites."""
    ids = list(boundary_ids or config.boundaries)
    items = load_items(config.items, config.columns)
    folds = load_folds(config.folds, config.n_outer_folds)
    by_boundary = {bid: boundary_items(items, folds, config.boundary(bid)) for bid in ids}
    manual = load_manual_rewrites(config.manual_rewrites, config.boundaries)
    return Inputs(items=by_boundary, manual={bid: manual[bid] for bid in ids})


def estimate(config: PilotConfig, inputs: Inputs) -> list[BoundaryCost]:
    return [
        estimate_boundary(config, bid, items, len(inputs.manual[bid]))
        for bid, items in inputs.items.items()
    ]


def check_budget(costs: Sequence[BoundaryCost], max_calls: int | None) -> None:
    if max_calls is None:
        return
    total = sum(c.answers_with_cache + c.optimizer_max for c in costs)
    if total > max_calls:
        raise BudgetExceeded(
            f"the estimate allows up to {total:,} calls, above --max-calls {max_calls:,}"
        )


def clients(
    config: PilotConfig, backends: dict[str, ChatBackend] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> tuple[LLMClient, LLMClient]:
    kwargs = {} if sleep is None else {"sleep": sleep}
    return make_clients(config.llm, config.cache_dir, config.output_dir / "calls.jsonl",
                        backends, **kwargs)


def _merge_rows(path: Path, new_rows: list[dict], replaced: set[tuple[int, int]]) -> list[dict]:
    """Rows of earlier runs are kept unless their (outer fold, seed) was run again."""
    if not path.exists():
        return new_rows
    with path.open(encoding="utf-8", newline="") as handle:
        old = [r for r in csv.DictReader(handle)
               if (int(r["outer_fold"]), int(r["seed"])) not in replaced]
    return old + new_rows


def run_act_stage(
    config: PilotConfig,
    inputs: Inputs,
    answerer: LLMClient,
    optimizer: LLMClient,
    *,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, list[ActRun]]:
    """ACT runs for every selected boundary, outer fold and seed; CSVs are written."""
    out: dict[str, list[ActRun]] = {}
    directory = act_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    for bid, items in inputs.items.items():
        boundary = config.boundary(bid)
        runs = []
        for split in fold_splits(items, config.act):
            if folds is not None and split.outer_fold not in folds:
                continue
            for seed in seeds if seeds is not None else config.act.seeds:
                runs.append(
                    run_act(
                        boundary,
                        split.inner_train,
                        split.inner_val,
                        answerer,
                        optimizer,
                        config.act,
                        seed=seed,
                        outer_fold=split.outer_fold,
                        policy=config.evaluation.undetermined_policy,
                    )
                )
        replaced = {(r.outer_fold, r.seed) for r in runs}
        for name, new_rows in (
            (f"{bid}_act_candidates.csv", act_candidate_rows(runs)),
            (f"{bid}_act_selected.csv", act_selected_rows(runs)),
        ):
            rows = _merge_rows(directory / name, new_rows, replaced)
            rows.sort(key=lambda r: (int(r["seed"]), int(r["outer_fold"]),
                                     int(r.get("step", 0) or 0)))
            write_rows(directory / name, rows)
        out[bid] = runs
    return out


def evaluate_stage(
    config: PilotConfig,
    inputs: Inputs,
    answerer: LLMClient,
    *,
    costs: Sequence[BoundaryCost],
    extra_stats_clients: Sequence[LLMClient] = (),
) -> list[BoundaryEvaluation]:
    """Answers on the outer folds, baselines, summaries, CSVs and the report."""
    validity = load_content_validity(config.content_validity)
    evaluations = []
    out_dir = evaluation_dir(config)
    for bid, items in inputs.items.items():
        boundary = config.boundary(bid)
        folds = sorted({i.outer_fold for i in items})
        candidates, notes = build_candidates(
            boundary, inputs.manual[bid], config.controls,
            read_act_selected(act_dir(config), bid), folds,
        )
        predictions = [
            predict(answerer, c, items, config.evaluation.undetermined_policy, bid)
            for c in candidates
        ]
        lexical = config.evaluation.lexical
        baselines = [tfidf_cart(items, lexical), tfidf_rulefit(items, lexical),
                     source_only(items)]
        ev = summarise_boundary(
            boundary, items, predictions, baselines, config.evaluation,
            validity_scores=validity,
            max_logical_ops=config.act.max_logical_ops,
            max_question_chars=config.act.max_question_chars,
            notes=notes,
        )
        write_item_csv(out_dir / f"{bid}_items.csv", ev)
        write_question_csv(out_dir / f"{bid}_questions.csv", ev)
        write_source_csv(out_dir / f"{bid}_sources.csv", ev)
        evaluations.append(ev)
    stats = total_stats([answerer, *extra_stats_clients])
    report = render_report(config, evaluations, costs, stats, act_dir(config))
    (config.output_dir / "report.md").write_text(report, encoding="utf-8")
    return evaluations


def validity_template(config: PilotConfig, inputs: Inputs) -> Path:
    """The template of content_validity.csv, with every question of every candidate."""
    rows = []
    for bid, items in inputs.items.items():
        boundary = config.boundary(bid)
        folds = sorted({i.outer_fold for i in items})
        candidates, _ = build_candidates(
            boundary, inputs.manual[bid], config.controls,
            read_act_selected(act_dir(config), bid), folds,
        )
        for c in candidates:
            if c.kind == "control":
                continue
            rows += [(bid, c.candidate_id, q) for q in c.distinct_questions]
    path = config.output_dir / "content_validity_template.csv"
    write_validity_template(path, rows)
    return path


def write_estimate(config: PilotConfig, costs: Sequence[BoundaryCost]) -> Path:
    path = config.output_dir / "cost_estimate.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Cost estimate", "", format_costs(list(costs)), ""]
    for c in costs:
        lines.append(f"## {c.boundary}")
        lines.append("")
        lines.append(
            f"ACT answers: {c.act_answers_with_cache:,} with cache "
            f"({c.act_answers_without_cache:,} without); evaluation answers: "
            f"{c.eval_answers_with_cache:,} ({c.n_fixed_questions} fixed questions plus the "
            f"ACT seeds); optimizer: {c.optimizer_min:,} to {c.optimizer_max:,}."
        )
        lines.append("")
        lines.append("| outer fold | inner train | inner validation | test |")
        lines.append("|---|---|---|---|")
        for f in c.per_fold:
            lines.append(
                f"| {f['outer_fold']} | {f['inner_train']} | {f['inner_val']} | {f['test']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_all(
    config: PilotConfig,
    *,
    backends: dict[str, ChatBackend] | None = None,
    boundary_ids: Sequence[str] | None = None,
    max_calls: int | None = None,
    sleep: Callable[[float], None] | None = None,
) -> list[BoundaryEvaluation]:
    """Estimate, ACT stage and evaluation in one go."""
    inputs = load_inputs(config, boundary_ids)
    costs = estimate(config, inputs)
    write_estimate(config, costs)
    check_budget(costs, max_calls)
    answerer, optimizer = clients(config, backends, sleep)
    run_act_stage(config, inputs, answerer, optimizer)
    evaluations = evaluate_stage(config, inputs, answerer, costs=costs,
                                 extra_stats_clients=[optimizer])
    validity_template(config, inputs)
    return evaluations
