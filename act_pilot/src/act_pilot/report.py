"""The CSV outputs and the Markdown report."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .act import ActRun
from .config import PilotConfig
from .costs import BoundaryCost, format_costs
from .evaluate import BoundaryEvaluation, SourceRow


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _signed(x: float | None) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return f"{x:+.3f}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_item_csv(path: Path, ev: BoundaryEvaluation) -> None:
    """One row per item and candidate (questions and baselines)."""
    rows = []
    for cp in ev.predictions:
        for k, item in enumerate(ev.items):
            rows.append(
                {
                    "boundary": ev.boundary.id,
                    "candidate_id": cp.candidate.candidate_id,
                    "kind": cp.candidate.kind,
                    "item_id": item.item_id,
                    "source": item.source,
                    "outer_fold": item.outer_fold,
                    "gold": item.gold,
                    "label": item.label,
                    "question": cp.candidate.questions[item.outer_fold],
                    "answer": cp.answers[k].value,
                    "prediction": cp.predictions[k],
                    "correct": int(cp.predictions[k] == item.label),
                }
            )
    for b in ev.baselines:
        for k, item in enumerate(ev.items):
            rows.append(
                {
                    "boundary": ev.boundary.id,
                    "candidate_id": b.result.name,
                    "kind": "baseline",
                    "item_id": item.item_id,
                    "source": item.source,
                    "outer_fold": item.outer_fold,
                    "gold": item.gold,
                    "label": item.label,
                    "question": "",
                    "answer": "",
                    "prediction": b.result.predictions[k],
                    "correct": int(b.result.predictions[k] == item.label),
                }
            )
    _write_csv(path, rows)


def write_question_csv(path: Path, ev: BoundaryEvaluation) -> None:
    """One row per candidate and baseline, with the global metrics and the acceptance."""
    rows = []
    for s in ev.questions:
        acc = s.acceptance
        rows.append(
            {
                "boundary": ev.boundary.id,
                "candidate_id": s.candidate.candidate_id,
                "kind": s.candidate.kind,
                "questions": " || ".join(
                    f"fold {f}: {q}" for f, q in sorted(s.candidate.questions.items())
                ) if s.candidate.kind == "act" else s.candidate.questions[
                    min(s.candidate.questions)],
                "n": s.n,
                "balanced_accuracy": s.ba,
                "recall_positive": s.recall_positive,
                "recall_negative": s.recall_negative,
                "per_fold_ba": "; ".join(f"{f}:{fmt(v)}" for f, v in s.per_fold_ba.items()),
                "n_undetermined": s.n_undetermined,
                "n_invalid": s.n_invalid,
                "delta_vs_original": s.difference.value,
                "ci_low": s.difference.low,
                "ci_high": s.difference.high,
                "mcnemar_b": s.mcnemar.b,
                "mcnemar_c": s.mcnemar.c,
                "mcnemar_p": s.mcnemar.p_value,
                "answer_yes_rate": s.lexical.yes_rate,
                "answer_lexical_auc": s.lexical.auc,
                "answer_yes_terms": ", ".join(s.lexical.positive_terms),
                "answer_no_terms": ", ".join(s.lexical.negative_terms),
                "constraint_violations": "; ".join(s.violations),
                "acceptance": acc.status if acc else "",
                "criterion_i": acc.criteria["i"].status if acc else "",
                "criterion_ii": acc.criteria["ii"].status if acc else "",
                "criterion_iii": acc.criteria["iii"].status if acc else "",
                "acceptance_detail": acc.summary() if acc else "",
            }
        )
    for b in ev.baselines:
        rows.append(
            {
                "boundary": ev.boundary.id,
                "candidate_id": b.result.name,
                "kind": "baseline",
                "questions": b.result.description,
                "n": len(ev.items),
                "balanced_accuracy": b.ba,
                "recall_positive": b.recall_positive,
                "recall_negative": b.recall_negative,
            }
        )
    _write_csv(path, rows)


def _source_dict(boundary: str, row: SourceRow) -> dict:
    return {
        "boundary": boundary,
        "candidate_id": row.name,
        "source": row.source,
        "n_positive": row.n_positive,
        "n_negative": row.n_negative,
        "evaluable": int(row.evaluable),
        "balanced_accuracy": row.ba,
        "recall_positive": row.recall_positive,
        "recall_negative": row.recall_negative,
        "delta_vs_original": row.difference.value if row.difference else "",
        "ci_low": row.difference.low if row.difference else "",
        "ci_high": row.difference.high if row.difference else "",
        "mcnemar_p": row.mcnemar.p_value if row.mcnemar else "",
    }


def write_source_csv(path: Path, ev: BoundaryEvaluation) -> None:
    rows = [_source_dict(ev.boundary.id, r) for s in ev.questions for r in s.sources]
    rows += [_source_dict(ev.boundary.id, r) for b in ev.baselines for r in b.sources]
    _write_csv(path, rows)


def act_candidate_rows(runs: Sequence[ActRun]) -> list[dict]:
    """All candidates of all ACT runs."""
    return [row for run in runs for row in run.as_rows()]


def act_selected_rows(runs: Sequence[ActRun]) -> list[dict]:
    """The selected question of each ACT run."""
    selected = []
    for run in runs:
        best = next(c for c in run.candidates if c.selected)
        selected.append(
            {
                "boundary": run.boundary,
                "outer_fold": run.outer_fold,
                "seed": run.seed,
                "selected_step": run.selected_step,
                "val_ba": best.val_ba,
                "train_ba": best.train_ba,
                "n_inner_train": run.n_inner_train,
                "n_inner_val": run.n_inner_val,
                "n_rejected_steps": sum(c.status == "rejected" for c in run.candidates),
                "question": run.selected_question,
            }
        )
    return selected


def write_rows(path: Path, rows: list[dict]) -> None:
    _write_csv(path, rows)


def read_act_selected(directory: Path, boundary_id: str) -> dict[int, dict[int, str]]:
    """Seed -> outer fold -> selected question, from the ACT stage output."""
    path = directory / f"{boundary_id}_act_selected.csv"
    if not path.exists():
        return {}
    out: dict[int, dict[int, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(int(row["seed"]), {})[int(row["outer_fold"])] = row["question"]
    return out


def read_act_summary(directory: Path, boundary_id: str) -> list[dict]:
    path = directory / f"{boundary_id}_act_selected.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    def esc(cell: str) -> str:
        return str(cell).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return lines


def render_report(
    config: PilotConfig,
    evaluations: Sequence[BoundaryEvaluation],
    costs: Sequence[BoundaryCost],
    call_stats: dict[str, dict[str, int]],
    act_dir: Path,
) -> str:
    """The Markdown report of the evaluation stage."""
    ev_cfg = config.evaluation
    lines = [
        "# ACT pilot report",
        "",
        f"Generated on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`{config.path.name}`.",
        "",
        "- Original questions: the critical questions of the schemes version "
        f"{', '.join(sorted({b.schemes_version for b in config.boundaries.values()}))}, "
        "read from the scheme files.",
        f"- Answerer: `{config.llm.answerer.model}`; optimizer: `{config.llm.optimizer.model}`; "
        f"temperature {config.llm.temperature}.",
        f"- ACT: depth 1, {config.act.steps} steps, m = {config.act.examples_per_group} "
        f"examples per group, seeds {list(config.act.seeds)}, selection on an inner "
        f"validation share of {config.act.validation_fraction:.0%} (never on training).",
        f"- Bootstrap: {ev_cfg.bootstrap} paired, class-stratified resamples, "
        f"{ev_cfg.ci_level:.0%} percentile intervals; McNemar exact test.",
        f"- 'cannot determine' and unreadable answers count as the "
        f"{ev_cfg.undetermined_policy} class; their numbers are reported per question.",
        *(
            [f"- Sensitivity: the evaluation is repeated on the same answers with them counted "
             f"as the {', '.join(ev_cfg.sensitivity_policies)} class; the ACT questions stay "
             f"those selected under the {ev_cfg.undetermined_policy} policy."]
            if ev_cfg.sensitivity_policies else []
        ),
        f"- A source is evaluable with at least {ev_cfg.min_items_per_class_in_source} items "
        f"per class.",
        "",
        "## Calls",
        "",
        "Estimate made before the run:",
        "",
        format_costs(list(costs)),
        "",
        "Calls made by this process (the ACT stage included, when it ran in the same "
        "process; cache hits cost nothing):",
        "",
    ]
    lines += _md_table(
        ["role", "requests", "cache hits", "server calls", "retries", "failures"],
        [
            [role, str(s["requests"]), str(s["cache_hits"]), str(s["backend_calls"]),
             str(s["retries"]), str(s["failures"])]
            for role, s in call_stats.items()
        ],
    )

    for ev in evaluations:
        b = ev.boundary
        lines += [
            "",
            f"## {b.id}: {b.scheme}, {b.positive} (yes) against {b.negative} (no)",
            "",
            f"Original question: {b.seed_cq} of `{b.scheme}`, schemes version "
            f"{b.schemes_version}: {b.seed_question}",
            "",
            "### Data",
            "",
        ]
        lines += _md_table(
            ["source", f"{b.positive}", f"{b.negative}", "evaluable"],
            [
                [s, str(pos), str(neg), "yes" if s in ev.evaluable_sources else "no"]
                for s, (pos, neg) in ev.sources.items()
            ],
        )
        if not ev.evaluable_sources:
            lines += [
                "",
                "No source has enough items of both classes: the within-source analysis is "
                "descriptive only, and the boundary cannot be separated from the source of "
                "the items with these data.",
            ]
        lines += ["", "### Questions", ""]
        rows = []
        for s in ev.questions:
            lines_q = s.candidate
            rows.append(
                [
                    lines_q.candidate_id,
                    lines_q.kind,
                    fmt(s.ba),
                    fmt(s.recall_positive),
                    fmt(s.recall_negative),
                    f"{_signed(s.difference.value)} [{_signed(s.difference.low)}, "
                    f"{_signed(s.difference.high)}]" if lines_q.kind != "original" else "",
                    f"{s.mcnemar.b}/{s.mcnemar.c}, p={fmt(s.mcnemar.p_value)}"
                    if lines_q.kind != "original" else "",
                    f"{s.n_undetermined}/{s.n_invalid}",
                    fmt(s.lexical.auc),
                    s.acceptance.status if s.acceptance else "",
                ]
            )
        lines += _md_table(
            ["candidate", "kind", "BA", f"recall {b.positive}", f"recall {b.negative}",
             "delta vs original [CI]", "McNemar b/c", "undet./invalid",
             "lexical AUC of the answer", "acceptance"],
            rows,
        )
        lines += ["", "Full text of the questions:", ""]
        for s in ev.questions:
            c = s.candidate
            if c.kind == "act":
                lines.append(f"- `{c.candidate_id}`:")
                lines += [f"  - fold {f}: {q}" for f, q in sorted(c.questions.items())]
            else:
                lines.append(f"- `{c.candidate_id}`: {c.questions[min(c.questions)]}")
            if s.violations:
                lines.append(f"  - constraint warnings: {'; '.join(s.violations)}")

        lines += ["", "### Baselines", ""]
        lines += _md_table(
            ["baseline", "BA", f"recall {b.positive}", f"recall {b.negative}", "description"],
            [
                [x.result.name, fmt(x.ba), fmt(x.recall_positive), fmt(x.recall_negative),
                 x.result.description]
                for x in ev.baselines
            ],
        )
        lp = ev.label_predictability
        lines += [
            "",
            f"TF-IDF + logistic regression predicting the gold label: out-of-fold AUC "
            f"{fmt(lp.auc)}; terms for {b.positive}: {', '.join(lp.positive_terms)}; "
            f"for {b.negative}: {', '.join(lp.negative_terms)}.",
            "",
            "`source_only` is reported for context and is not part of the acceptance "
            "criterion.",
        ]

        lines += ["", "### Acceptance", ""]
        for s in ev.questions:
            if not s.acceptance:
                continue
            lines.append(f"- `{s.candidate.candidate_id}`: **{s.acceptance.status}**")
            for key, crit in s.acceptance.criteria.items():
                lines.append(f"  - ({key}) {crit.status}: {crit.detail}")

        lines += ["", "### Per source", ""]
        src_rows = []
        for s in ev.questions:
            for r in s.sources:
                src_rows.append(_source_md(r))
        for x in ev.baselines:
            for r in x.sources:
                src_rows.append(_source_md(r))
        lines += _md_table(
            ["candidate", "source", f"n {b.positive}", f"n {b.negative}", "evaluable", "BA",
             f"recall {b.positive}", f"recall {b.negative}", "delta vs original [CI]",
             "McNemar p"],
            src_rows,
        )

        lines += ["", "### Lexical predictability of the answers", ""]
        for s in ev.questions:
            lx = s.lexical
            lines.append(
                f"- `{s.candidate.candidate_id}`: yes rate {fmt(lx.yes_rate)}, AUC "
                f"{fmt(lx.auc)}; yes terms: {', '.join(lx.positive_terms) or 'n/a'}; "
                f"no terms: {', '.join(lx.negative_terms) or 'n/a'}"
            )

        for alt in ev.sensitivity:
            lines += [
                "",
                f"### Sensitivity: undetermined and unreadable answers as "
                f"{alt.undetermined_policy}",
                "",
                "Same answers and same ACT questions; only their mapping to a class changes. "
                "Acceptance here is computed with this mapping, for comparison only.",
                "",
            ]
            main = {s.candidate.candidate_id: s for s in ev.questions}
            lines += _md_table(
                ["candidate", "BA", f"BA ({ev.undetermined_policy})",
                 "delta vs original [CI]", "acceptance",
                 f"acceptance ({ev.undetermined_policy})"],
                [
                    [s.candidate.candidate_id, fmt(s.ba),
                     fmt(main[s.candidate.candidate_id].ba),
                     f"{_signed(s.difference.value)} [{_signed(s.difference.low)}, "
                     f"{_signed(s.difference.high)}]" if s.candidate.kind != "original" else "",
                     s.acceptance.status if s.acceptance else "",
                     main[s.candidate.candidate_id].acceptance.status
                     if main[s.candidate.candidate_id].acceptance else ""]
                    for s in alt.questions
                ],
            )
            lines += ["", "The lexical baselines do not depend on the answers: "
                      + ", ".join(f"{x.result.name} {fmt(x.ba)}" for x in alt.baselines
                                  if x.result.name != "source_only") + "."]

        act_rows = read_act_summary(act_dir, b.id)
        lines += ["", "### ACT runs", ""]
        if act_rows:
            lines += _md_table(
                ["seed", "fold", "selected step", "val BA", "train BA", "rejected steps",
                 "question"],
                [
                    [r["seed"], r["outer_fold"], r["selected_step"], fmt(float(r["val_ba"])),
                     fmt(float(r["train_ba"])), r["n_rejected_steps"], r["question"]]
                    for r in sorted(act_rows, key=lambda r: (int(r["seed"]),
                                                             int(r["outer_fold"])))
                ],
            )
        else:
            lines.append("No ACT run found for this boundary.")
        if ev.notes:
            lines += ["", "### Notes", ""] + [f"- {n}" for n in ev.notes]
    lines.append("")
    return "\n".join(lines)


def _source_md(r: SourceRow) -> list[str]:
    diff = (
        f"{_signed(r.difference.value)} [{_signed(r.difference.low)}, "
        f"{_signed(r.difference.high)}]" if r.difference else ""
    )
    return [
        r.name, r.source, str(r.n_positive), str(r.n_negative),
        "yes" if r.evaluable else "no", fmt(r.ba), fmt(r.recall_positive),
        fmt(r.recall_negative), diff, fmt(r.mcnemar.p_value) if r.mcnemar else "",
    ]
