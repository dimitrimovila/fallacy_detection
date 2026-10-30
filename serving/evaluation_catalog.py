"Build a small reading guide from saved aggregate tables, without fitting or scoring."

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path

FOLDERS = {
    "decision_v2": "Round 1: schemes 1.1, five models, A0–A5 and zero-shot.",
    "round2": "Round 2: learned combinations, routing, trees, S-soft and D+Q.",
    "exploratory_aggregators_v3": (
        "Exploratory A0–A5 evaluation on schemes 2.0; not a new preregistered family."
    ),
    "analyses_v1": (
        "Cross-round breakdowns, CQ/human agreement, uncertainty and provenance. "
        "Reuses saved experiments."
    ),
    "b3_schemes2_diagnostic_20261029": (
        "Completed B3@2.0 versus B1.hard@2.0 diagnostic: full trees, paired "
        "sensitivities and qualified semantic review."
    ),
}

QUESTIONS = [
    (
        "Overall round-1 results",
        "decision_v2/main_table.csv",
        "A0–A5 and five-model zero-shot; deciding.csv contains paired tests.",
    ),
    (
        "Overall round-2 results",
        "round2/main_table.csv",
        "B1–B4, controls and exploratory variants; deciding.csv contains the original tests.",
    ),
    (
        "Aggregators on schemes 2.0",
        "exploratory_aggregators_v3/main_table.csv",
        "A1, A3 and A4 are exploratory. The separate B3@2.0 diagnostic is complete.",
    ),
    (
        "B3@2.0 diagnostic result and full rules",
        "b3_schemes2_diagnostic_20261029/review.md",
        "Macro-F1 0.2853 versus B1.hard 0.3893; paired delta -0.1040 "
        "[-0.1539, -0.0019]. Full trees and exact replay are retained.",
    ),
    (
        "D+Q and matched direct baselines",
        "round2/dq_main_table.csv",
        "Use dq_deciding.csv for H_DQ and dq_exploratory.csv for other contrasts.",
    ),
    (
        "CQ correctness and comparison with human answers",
        "analyses_v1/cq_level.csv",
        "CQ pairs, not final verdicts. Check population, scheme version and denominator.",
    ),
    (
        "Final verdicts obtained from human CQ paths",
        "analyses_v1/human_path.csv",
        "Different from CQ agreement; coverage and exclusions matter.",
    ),
    (
        "Per-class and source/scheme breakdowns",
        "analyses_v1/per_class.csv",
        (
            "Filter evaluation, system, subset and class before reading; "
            "main_table.csv has subgroup summaries."
        ),
    ),
    (
        "Abstention and confidence",
        "analyses_v1/abstention.csv",
        "Summary first; *_curves.csv and abstention_thresholds.csv contain detailed curves.",
    ),
    (
        "D+Q explanation consistency",
        "round2/dq_consistency.csv",
        (
            "Agreement of declared verdict and diagram traversal; not accuracy or "
            "causal faithfulness."
        ),
    ),
    (
        "Integrity, inputs and pending work",
        "analyses_v1/execution_audit.json",
        "Exact routing/conformal item exports remain pending; empty tables are not zero results.",
    ),
]

SELECTED = [
    ("decision_v2/main_table.csv", "A0:qwen3_8_27b", "1.1", "Qwen · fixed diagram"),
    ("decision_v2/main_table.csv", "zeroshot:qwen3_8_27b", "1.1", "Qwen · original zero-shot"),
    ("decision_v2/main_table.csv", "A5", "1.1", "Qwen · judge with justifications"),
    ("round2/main_table.csv", "B1", "1.1", "B1 · learned CQ combination"),
    ("round2/main_table.csv", "B2", "1.1", "B2 · routing"),
    ("round2/main_table.csv", "B3", "1.1", "B3 · learned trees"),
    (
        "exploratory_aggregators_v3/main_table.csv",
        "A0:qwen3_8_27b",
        "2.0",
        "Qwen · revised fixed diagram",
    ),
    (
        "exploratory_aggregators_v3/main_table.csv",
        "A1",
        "2.0",
        "A1 · reordered diagram (exploratory)",
    ),
    (
        "exploratory_aggregators_v3/main_table.csv",
        "A3",
        "2.0",
        "A3 · logistic regression (exploratory)",
    ),
    (
        "exploratory_aggregators_v3/main_table.csv",
        "A4",
        "2.0",
        "A4 · equal-weight models (exploratory)",
    ),
    ("round2/main_table.csv", "A2.pruned", "2.0", "Qwen · soft traversal, S-soft reference"),
    ("round2/main_table.csv", "B4", "2.0", "Qwen · S-soft"),
    ("round2/dq_main_table.csv", "zeroshot.rerun:qwen3_8_27b", "1.1", "Qwen · matched zero-shot"),
    ("round2/dq_main_table.csv", "DQ:qwen3_8_27b", "1.1", "Qwen · D+Q"),
    ("round2/dq_main_table.csv", "zeroshot:qwen3_8_27b@2.0", "2.0", "Qwen · matched zero-shot"),
    ("round2/dq_main_table.csv", "DQ:qwen3_8_27b@2.0", "2.0", "Qwen · D+Q"),
]


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def purpose(name):
    specific = {
        "main_table.csv": "Accuracy, macro F1, coverage and intervals by system/population",
        "main_table_without_nlas.csv": "Verdict scores excluding NLAS; a different population",
        "diagnostic.csv": "Round-1 diagnostic contrasts and controls",
        "exploratory.csv": "Additional exploratory contrasts, separate from deciding tests",
        "human_agreement.csv": "Historical human/model agreement: units and populations",
        "human_path.csv": "Final verdict scores from human CQ paths, including coverage/stops",
        "human_direct.csv": "Optional blind human direct verdicts; inspect pending status first",
        "cq_metrics.csv": "CQ-level agreement, probability quality and support",
        "cq_agreement.csv": "Agreement between models on CQ answers",
        "cq_calibration.csv": "CQ probability calibration",
        "cq_report.md": "Definitions and narrative for the CQ diagnostic CSVs",
        "cq_level.csv": "CQ correctness by model, scheme, question and population",
        "cq_level_differences.csv": "Paired CQ correctness contrasts with uncertainty",
        "abstention.csv": "Aggregate selective-prediction measures, including AUGRC/AURC",
        "abstention_composition.csv": "Which classes/schemes/sources are rejected at coverage cuts",
        "difficulty.csv": "Confidence/rejection in relation to historical difficulty ratings",
        "conformal.csv": "Conformal summary; distinguish computed from pending outputs",
        "conformal_items.csv": "Exact item-level conformal sets; currently pending",
        "source_conformal.csv": "Original saved conformal aggregates, not exact item exports",
        "integrity.csv": "Completeness, validity and input-integrity diagnostics",
        "stage1.csv": "Scheme-recognition scores; distinct from gold-scheme verdict evaluation",
        "rare_classes.csv": "Rare-class recall and support; full-test cutoff below 20",
        "frequent_classes.csv": "Post-results scores: classes with support at least 10",
        "substitution.csv": "Exploratory replacement of model CQ answers with human CQ answers",
        "ssoft_changes.csv": "S-soft departures from isolated-CQ controls",
        "uncertainty_disagreement.csv": "Optional uncertainty/disagreement; inspect readiness",
        "dq_consistency.csv": "Declared D+Q verdict versus its CQ-derived diagram verdict",
        "dq_cq_level.csv": "D+Q versus isolated-CQ correctness and Brier score",
        "dq_cq_changes.csv": "How CQ answers change under D+Q",
        "dq_replication.csv": "Matched direct-baseline replication/control diagnostics",
        "dq_departures.csv": "Where D+Q declared verdicts depart from diagram traversal",
    }
    if name in specific:
        return specific[name]
    if name.startswith("manifest") or name.endswith("audit.json"):
        return "Provenance, saved inputs and validation status"
    if name.startswith("summary"):
        return "Detailed presentation of saved tables; not an additional experiment"
    if "choices" in name or "orders" in name:
        return "Fit choices, folds and reproducibility detail"
    if "verdicts" in name or name == "confidence.csv" or name == "cq_combined.csv":
        return "Item-level records; read only after selecting a system and population"
    if "curve" in name or "threshold" in name:
        return "Detailed curve points; start with the corresponding aggregate table"
    if "deciding" in name or name == "comparisons.csv" or "differences" in name:
        return "Paired contrasts and uncertainty; preserve families and read gates"
    if name == "per_class.csv":
        return "Per-class scores across systems and subgroups; filter first"
    if name.startswith("dq_"):
        return "D+Q-specific results; round2 is the source, analyses_v1 adds provenance/views"
    return "Saved diagnostic/aggregate table; definitions and status are in the manifest/report"


def build(root):
    root = Path(root).resolve()
    blocks = [
        "# Evaluation results — start here",
        (
            "Read this guide first. Open only the table needed for the question, then "
            "filter by system, scheme version and population. Do not read every "
            "summary or item-level CSV together."
        ),
        "## The five current collections",
        table(
            ["Collection", "Scope", "Reading guide"],
            [
                [name, scope, f"[Files and detailed report]({name}/README.md)"]
                for name, scope in FOLDERS.items()
            ],
        ),
        (
            "The first three collections contain evaluations of model outputs. "
            "analyses_v1 adds analyses of saved results; it is not another model run. "
            "Historical audits and the alternative round-2 snapshot are in "
            "[_archive](_archive/README.md)."
        ),
        "## Find the answer",
        table(
            ["Question", "Start with", "Interpretation / next file"],
            [[q, f"[{p}]({p})", note] for q, p, note in QUESTIONS],
        ),
        "## Selected results, not the complete leaderboard",
        (
            "Gold scheme supplied, fine fallacy labels plus Good Argumentation. Most "
            "rows use 601 test items; N is explicit below. These are saved point "
            "estimates, not new calculations. References differ across experiments: "
            "do not compare S-soft with the hard diagram or D+Q with the original "
            "unmatched zero-shot. Multi-model systems combine the five models; other "
            "selected rows use Qwen. See deciding/exploratory tables for paired "
            "intervals and inference."
        ),
    ]
    cache, rows = {}, []
    for path, system, version, description in SELECTED:
        if path not in cache:
            with (root / path).open(encoding="utf-8", newline="") as f:
                cache[path] = list(csv.DictReader(f))
        found = [r for r in cache[path] if r["system"] == system]
        if len(found) != 1:
            raise ValueError(f"Expected one saved row: {path}, {system}")
        row = found[0]
        rows.append(
            [
                description,
                version,
                row["n_items"],
                row["accuracy"],
                row["macro_f1"],
                f"[{system}]({path})",
            ]
        )
    blocks += [
        table(["Method", "Schemes", "N", "Accuracy", "Macro F1", "Saved source / system"], rows),
        "## What the comparisons support",
        (
            "The primary round-2 comparisons do not establish improvements for B1, B2 "
            "or B3. S-soft is worse than its matched A2.pruned reference; further "
            "S-soft experiments were discontinued. D+Q does not demonstrate a verdict "
            "improvement over its matched direct baseline. An interval containing "
            "zero does not establish equivalence. Schemes 2.0 and later aggregator "
            "additions are exploratory after earlier results. CQ correctness, final "
            "verdict quality and explanation consistency are separate outcomes."
        ),
        "## Reading rules and pending work",
        (
            "Use decision_v2 and round2 as the original evaluation sources. "
            "analyses_v1 provides extra breakdowns and records imported sources; it "
            "does not replace those originals. The completed B3@2.0 diagnostic "
            "is in b3_schemes2_diagnostic_20261029; it does not improve its matched "
            "baseline. See ../docs/research_findings.md for scientific conclusions. "
            "Exact B2 routing and conformal item exports are pending because replay "
            "has not passed saved-output verification. The alternative round-2 "
            "snapshot must not replace original confidence values. New human "
            "direct-verdict inputs remain pending in analyses_v1."
        ),
        (
            "CSV files are the complete machine-readable results. summary.html files "
            "are detailed views of those same results. The analyses_v1 HTML contains "
            "labelled 25-row previews, not every row. Its previous complete HTML is "
            "preserved in the archive. Missing or undefined values are not zeros. "
            "Manifests remain the authoritative records for parameters, provenance "
            "and table status."
        ),
        "## Rebuild this guide",
        (
            "From the repository root: `python serving/evaluation_catalog.py`. This "
            "reads saved aggregate tables only; it does not run models, fit systems "
            "or recompute metrics. Detailed analysis HTML: `argfallacy report summary "
            "evaluations/analyses_v1 --format analysis`."
        ),
    ]
    text = "\n\n".join(blocks) + "\n"
    (root / "README.md").write_text(text, encoding="utf-8")
    # Render the same content, keeping a single editorial source for both entry points.
    import re

    def inline(value):
        value = escape(value)
        return re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)

    html = []
    for block in blocks:
        if block.startswith("#"):
            depth = len(block) - len(block.lstrip("#"))
            html.append(f"<h{depth}>{inline(block[depth:].strip())}</h{depth}>")
        elif block.startswith("|"):
            lines = block.splitlines()
            grid = [[v.strip() for v in line.strip("|").split("|")] for line in lines]
            html.append(
                "<div class=scroll><table><thead><tr>"
                + "".join(f"<th>{inline(v)}</th>" for v in grid[0])
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join(f"<td>{inline(v)}</td>" for v in row) + "</tr>"
                    for row in grid[2:]
                )
                + "</tbody></table></div>"
            )
        else:
            html.append(f"<p>{inline(block)}</p>")
    (root / "index.html").write_text(
        (
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta '
            'name="viewport" content="width=device-width, '
            'initial-scale=1"><title>Evaluation '
            "results</title><style>body{font:17px/1.6 "
            "system-ui;color:#163142;background:#f4f7f8;margin:0}main{max-width:1150p"
            "x;margin:auto;padding:36px}h1,h2{line-height:1.2}h2{margin-top:42px}a{co"
            "lor:#076a78}table{border-collapse:collapse;background:white;width:100%;f"
            "ont-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid "
            "#dce4e8;vertical-align:top}.scroll{overflow:auto}th{background:#e7f0f1}<"
            "/style><main>"
        )
        + "".join(html)
        + "</main></html>",
        encoding="utf-8",
    )
    for name, scope in FOLDERS.items():
        folder = root / name
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "README.md")
        introduction = (
            "Start with [the result review](review.md), [the summary](summary.md) "
            "and [the frozen protocol](protocol.md). Complete trees, rules and "
            "CSV tables remain available locally.\n\n"
            if name == "b3_schemes2_diagnostic_20261029"
            else "Start with the main table and paired comparisons; open the "
            "[detailed report](summary.html) only for its explanations. "
            "Complete CSV tables are preserved.\n\n"
        )
        guide = (
            f"# {name}\n\n[Back to the results guide](../README.md)\n\n{scope}\n\n"
            + introduction
        )
        guide += table(
            ["File", "Purpose"], [[f"[{p.name}]({p.name})", purpose(p.name)] for p in files]
        )
        existing = folder / "README.md"
        # Keep the original exploratory methodological note verbatim in a separate section.
        marker = "## Original methodological note\n\n"
        if name == "exploratory_aggregators_v3" and existing.exists():
            old = existing.read_text(encoding="utf-8")
            old = old.split(marker, 1)[-1] if marker in old else old
            guide += "\n\n" + marker + old.rstrip()
        existing.write_text(guide + "\n", encoding="utf-8")
    return root / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / "evaluations"
    )
    print(build(parser.parse_args().root))
