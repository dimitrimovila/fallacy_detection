"""Present the saved integrated-reference evaluation by aggregator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from html import escape
from pathlib import Path
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[1]
POPULATIONS = {
    "test_601": (
        "All 601 test items",
        "The updated reference combines 406 app verdicts and 195 original dataset labels.",
    ),
    "audit_405": (
        "405 audit items",
        "The same 405 items selected by the original-label audit are scored under both references. "
        "The updated reference uses the app verdicts for these items.",
    ),
}
FAMILIES = {
    "zeroshot": ("Zero-shot", "Direct verdict with the reference scheme and label menu supplied."),
    "A0": ("A0 · Fixed diagram", "The diagram follows one model’s hard answers."),
    "A1": ("A1 · Reordered diagram", "The diagram uses an order selected on training folds."),
    "A2": ("A2 · Soft diagram", "Probabilistic traversal of the diagram."),
    "A3": ("A3 · Learned verdict", "A classifier maps the critical-question answers to a verdict."),
    "A4": ("A4 · Model average", "The five models’ answers are combined with equal weights."),
    "A5": ("A5 · LLM judge", "Qwen judges the answers, with or without their justifications."),
    "B1": ("B1 · Combined answers", "Multiple models’ answers are combined before traversal."),
    "B2": ("B2 · Routing", "A saved router selects which available method gives the verdict."),
    "B3": ("B3 · Learned trees", "Per-scheme trees and pooled variants based on combined answers."),
    "B4": ("B4 · Sequential questions", "S-soft and its controls on schemes 2.0."),
    "DQ": ("D+Q · Direct verdict with questions", "Declared verdict and diagram-based variants."),
    "prior": ("Prior runs", "Earlier direct and predicted-scheme evaluations."),
    "human": (
        "Human CQ diagnostics",
        "Matched comparisons on the historical human-answer subsets.",
    ),
    "oracle": (
        "In-sample references",
        "Majority and answer-vector ceiling from historical labels.",
    ),
    "tree": ("Original tree diagnostic", "GOSDT and matched CART before the tie correction."),
}
LABELS = {
    "zeroshot": "Direct verdict",
    "zeroshot.rerun": "Matched direct rerun",
    "A0": "Fixed diagram",
    "A1": "Reordered diagram",
    "A2": "Soft traversal",
    "A2.pruned": "Soft traversal with pruning",
    "A3": "Logistic regression",
    "A4": "Equal-weight model average",
    "A5.with_justifications": "Judge with justifications",
    "A5.answers_only": "Judge with answers only",
    "B1": "CQ-supervised stacking, soft traversal",
    "B1.hard": "CQ-supervised stacking, hard traversal",
    "B2": "Diagram or direct verdict",
    "B2.dq": "Routing with D+Q",
    "B3": "Tree per scheme",
    "B4": "S-soft",
    "B4.path": "S-soft single path",
    "B4.nohistory": "S-soft without answer history",
    "DQ": "Declared verdict",
    "DQ.path": "Hard diagram path",
    "DQ.soft": "Soft diagram traversal",
    "pipeline": "Predicted-scheme pipeline",
    "zero-shot": "Direct verdict from text",
    "manual_human": "Fixed diagram on human answers",
    "human_logistic": "Logistic regression on human answers",
    "human_tree": "Tree on human answers",
    "majority": "Majority by scheme",
    "ceiling": "Ceiling by answer vector",
    "GOSDT": "GOSDT, original",
    "CART.binary": "Matched CART, original",
}
COHORTS = {
    "annotation_exclusion": "Restricted scheme subset",
    "scheme_correct": "Correctly predicted schemes",
    "human_complete476": "Complete human-answer subset",
    "human_resolved438": "Resolved human-path subset",
    "superseded_original_tree_run": "Superseded verdicts",
}
STATUS = {
    "no_historical_run": "No completed run",
    "not_recalculable": "Item verdicts unavailable",
    "different_endpoint": "Different evaluation outcome",
}

CSS = """
:root{color-scheme:light;--ink:#172b33;--muted:#4b636b;--line:#d5e0e2;--teal:#006458;
  --wash:#f3f7f7;--soft:#e8f5f1;font:16px/1.5 system-ui,sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink)}
main{width:min(1120px,100% - 36px);margin:0 auto;padding:36px 0 56px}
h1{font-size:clamp(1.85rem,4vw,2.7rem);line-height:1.15;letter-spacing:-.035em;margin:0}
h2{font-size:1.23rem;margin:0}h3{font-size:1.05rem;margin:0 0 .8rem}
p{margin:.45rem 0}.muted,small{color:var(--muted)}
a{color:var(--teal);text-underline-offset:3px}
:focus-visible{outline:3px solid #1678a0;outline-offset:3px}
[hidden]{display:none!important}
.eyebrow{font-size:.75rem;font-weight:750;letter-spacing:.09em;text-transform:uppercase;
  color:var(--teal);margin:0 0 .5rem}
.intro{max-width:78ch;margin:.65rem 0 1.15rem}
.links{display:flex;gap:.5rem 1.2rem;flex-wrap:wrap;font-size:.86rem}
.controls{background:white;border:1px solid var(--line);border-radius:12px;
  padding:15px 18px;margin:22px 0 15px}
fieldset{border:0;padding:0;margin:0;min-width:0}legend,label{font-weight:650;font-size:.85rem}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.tabs label{display:inline-flex;align-items:center;gap:6px;padding:7px 10px;cursor:pointer;
  border:1px solid var(--line);border-radius:7px}
.tabs label:has(:checked){background:var(--soft);color:var(--teal);border-color:var(--teal)}
input[type=radio]{accent-color:var(--teal)}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.filters label{display:flex;flex-direction:column;gap:4px}
select,input[type=search]{font:inherit;font-size:.9rem;max-width:100%;padding:6px 8px;
  background:white;color:var(--ink);border:1px solid #9bb2b6;border-radius:6px}
#filter-note{margin-top:8px;font-size:.86rem}
.population-title{margin-top:18px}
.family{margin:13px 0;background:white;border:1px solid var(--line);border-radius:10px}
.family header{padding:13px 16px}.family h3{margin:0}
.family header p{font-size:.87rem;margin:2px 0 0}
.scroll{overflow-x:auto;border-top:1px solid var(--line)}
table{width:100%;min-width:730px;border-collapse:collapse;font-size:.87rem}
th,td{padding:9px 12px;text-align:left;vertical-align:top}
thead th{font-weight:700;background:#edf3f3}
tbody tr+tr{border-top:1px solid #e5ecec}
tbody tr:nth-child(even){background:#f8fafa}
tbody th{font-weight:650;width:40%}td{font-variant-numeric:tabular-nums}
small{display:block;font-size:.75rem}code{font-size:.75rem;overflow-wrap:anywhere}
.score,.item-count,.coverage{white-space:nowrap}.score b{font-weight:750}
.delta{display:inline-block;margin-top:3px;border-radius:4px;padding:1px 5px;background:#edf1f2;
  color:#3f565e;font-weight:650;font-size:.75rem}
.gain{background:#e0f2ea;color:#075c42}.loss{background:#f9e8e2;color:#8b3822}
.tag{color:#774e13;font-weight:650}
summary{cursor:pointer}tbody details{font-size:.76rem;font-weight:450;margin-top:2px}
tbody details p{margin:.25rem 0}
footer{margin-top:25px;font-size:.86rem}
footer details{padding:10px 16px;background:white;border:1px solid var(--line);border-radius:8px}
footer li{margin:.5rem 0}
@media (max-width:760px){
  table{min-width:0}
  thead{position:absolute;clip-path:inset(50%);width:1px;height:1px;overflow:hidden}
  tbody{display:block}
  tbody tr{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));padding:6px 0}
  tbody th,.scheme{grid-column:1/-1;width:auto}
  th,td{padding:5px 11px}
  td::before{content:attr(data-label);display:block;color:var(--muted);
    font-size:.72rem;font-weight:650}
  .scheme::before{display:inline;margin-right:6px}
}
@media (max-width:540px){main{width:calc(100% - 20px);padding-top:22px}
  .controls,.family header{padding:11px}.filters label{flex:1 0 145px}}
"""

SCRIPT = """
const controls = document.querySelector('.controls');
const family = document.querySelector('#family');
const version = document.querySelector('#version');
const search = document.querySelector('#search');
const note = document.querySelector('#filter-note');
const empty = document.querySelector('#no-results');
const populations = document.querySelectorAll('[data-population]');
function refresh() {
  const selected = controls.querySelector('input[name="population"]:checked').value;
  const query = search.value.trim().toLowerCase();
  let count = 0;
  for (const population of populations) {
    population.hidden = population.dataset.population !== selected;
    if (population.hidden) continue;
    for (const section of population.querySelectorAll('[data-family]')) {
      let matches = 0;
      for (const row of section.querySelectorAll('[data-system]')) {
        row.hidden = !((!family.value || family.value === section.dataset.family)
          && (!version.value || version.value === row.dataset.version)
          && row.dataset.search.includes(query));
        if (!row.hidden) matches++;
      }
      section.hidden = matches === 0;
      count += matches;
    }
  }
  note.textContent = count + (count === 1 ? ' score row shown' : ' score rows shown');
  empty.hidden = count !== 0;
}
controls.addEventListener('change', refresh);
search.addEventListener('input', refresh);
refresh();
controls.hidden = false;
"""


def checked_csv(folder, manifest, name):
    raw = (folder / name).read_bytes()
    record = manifest["outputs"][name]
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"Saved file differs from its manifest: {name}")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    if len(rows) != record["rows"]:
        raise ValueError(f"Row count differs from its manifest: {name}")
    return rows


def group(meta):
    if meta["condition"].startswith("prior_"):
        return "prior"
    if meta["condition"] == "gold_scheme_human_diagnostic":
        return "human"
    if meta["role"] == "oracle":
        return "oracle"
    if meta["cohort"] == "superseded_original_tree_run":
        return "tree"
    return meta["method"].split(".")[0]


def grouped_scores(systems, scores, populations):
    expected = {
        (system_id, population, space)
        for system_id, meta in systems.items()
        for population in populations
        for space in meta["label_spaces"].split("|")
    }
    groups = defaultdict(list)
    for row in scores:
        key = row["system_id"], row["population"], row["label_space"]
        if key not in expected:
            raise ValueError(f"Unexpected or duplicate score row {key}")
        expected.remove(key)
        meta = systems[key[0]]
        for field in ("condition", "cohort", "role", "scheme_version"):
            if row[field] != meta[field]:
                raise ValueError(f"Conflicting {field} for {key}")
        if int(row["n_population"]) != populations[row["population"]]:
            raise ValueError(f"Population size differs from the manifest for {key}")
        n_scored, n_verdicts = (int(row[f]) for f in ("n_scored", "n_verdicts"))
        if not 0 < n_scored <= int(row["n_population"]) or not 0 <= n_verdicts <= n_scored:
            raise ValueError(f"Inconsistent denominators for {key}")
        if not math.isclose(float(row["coverage"]), n_verdicts / n_scored):
            raise ValueError(f"Inconsistent verdict coverage for {key}")
        for ref in ("historical", "integrated"):
            if not math.isclose(
                float(row[f"accuracy_{ref}"]), int(row[f"n_correct_{ref}"]) / n_scored
            ):
                raise ValueError(f"Inconsistent {ref} accuracy for {key}")
        for metric in ("accuracy", "macro_f1"):
            previous, current = (float(row[f"{metric}_{r}"]) for r in ("historical", "integrated"))
            delta = float(row[f"{metric}_delta_pp"])
            if not (0 <= previous <= 1 and 0 <= current <= 1):
                raise ValueError(f"Invalid {metric} for {key}")
            if not math.isclose(delta, 100 * (current - previous), abs_tol=1e-10):
                raise ValueError(f"Inconsistent {metric} change for {key}")
        groups[row["population"], group(meta)].append((row, meta))
    if expected:
        raise ValueError(f"Missing score rows {sorted(expected)[:5]}")
    return groups


def link(folder, path):
    relative = Path(os.path.relpath(ROOT / path, folder)).as_posix()
    return escape(quote(relative, safe="/"))


def change(raw):
    value = float(raw)
    sign = "gain" if value > 0 else "loss"
    if abs(value) < 0.005:
        return "", "0.00"
    return sign, f"{value:+.2f}".replace("-", "−")


def score_cell(row, metric):
    label = "Accuracy" if metric == "accuracy" else "Macro-F1"
    historical = f"{float(row[f'{metric}_historical']):.2%}"
    integrated = f"{float(row[f'{metric}_integrated']):.2%}"
    color, delta = change(row[f"{metric}_delta_pp"])
    return (
        f'<td class="score" data-metric="{metric}" data-label="{label}">'
        f'<span data-reference="historical">{historical}</span> → '
        f'<b data-reference="integrated">{integrated}</b><br>'
        f'<span data-reference="delta" class="delta {color}">{delta} pp</span></td>'
    )


def render_row(folder, row, meta, model_names):
    method = meta["method"]
    title = LABELS.get(method, method)
    models = [model_names.get(name, name) for name in meta["model"].split("+") if name]
    model_list = ", ".join(models)
    model_label = f"{len(models)} model inputs" if len(models) > 1 else model_list
    cohort = COHORTS.get(meta["cohort"], "")
    if row["label_space"] == "collapsed_symmetric":
        cohort = " · ".join(filter(None, (cohort, "Collapsed ad hominem labels")))
    system_id = escape(row["system_id"])
    scheme = escape(meta["scheme_version"])
    scheme_label = "Prior" if scheme == "prior" else scheme
    search_text = escape(" ".join((row["system_id"], title, model_list, cohort)).lower())
    source = link(folder, meta["source_file"])
    label = f'<small class="tag">{escape(cohort)}</small>' if cohort else ""
    return (
        f'<tr data-system="{system_id}" data-version="{scheme}" '
        f'data-space="{escape(row["label_space"])}" data-search="{search_text}">'
        f'<th scope="row">{escape(title)}{label}'
        f'<small title="{escape(model_list)}">{escape(model_label)}</small>'
        "<details><summary>Configuration and source</summary>"
        f"<p><code>{system_id}</code></p>"
        f'<p><a href="{source}">Saved verdict file</a> · {escape(meta["source_key"])}</p>'
        f"<p>{row['wrong_to_correct']} newly matching and {row['correct_to_wrong']} "
        "no longer matching the reference.</p></details></th>"
        f'<td class="scheme" data-label="Schemes">{scheme_label}</td>'
        f"{score_cell(row, 'accuracy')}{score_cell(row, 'macro_f1')}"
        '<td class="item-count" data-field="n_scored" data-label="Items scored">'
        f"{row['n_scored']} / {row['n_population']}</td>"
        '<td class="coverage" data-field="coverage" data-label="Verdict coverage">'
        f"{float(row['coverage']):.2%}"
        f"<small>{row['n_verdicts']} verdicts</small></td></tr>"
    )


def render_group(folder, family, rows, model_names):
    title, note = FAMILIES[family]
    rows = sorted(
        rows,
        key=lambda entry: (
            entry[1]["scheme_version"],
            entry[1]["method"] not in (family, "A5.with_justifications"),
        ),
    )
    return (
        f'<section class="family" data-family="{family}"><header><h3>{escape(title)}</h3>'
        f'<p class="muted">{escape(note)}</p></header>'
        f'<div class="scroll" tabindex="0" aria-label="{escape(title)} results"><table>'
        '<thead><tr><th scope="col">Configuration</th><th scope="col">Schemes</th>'
        '<th scope="col">Accuracy<small>Historical → Updated</small></th>'
        '<th scope="col">Macro-F1<small>Historical → Updated</small></th>'
        '<th scope="col">Items<small>Scored / population</small></th>'
        '<th scope="col">Coverage<small>Among scored items</small></th></tr></thead><tbody>'
        + "".join(render_row(folder, row, meta, model_names) for row, meta in rows)
        + "</tbody></table></div></section>"
    )


def render(folder, groups, unavailable, model_names, populations, manifest):
    selected = "audit_405" if "audit_405" in populations else "test_601"
    controls = "".join(
        f'<label><input type="radio" name="population" value="{p}"'
        f"{' checked' if p == selected else ''}> {escape(title)}</label>"
        for p, (title, _) in POPULATIONS.items()
        if p in populations
    )
    options = "".join(
        f'<option value="{family}">{escape(title)}</option>'
        for family, (title, _) in FAMILIES.items()
    )
    results = []
    for population, (title, note) in POPULATIONS.items():
        if population not in populations:
            continue
        if population == "test_601" and selected == "audit_405":
            note = (
                "The full set includes the 405 audit items and 196 other test items. "
                "The two references agree on all 196 other items."
            )
        sections = [
            render_group(folder, f, groups[population, f], model_names)
            for f in FAMILIES
            if groups[population, f]
        ]
        results.append(
            f'<article data-population="{population}" aria-label="{escape(title)}">'
            f'<h2 class="population-title">{escape(title)}</h2>'
            f'<p class="muted">{escape(note)}</p>{"".join(sections)}</article>'
        )
    excluded = "".join(
        f"<li><b>{escape(row['system_id'])}</b> · {escape(STATUS[row['status']])}. "
        f"{escape(row['reason'])}</li>"
        for row in unavailable
    )
    checks = manifest["reference_checks"]
    if selected == "audit_405":
        audit = manifest["original_label_audit"]
        changed = int(audit["n_labels_different_from_historical"])
        count = f"{changed} of the 405 audit labels"
    else:
        changed = int(checks["changed_app_labels"])
        count = f"{changed} of the {int(checks['verdict_origin_counts']['dumitru_app'])} app labels"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Results by aggregator</title><style>{CSS}</style></head>
<body><main>
<header><p class="eyebrow">Saved verdicts · Reference comparison</p>
<h1>Results by aggregator</h1>
<p class="intro">The same saved verdicts scored against the historical gold and
<strong>Dumitru’s updated reference</strong>. Select a population and an aggregator
to compare Accuracy and Macro-F1 side by side.</p>
<p><strong>{count}</strong> differ from the historical gold.</p>
<nav class="links" aria-label="Supporting files">
<a href="README.md">Interpretation and methods</a>
<a href="rankings.csv">Rankings</a>
<a href="main_table.csv">Full scores CSV</a>
<a href="per_scheme.csv">By scheme</a>
<a href="per_class.csv">By class</a>
<a href="#unavailable">Unavailable results</a>
</nav></header>
<div class="controls" hidden>
<fieldset><legend>Population</legend><div class="tabs">{controls}</div></fieldset>
<div class="filters">
<label>Aggregator<select id="family"><option value="">All aggregators</option>
{options}</select></label>
<label>Scheme version<select id="version"><option value="">All versions</option>
<option value="1.1">1.1</option><option value="2.0">2.0</option>
<option value="prior">Prior runs</option></select></label>
<label>Find a configuration
<input type="search" id="search" placeholder="e.g. Qwen or B1.hard"></label>
</div><p id="filter-note" class="muted" aria-live="polite"></p></div>
<p class="muted"><small>Percentages are rounded for display. Changes in percentage points
(pp) use the unrounded scores. Coverage counts class verdicts among scored items.
Abstentions count as errors. The main evaluation supplies the reference scheme.</small></p>
<p id="no-results" hidden>There are no results for these filters.</p>
{"".join(results)}
<footer id="unavailable"><details><summary>Unavailable or separately evaluated configurations
({len(unavailable)})</summary><ul>{excluded}</ul></details>
<p class="links"><a href="systems.csv">System inventory</a><a href="source_map.csv">Source map</a>
<a href="verdicts.csv">Item-level checks</a><a href="paired_comparisons.csv">Paired contrasts</a>
<a href="manifest.json">Manifest</a></p>
</footer></main><script>{SCRIPT}</script></body></html>
"""


def write_report(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["analysis"] != "offline reference sensitivity of fixed saved verdicts":
        raise ValueError("Expected the integrated-reference evaluation folder")
    rows = checked_csv(folder, manifest, "systems.csv")
    systems = {row["system_id"]: row for row in rows}
    if len(systems) != len(rows) or len(rows) != manifest["n_system_conditions"]:
        raise ValueError("System inventory does not match the manifest")
    scores = checked_csv(folder, manifest, "main_table.csv")
    unavailable = checked_csv(folder, manifest, "unavailable.csv")
    populations = manifest["populations"]
    if populations.keys() - POPULATIONS.keys():
        raise ValueError("Unknown population in the evaluation manifest")
    groups = grouped_scores(systems, scores, populations)
    if {family for _, family in groups} - FAMILIES.keys():
        raise ValueError("Unknown aggregator family in the inventory")
    models = yaml.safe_load((ROOT / "serving/models.yaml").read_text(encoding="utf-8"))["models"]
    model_names = {key: value["display_name"] for key, value in models.items()}
    output = folder / "summary.html"
    page = render(folder, groups, unavailable, model_names, populations, manifest)
    output.write_text(page, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(write_report(parser.parse_args().folder.resolve()))
