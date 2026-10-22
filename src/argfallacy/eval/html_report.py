"""A self-contained presentation of saved evaluation tables; no metrics are recomputed."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import pandas as pd

from ..client.run import load_models

CSS = """
:root{--ink:#172638;
--muted:#526477;
--line:#dbe3ec;
--blue:#176aa2;
--paper:#f4f7fb}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;
background:var(--paper);

color:var(--ink);
font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif}
main{max-width:1280px;
margin:auto;
padding:35px 26px 65px}h1{font-size:34px;
line-height:1.2;

margin:7px 0 12px;
letter-spacing:-1px}h2{font-size:24px;
margin:0 0 9px}h3{font-size:18px}
.eyebrow{text-transform:uppercase;
letter-spacing:2px;
font-size:11px;
font-weight:750;
color:#127a73}
.muted,.note{color:var(--muted)}.note{font-size:13px;
max-width:110ch}p{margin:8px 0 14px}
nav{position:sticky;
top:0;
z-index:2;
background:#f4f7fbf5;
border-bottom:1px solid var(--line);

display:flex;
gap:9px;
flex-wrap:wrap;
padding:12px 0;
margin-bottom:20px}
nav a,button{padding:7px 12px;
border:1px solid var(--line);
border-radius:7px;
background:white;

color:var(--ink);
text-decoration:none;
font:inherit;
font-size:13px;
cursor:pointer}
nav a:hover,button:hover{border-color:#176aa2}a{color:var(--blue)}
section{scroll-margin-top:88px;
background:white;
border:1px solid var(--line);
border-radius:12px;

padding:25px;
margin:20px 0;
box-shadow:0 3px 12px #122a4405}
.cards{display:grid;
grid-template-columns:repeat(3,1fr);
gap:14px;
margin:22px 0}
.card{background:white;
border:1px solid var(--line);
border-top:4px solid #24938b;
border-radius:9px;

padding:17px}.card strong{display:block;
font-size:37px;
line-height:1.3;
font-variant-numeric:tabular-nums}
.card span{font-size:13px;
color:var(--muted)}.verdict{background:#edf6fa;
border-left:4px solid #176aa2;

padding:16px 19px;
border-radius:5px}.scroll{overflow:auto;
max-height:650px}
table{border-collapse:separate;
border-spacing:3px;
width:100%;
font-size:13px;

font-variant-numeric:tabular-nums}th{text-align:left;
vertical-align:bottom;
color:#42576a}
th,td{padding:8px 10px;
border-radius:4px}thead th{background:#f0f4f8;
position:sticky;
top:0;
z-index:1}
tbody th{font-weight:500;
min-width:165px}td{text-align:right;
white-space:nowrap}
td.text{text-align:left;
white-space:normal;
min-width:180px}.ci{display:block;
font-size:10px;

color:inherit;
opacity:.85;
font-weight:400}.bar{height:6px;
background:#dce6ef;
border-radius:4px;
margin-top:5px}
.bar i{display:block;
background:#176aa2;
height:6px;
border-radius:4px}.empty{color:#8796a3}
.pill{display:inline-block;
border-radius:20px;
padding:2px 8px;
font-size:11px;

background:#e8f4ef;
color:#11624b}.pill.neutral{background:#edf0f4;
color:#516174}
.legend{font-size:12px;
color:var(--muted);
display:flex;
flex-wrap:wrap;
gap:12px;
margin:13px 0}
.swatch{display:inline-block;
width:12px;
height:12px;
margin-right:5px;
border-radius:3px}
details{border-top:1px solid var(--line);
margin-top:18px;
padding-top:13px}summary{cursor:pointer;

font-weight:650;
color:var(--blue)}.filters{display:flex;
flex-wrap:wrap;
gap:12px;
margin:14px 0}
input,select{padding:9px;
border:1px solid #bdcad6;
border-radius:6px;
font:inherit;
font-size:13px}
.stack{display:flex;
height:13px;
min-width:140px;
overflow:hidden;
border-radius:4px;

background:#edf0f4}.stack span{display:block;
height:100%}.dist{font-size:10px;
color:#526477;

white-space:normal;
max-width:210px;
margin-top:4px}.question{min-width:240px!important}
.figure svg{width:100%;
height:auto}code{font-size:12px;
overflow-wrap:anywhere}
footer{color:var(--muted);
font-size:12px}ul{padding-left:22px}li{margin:6px 0}
@media(max-width:760px){main{padding:20px 12px}.cards{grid-template-columns:1fr}
section{padding:16px}
h1{font-size:28px}nav{position:static}.scroll{max-height:550px}}
@media print{body{background:white;
font-size:11px}main{max-width:none;
padding:0}nav,button,.filters{
display:none}section{box-shadow:none;
break-inside:avoid;
padding:12px}.scroll{max-height:none;

overflow:visible}thead th{position:static}table{font-size:9px}td,th{padding:4px}.cards{gap:8px}}
"""

SCRIPT = """
document.querySelectorAll('.cq-filter').forEach(box=>{
 const select=box.querySelector('select'), input=box.querySelector('input');
 function filter(){let n=0;box.parentElement.querySelectorAll('tr[data-model]').forEach(row=>{
  const show=(!select.value||row.dataset.model===select.value)&&
   row.textContent.toLowerCase().includes(input.value.toLowerCase());
  row.hidden=!show;if(show)n++;
 });box.querySelector('output').textContent=n+' rows';}
 select.addEventListener('change',filter);input.addEventListener('input',filter);filter();
});
document.querySelector('#print').addEventListener('click',()=>window.print());
"""


def E(value):
    return html.escape(str(value), quote=True)


METRICS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "balanced_accuracy": "Balanced accuracy",
    "good_recall": "Recall of good arguments",
    "good_precision": "Precision of good arguments",
    "coverage": "Coverage",
}
COLORS = {
    "yes": "#167f83",
    "positive": "#167f83",
    "no": "#b15b76",
    "negative": "#b15b76",
    "cannot_be_determined": "#dcaa45",
    "na": "#8b99a5",
    "invalid": "#303d4a",
    "missing": "#303d4a",
}


def cell(value, low=None, high=None, difference=False, reverse=False):
    if value is None or pd.isna(value):
        return '<td class="empty">—</td>'
    value = float(value)
    strength = min(abs(value) * 4 if difference else max(1 - value if reverse else value, 0), 1)
    rgb = (35, 120, 160) if value >= 0 or not difference else (180, 100, 40)
    color = ",".join(str(round(255 + (c - 255) * strength)) for c in rgb)
    number = f"{value:+.3f}" if difference else f"{value:.3f}"
    interval = (
        f'<span class="ci">[{float(low):.3f}, {float(high):.3f}]</span>'
        if low is not None and high is not None and pd.notna(low) and pd.notna(high)
        else ""
    )
    ink = "color:white;--muted:white;" if strength > .72 else ""
    return f'<td style="background:rgb({color});{ink}">{number}{interval}</td>'


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f'<th scope="col">{E(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def section(identifier, title, note, content):
    return (
        f'<section id="{identifier}"><h2>{E(title)}</h2>'
        f'<p class="note">{E(note)}</p>{content}</section>'
    )


def distribution(raw):
    counts = json.loads(raw) if isinstance(raw, str) else {}
    total = sum(counts.values())
    bars = (
        "".join(
            f'<span style="width:{100 * n / total:.3f}%;background:{COLORS.get(a, "#64748b")}" '
            f'title="{E(a)}: {n}/{total}"></span>'
            for a, n in counts.items()
        )
        if total
        else ""
    )
    labels = ", ".join(
        f"{a.replace('cannot_be_determined', 'uncertain')}: {n}" for a, n in counts.items()
    )
    return (
        f'<td><div class="stack">{bars}</div>'
        f'<div class="dist">{E(labels or "Not available")}</div></td>'
    )


def metric_table(frame, label):
    rows = []
    for r in frame.to_dict("records"):
        name = label(r["system"])
        f1 = r["macro_f1"]
        row = (
            f'<tr><th scope="row">{E(name)}<div class="bar">'
            f'<i style="width:{100 * f1:.2f}%"></i></div></th>'
        )
        for metric in METRICS:
            row += cell(r[metric], r.get(f"{metric}_low"), r.get(f"{metric}_high"))
        rows.append(row + "</tr>")
    return table(["System", *METRICS.values()], rows)


def cq_section(folder, suffix, version, label, read):
    metrics = read(folder / "cq_metrics.csv")
    agree = read(folder / "cq_agreement.csv")
    rows = []
    for r in agree.itertuples():
        rows.append(
            f"<tr><th>{E(r.scheme.replace('_', ' '))} · {E(r.cq_id)}</th><td>{r.items}</td>"
            + cell(r.alpha)
            + cell(r.unanimous_share)
            + f"<td>{r.complete_items}</td></tr>"
        )
    content = table(
        ["Scheme / CQ", "Texts", "Model alpha", "All models agree", "Complete texts"], rows
    )
    models = metrics.model.unique().tolist()
    options = '<option value="">All models</option>' + "".join(
        f'<option value="{E(m)}">{E(label(m))}</option>' for m in models
    )
    filters = (
        f'<div class="filters cq-filter"><label>Model <select>{options}</select></label>'
        '<label>Find a question <input type="search" placeholder="Scheme, CQ or words"></label>'
        '<output class="note"></output></div>'
    )
    rows = []
    for r in metrics.itertuples():
        rows.append(
            f'<tr data-model="{E(r.model)}"><th>{E(r.scheme.replace("_", " "))} · '
            f"{E(r.cq_id)}<br><small>{E(label(r.model))}</small></th>"
            f'<td class="text question">{E(r.text)}</td>'
            + distribution(r.distribution)
            + distribution(r.human_distribution)
            + f"<td>{r.annotated}</td>"
            + cell(r.human_agreement)
            + cell(r.reached)
            + cell(r.weight_on_verdict)
            + "</tr>"
        )
    legend = (
        '<div class="legend">'
        + "".join(
            f'<span><i class="swatch" style="background:{COLORS[k]}"></i>{name}</span>'
            for k, name in [
                ("yes", "Yes / positive"),
                ("no", "No / negative"),
                ("cannot_be_determined", "Cannot determine"),
                ("na", "Not applicable"),
            ]
        )
        + "</div>"
    )
    detail = table(
        [
            "Scheme / CQ / model",
            "Question",
            "Model answers",
            "Human answers",
            "Annotated texts",
            "Human agreement",
            "Reached by diagram",
            "Verdict changes",
        ],
        rows,
    )
    content += (
        "<details><summary>Inspect individual questions and answer distributions</summary>"
        + legend
        + filters
        + detail
        + "</details>"
    )
    if (folder / "cq_calibration.svg").is_file():
        svg = (folder / "cq_calibration.svg").read_text(encoding="utf-8")
        svg = svg[svg.index("<svg") :]
        content += (
            "<details><summary>Probability calibration and Brier scores</summary>"
            '<p class="note">The dashed line is perfect calibration. Points below it mean '
            "the probability of yes/positive is too high for the observed human answers. "
            "Ten equal-count bins pool the questions; ties are split deterministically. "
            "Brier is a squared probability error: smaller is better. Each score shows "
            "its number of binary human targets; NA and uncertain targets are excluded.</p>"
            f'<div class="figure">{svg}</div>'
        )
        brier = []
        for r in metrics.to_dict("records"):
            row = f"<tr><th>{E(r['scheme'])} · {E(r['cq_id'])} · {E(label(r['model']))}</th>"
            for source in ("p_logprob", "p_verbal", "p_sample"):
                row += cell(r[f"{source}_brier"], reverse=True) + f"<td>{r[f'{source}_n']}</td>"
            brier.append(row + "</tr>")
        content += (
            table(
                [
                    "Question / model",
                    "Logprob Brier ↓",
                    "n",
                    "Verbal Brier ↓",
                    "n",
                    "Sample Brier ↓",
                    "n",
                ],
                brier,
            )
            + "</details>"
        )
    note = (
        "Alpha measures agreement beyond chance, accounting for the answer distribution. "
        "1 means perfect agreement; 0 means no agreement beyond chance; negative means less. "
        "A dash means alpha is undefined. Unanimity uses only complete model ratings. "
        "Reached is the share of texts visiting the node; verdict changes is the share whose "
        "result changes when that answer alone is flipped. "
        + (
            "Human comparisons use original-scheme annotations and need at least 20 texts. "
            "NA is a distinct category: different applicability judgements can lower agreement."
            if suffix == "v2"
            else "No human CQ comparison: the annotations answer the original "
            "questions. CQ IDs alone do not establish equivalence."
        )
    )
    return section(f"cqs-{suffix}", f"Critical questions · {version}", note, content)


def write_summary(folder, judge_audit=None, revised_cqs=None):
    """Render existing CSV values verbatim in summary.html; fingerprint all table inputs."""
    folder = Path(folder)
    inputs = {}

    def read(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    def metadata(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8"))

    manifest = metadata(folder / "manifest.json")
    main = read(folder / "main_table.csv")
    decide = read(folder / "deciding.csv")
    deciding_level = f"{100 * (1 - .05 / manifest.get('m', 5)):g}%"
    names = {k: v.get("display_name", k) for k, v in load_models().items()}
    methods = {
        "A1": "A1 · Reordered diagram",
        "A2": "A2 · Probabilistic diagram",
        "A3": "A3 · Learned classifier",
        "A4": "A4 · Model ensemble",
        "A5": "A5 · LLM judge with explanations",
        "majority": "Majority per scheme (oracle)",
    }

    def label(name):
        prefix, sep, model = name.partition(":")
        if sep:
            return (
                {"A0": "Diagram", "zeroshot": "Direct verdict", "ceiling": "Oracle ceiling"}.get(
                    prefix, prefix
                )
                + " · "
                + names.get(model, model)
            )
        return methods.get(name, names.get(name, name))

    reference = manifest["reference_model"]
    by_system = main.set_index("system")
    primary = main[main.system.isin(["A1", "A2", "A3", "A4", "A5"])]
    best = primary.sort_values("macro_f1", ascending=False).iloc[0] if len(primary) else None
    cards = [("Reference diagram", by_system.loc[f"A0:{reference}"], label(f"A0:{reference}"))]
    if best is not None:
        cards.append(("Highest observed aggregator score", best, label(best.system)))
    if f"zeroshot:{reference}" in by_system.index:
        cards.append(
            (
                "Direct verdict · same model",
                by_system.loc[f"zeroshot:{reference}"],
                "Text + gold scheme → verdict; no CQ answers",
            )
        )
    intro = (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><span>{E(title)}</span><strong>{r.macro_f1:.3f}</strong>'
            f"<span>Macro F1 · {E(description)}</span><br>"
            f"<span>Accuracy {100 * r.accuracy:.1f}% · "
            f"{int(r.n_items)} texts</span></div>"
            for title, r, description in cards
        )
        + "</div>"
    )
    passed = decide[decide.system.isin(["A2", "A3", "A4"]) & decide.improvement.eq(True)]
    conclusion = (
        "At least one soft-answer or ensemble aggregator passes the prespecified "
        "comparison against the reference diagram."
        if len(passed)
        else "The soft-answer and ensemble aggregators do not show a measurable macro F1 "
        "improvement over the reference diagram under the prespecified test."
    )
    intro += (
        f'<div class="verdict"><strong>What the main experiment says</strong><p>{E(conclusion)}</p>'
    )
    if "A5" in set(decide.system):
        judge = decide.set_index("system").loc["A5"]
        significant = bool(judge.improvement)
        intro += (
            "<p>The LLM judge "
            + ("improves on" if significant else "does not measurably improve on")
            + " the diagram. Its comparison with direct classification is a separate question: "
            + f"macro F1 difference {judge.zeroshot_macro_f1_difference:+.3f}, "
            f"95% interval [{judge.zeroshot_macro_f1_low:.3f}, "
            f"{judge.zeroshot_macro_f1_high:.3f}]. "
            + (
                "This interval includes zero; it does not establish a difference."
                if judge.zeroshot_macro_f1_low <= 0 <= judge.zeroshot_macro_f1_high
                else "This comparison is exploratory."
            )
            + "</p>"
        )
    intro += '</div><p class="note">Card ranking is descriptive. Macro F1 averages class F1 scores '
    intro += (
        "over gold-supported classes; it prevents frequent classes from dominating as they "
        "can in accuracy. All results here use the correct (gold) scheme, not the full "
        "pipeline with predicted schemes. Higher scores are better except Brier.</p>"
    )
    sections = [
        section(
            "overview",
            "The result at a glance",
            f"Main evaluation: {manifest['name']} · "
            f"schemes {manifest.get('schemes', {}).get('tag', 'not recorded')} · "
            f"{manifest['items']} frozen test texts · "
            f"fine label space · {manifest['resamples']:,} paired bootstrap resamples",
            intro,
        )
    ]
    body = metric_table(main[~main.system.str.startswith("ceiling:")], label)
    body += (
        '<p class="note">Cell colour and bars encode the value; numbers remain authoritative. '
        "Intervals below scores are 95%. Accuracy is the fraction of correct verdicts; "
        "balanced accuracy gives each supported class equal weight. Coverage is the fraction "
        "with a verdict; missing verdicts count as errors. Majority is an in-sample oracle "
        "using gold labels, not a deployable model.</p>"
    )
    body += (
        '<details><summary>What do A0–A5 mean?</summary><ul>'
        '<li><strong>A0 — fixed diagram:</strong> follow the model’s yes/no answers through '
        'the original decision graph.</li>'
        '<li><strong>A1 — reordered diagram:</strong> learn the order of question blocks '
        'using training folds, then evaluate on held-out texts.</li>'
        '<li><strong>A2 — probabilistic diagram:</strong> carry answer probabilities through '
        'the graph instead of selecting only yes or no.</li>'
        '<li><strong>A3 — learned classifier:</strong> learn the relation between CQ probabilities '
        'and verdicts, using held-out evaluation folds.</li>'
        '<li><strong>A4 — ensemble:</strong> average the models’ answer probabilities, '
        'then traverse the probabilistic diagram.</li>'
        '<li><strong>A5 — LLM judge:</strong> let a language model read the CQ answers '
        'and their explanations and choose the verdict.</li></ul></details>'
    )
    body += (
        "<details><summary>Oracle ceilings · diagnostic bounds</summary>"
        + metric_table(main[main.system.str.startswith("ceiling:")], label)
        + "</details>"
    )
    sections.append(
        section(
            "systems",
            "Which methods work?",
            "All models and primary aggregators, "
            "scored on the same texts. Bars show macro F1 on a fixed 0–1 scale.",
            body,
        )
    )
    rows = []
    for r in decide.itertuples():
        status = (
            '<span class="pill">Interval excludes zero</span>'
            if r.improvement
            else ('<span class="pill neutral">No measurable improvement</span>')
        )
        rows.append(
            f"<tr><th>{E(label(r.system))}</th>"
            + cell(r.macro_f1_difference, r.macro_f1_low, r.macro_f1_high, True)
            + f"<td>{status}</td>"
            + cell(
                r.zeroshot_macro_f1_difference,
                r.zeroshot_macro_f1_low,
                r.zeroshot_macro_f1_high,
                True,
            )
            + "</tr>"
        )
    comparisons = table(
        [
            "Aggregator",
            f"Δ macro F1 vs diagram · {deciding_level} CI",
            "Decision vs diagram",
            "Δ macro F1 vs direct verdict · 95% CI",
        ],
        rows,
    )
    without = read(folder / "main_table_without_nlas.csv")
    comparisons += (
        "<details><summary>Remove the synthetic corpus (NLAS)</summary>"
        + metric_table(without[~without.system.str.startswith("ceiling:")], label)
        + "</details>"
    )
    sections.append(
        section(
            "comparisons",
            "Is the difference convincing?",
            "Positive differences "
            "favour the aggregator (blue); negative differences favour the comparator "
            f"(orange). The diagram comparison uses a {deciding_level} interval with Bonferroni "
            f"correction across {manifest.get('m', 5)} primary configurations. "
            "Only A2–A4 answer the "
            "soft-answer/ensemble criterion; A1 and A5 answer different questions. "
            "The direct-verdict comparison is exploratory.",
            comparisons,
        )
    )
    if judge_audit:
        audit = Path(judge_audit)
        metrics = read(audit / "metrics.csv")
        metrics = metrics[metrics.subset == "all"]
        judge_main = metrics.pivot(index="system", columns="metric", values="value").reset_index()
        for bound in ("low", "high"):
            for metric in METRICS:
                values = metrics[metrics.metric == metric].set_index("system")[bound]
                judge_main[f"{metric}_{bound}"] = judge_main.system.map(values)
        labels = {
            "diagram_full_v2": "Original diagram · v2",
            "diagram_full_v3": "Revised diagram · v3",
            "direct": "Direct verdict · original label menu",
            "judge_full_v2": "Judge v2 · explanations",
            "judge_full_v2_answers_only": "Judge v2 · answers only",
            "judge_full_v3": "Judge v3 · explanations",
            "judge_full_v3_answers_only": "Judge v3 · answers only",
        }
        content = metric_table(judge_main, lambda x: labels.get(x, x))
        costs = read(audit / "cost.csv")
        content += (
            "<details><summary>Judge inference cost</summary>"
            + table(
                ["Judge", "Calls", "Client minutes", "Input tokens", "Output tokens"],
                [
                    f"<tr><th>{E(labels.get(r.system, r.system))}</th><td>{r.calls}</td>"
                    f"<td>{r.client_wall_minutes:.1f}</td><td>{r.prompt_tokens:,}</td>"
                    f"<td>{r.completion_tokens:,}</td></tr>"
                    for r in costs.itertuples()
                ],
            )
            + "</details>"
        )
        sections.append(
            section(
                "judges",
                "The four LLM judges",
                "All judges use Qwen and 601 texts. "
                "v2/v3 comparisons combine changes in CQ evidence and in some label "
                "menus: they do not isolate wording alone. This audit uses its own "
                "recorded resamples; small interval differences from the main table "
                "are expected. Cost excludes CQ generation and server startup.",
                content,
            )
        )
    if (folder / "cq_metrics.csv").is_file():
        sections.append(cq_section(folder, "v2", "original schemes (full v2)", label, read))
    if revised_cqs:
        sections.append(
            cq_section(Path(revised_cqs), "v3", "revised schemes (full v3)", label, read)
        )
    if (folder / "human_agreement.csv").is_file():
        humans = read(folder / "human_agreement.csv")
        cq_manifest = metadata(folder / "cq_manifest.json")
        content = table(
            ["Annotators", "Texts with ≥2 ratings", "Alpha", "Raw agreement"],
            [
                f"<tr><th>{E(r.annotators)}</th><td>{r.units}</td>"
                + cell(r.alpha)
                + cell(r.raw_agreement)
                + "</tr>"
                for r in humans.itertuples()
            ],
        )
        content += (
            '<p class="note">Excluded item–annotator ratings: '
            f"{cq_manifest['conflicting_verdict_ratings_excluded']} conflicting across sheets; "
            f"{cq_manifest['nonterminal_verdict_ratings_excluded']} without a fine terminal label. "
            "Missing ratings are allowed; identical repeated labels count once. Populations "
            "differ by annotator pair, so rows are not a ranking of annotators.</p>"
        )
        sections.append(
            section(
                "humans",
                "How much do the humans agree?",
                "Raw agreement counts "
                "matching observed rater pairs. Alpha adjusts for the distribution "
                "of labels. These annotations are involved in the benchmark's "
                "construction; they are not independent validation. CQ agreement "
                "with a second annotator and blind self-reannotation remain unavailable.",
                content,
            )
        )
    inputs_json = E(json.dumps(inputs, indent=2))
    provenance = (
        f"<p>Evaluation: <code>{E(manifest['name'])}</code> · source run "
        f"<code>{E(manifest['runs']['answers'])}</code> · seed {manifest['seed']}.</p>"
        "<p>Every displayed metric and interval is read from a saved CSV. This page "
        "does not fit models, resample data or rerun inference. Additional analyses "
        "retain their own manifests and limitations.</p><details><summary>Input fingerprints "
        "(SHA-256)</summary><pre>" + inputs_json + "</pre></details>"
    )
    sections.append(
        section(
            "sources",
            "Scope and reproducibility",
            "Predicted-scheme evaluation, "
            "abstention curves and the complete set of scheme/corpus breakdowns "
            "are not included in this page yet.",
            provenance,
        )
    )
    nav = [("overview", "Overview"), ("systems", "Methods"), ("comparisons", "Comparisons")]
    if judge_audit:
        nav.append(("judges", "Judges v2 / v3"))
    if (folder / "cq_metrics.csv").is_file():
        nav.append(("cqs-v2", "Questions v2"))
    if revised_cqs:
        nav.append(("cqs-v3", "Questions v3"))
    if (folder / "human_agreement.csv").is_file():
        nav.append(("humans", "Human agreement"))
    nav.append(("sources", "Sources"))
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Thesis results · {E(manifest['name'])}</title><style>{CSS}</style></head>"
        '<body><main><div class="eyebrow">Walton schemes · experiment results</div>'
        "<h1>From critical questions to fallacy verdicts</h1>"
        '<p class="muted">A visual reading of the experiments: what improved, '
        "what remains uncertain, and where the questions disagree.</p><nav>"
        + "".join(f'<a href="#{key}">{E(text)}</a>' for key, text in nav)
        + '<button id="print" type="button">Print / Save PDF</button></nav>'
        + "".join(sections)
        + "<footer>Self-contained report · English labels · all scores shown with their scope. "
        "Source metric tables and manifests remain the reference.</footer></main>"
        f"<script>{SCRIPT}</script></body></html>"
    )
    path = folder / "summary.html"
    path.write_text(document, encoding="utf-8")
    return path
