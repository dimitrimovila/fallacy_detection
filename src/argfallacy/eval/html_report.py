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
    if manifest.get("round") == 2:
        return write_round2_summary(folder)
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
                {"A0": "Diagram", "zeroshot": "Zero-shot", "ceiling": "Oracle ceiling"}.get(
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
                "Zero-shot · same model",
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
            + " the diagram. Its comparison with zero-shot classification is a separate question: "
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
            "Δ macro F1 vs zero-shot · 95% CI",
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
            "The zero-shot comparison is exploratory.",
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
            "direct": "Zero-shot · original label menu",
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


# ------------------------------------------------------------------ round 2

ROUND2_METHODS = {
    "C1": "C1 · combined CQ answers, soft diagram",
    "C1.hard": "C1.hard · combined answers, hard diagram",
    "C2": "C2 · routing between diagram and zero-shot",
    "C3": "C3 · scheme trees on the combined answers",
    "C4": "C4 · S-soft",
    "C4.path": "C4.path · single S-soft path",
    "C4.nohistory": "C4.nohistory · S-soft without history",
    "A2.pruned": "A2.pruned · independent answers, same pruning",
    "A2": "A2 · probabilistic diagram (round 1)",
    "A4": "A4 · equal-weight ensemble (round 1)",
    "A5": "A5 · LLM judge with explanations",
    "A5.answers_only": "A5 · LLM judge, answers only",
    "C2.A0": "C2.A0 · routing with the reference diagram",
    "C3.pooled": "C3.pooled · recalibrated terminal masses",
    "A3.tree": "A3.tree · trees on the reference answers",
    "majority": "Majority per scheme (oracle)",
    "A3": "A3 · logistic regression per scheme (round 1)",
    "A3.combined": "A3.combined · A3 on the combined answers of the majority",
    "A2.prior": "A2.prior · A2, mass divided by the class frequency",
    "C1.prior": "C1.prior · C1, mass divided by the class frequency",
    "C3.pooled.balanced": "C3.pooled.balanced · recalibration with balanced classes",
    "C1.stacking.gold_balanced.soft": "(c) with balanced classes · soft diagram",
    "C1.stacking.gold_balanced.hard": "(c) with balanced classes · hard diagram",
}
OUTCOMES = {"improvement": "pill", "no measurable difference": "pill neutral"}
PALETTE = ("#176aa2", "#c4622d", "#24938b", "#8a4fb5", "#b8912f", "#b15b76", "#4d6b85",
           "#5d9c3f")


def _read_optional(path, read):
    """A saved table, or an empty one when the evaluation wrote no columns."""
    try:
        return read(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _figure(series, x_label, y_label, points=()):
    """Lines on the unit square: ``series`` is (name, colour, [(x, y)]), ``points`` is
    (colour, x, y, low, high) drawn with a vertical interval."""
    width, height, left, right, top, bottom = 640, 330, 52, 14, 12, 42

    def x(v):
        return left + v * (width - left - right)

    def y(v):
        return top + (1 - v) * (height - top - bottom)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{E(y_label)}">']
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{x(0)}" x2="{x(1)}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" '
                     'stroke="#dbe3ec"/>'
                     f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="11" '
                     f'text-anchor="end" fill="#526477">{tick:g}</text>'
                     f'<text x="{x(tick):.1f}" y="{height - bottom + 16}" font-size="11" '
                     f'text-anchor="middle" fill="#526477">{tick:g}</text>')
    for name, colour, values in series:
        line = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in values)
        parts.append(f'<polyline points="{line}" fill="none" stroke="{colour}" '
                     f'stroke-width="2"><title>{E(name)}</title></polyline>')
    for colour, a, b, low, high in points:
        if pd.notna(low) and pd.notna(high):
            parts.append(f'<line x1="{x(a):.1f}" x2="{x(a):.1f}" y1="{y(low):.1f}" '
                         f'y2="{y(high):.1f}" stroke="{colour}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="3.5" fill="{colour}"/>')
    parts.append(f'<text x="{(x(0) + x(1)) / 2:.1f}" y="{height - 6}" font-size="12" '
                 f'text-anchor="middle" fill="#172638">{E(x_label)}</text>'
                 f'<text transform="translate(13 {(y(0) + y(1)) / 2:.1f}) rotate(-90)" '
                 f'font-size="12" text-anchor="middle" fill="#172638">{E(y_label)}</text></svg>')
    legend = "".join(f'<span><i class="swatch" style="background:{c}"></i>{E(n)}</span>'
                     for n, c, _ in series)
    return f'<div class="figure">{"".join(parts)}</div><div class="legend">{legend}</div>'


def write_round2_summary(folder):
    """The page of a round 2 evaluation, from its saved tables only."""
    folder = Path(folder)
    inputs = {}

    def read(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    manifest_path = folder / "manifest.json"
    inputs[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = {name: _read_optional(folder / f"{name}.csv", read) for name in (
        "main_table", "deciding", "deciding_breakdowns", "exploratory", "cq_level",
        "cq_level_differences", "human_path", "routing_curve", "abstention",
        "abstention_curves", "abstention_composition", "conformal", "ssoft_changes")}
    # written only by evaluations made after the additions to round 2
    tables.update({name: _read_optional(folder / f"{name}.csv", read)
                   if (folder / f"{name}.csv").is_file() else pd.DataFrame()
                   for name in ("frequent_classes", "frequent_classes_differences",
                                "substitution", "substitution_curve")})
    names = {k: v.get("display_name", k) for k, v in load_models().items()}
    level = f"{100 * manifest['deciding_level']:g}%"

    def label(name):
        prefix, sep, rest = name.partition(":")
        if sep and prefix in ("A0", "zeroshot", "entry"):
            model, at, version = rest.partition("@")
            kind = {"A0": "Diagram", "zeroshot": "Zero-shot", "entry": "Entry"}[prefix]
            return f"{kind}{' ' + version if at else ''} · {names.get(model, model)}"
        return ROUND2_METHODS.get(name, name)

    def pill(outcome):
        kind = OUTCOMES.get(outcome, "pill neutral")
        return f'<td><span class="{kind}">{E(outcome)}</span></td>'

    def triple(row, metric, difference=True, reverse=False):
        return cell(row.get(metric if not difference else f"{metric}_difference"),
                    row.get(f"{metric}_low"), row.get(f"{metric}_high"), difference, reverse)

    main = tables["main_table"]
    by_system = main.set_index("system") if len(main) else pd.DataFrame()
    decide = tables["deciding"]
    sections = []

    # overview
    card = [("Evaluation", manifest["name"]), ("Date", manifest["date"]),
            ("Commit", manifest.get("commit") or "unknown"),
            ("Full run v2", manifest["runs"]["answers"]),
            ("Full run v3", manifest["runs"].get("answers_schemes_2") or "none"),
            ("S-soft runs", ", ".join(manifest["runs"].get("ssoft") or []) or "none"),
            ("Control without history",
             ", ".join(manifest["runs"].get("ssoft_nohistory") or []) or "none"),
            ("Reference model", label(f"A0:{manifest['reference_model']}").split(" · ")[-1]),
            ("Items", f"{manifest['items']} ({manifest['items_without_nlas']} without NLAS)"),
            ("Seed, resamples, folds", f"{manifest['seed']}, {manifest['resamples']:,}, "
                                       f"{manifest['folds']['k']}"),
            ("Deciding level", f"m = {manifest['m']}: {level} intervals")]
    cards = "".join(
        f'<div class="card"><span>{E(r.hypothesis)} · {E(label(r.system))}</span>'
        f"<strong>{'—' if pd.isna(r.macro_f1_difference) else f'{r.macro_f1_difference:+.3f}'}"
        f"</strong><span>Δ macro F1 against {E(label(r.reference))}</span><br>"
        f'<span class="{OUTCOMES.get(r.outcome, "pill neutral")}">{E(r.outcome)}</span></div>'
        for r in decide.itertuples() if not r.read_after or r.read)
    overview = (f'<div class="cards">{cards}</div>'
                + table(["", ""], [f"<tr><th>{E(k)}</th><td class=\"text\">{E(v)}</td></tr>"
                                   for k, v in card]))
    sections.append(section(
        "overview", "Round 2 at a glance",
        "Second family of aggregators, fixed before its results. Fine label space, gold "
        "scheme condition. Cards show the first comparison of each component and every "
        "second comparison that was read.", overview))

    # deciding comparisons
    rows = []
    for r in decide.to_dict("records"):
        rows.append(f"<tr><th>{E(r['hypothesis'])}<br><small>{E(label(r['system']))} vs "
                    f"{E(label(r['reference']))}</small></th>"
                    + triple(r, "macro_f1") + pill(r["outcome"])
                    + triple(r, "accuracy") + triple(r, "balanced_accuracy") + "</tr>")
    content = table(["Comparison", f"Δ macro F1 · {level} CI", "Outcome", "Δ accuracy · 95%",
                     "Δ balanced accuracy · 95%"], rows)
    breakdown = tables["deciding_breakdowns"]
    if len(breakdown):
        rows = [f"<tr><th>{E(r['hypothesis'])} · {E(r['kind'])}: {E(r['subset'])}"
                + (" ⚑" if r["flagged"] else "") + f"</th><td>{r['n_items']}</td>"
                f"<td>{r['n_classes']}</td>" + triple(r, "macro_f1") + triple(r, "accuracy")
                + triple(r, "balanced_accuracy") + "</tr>"
                for r in breakdown.to_dict("records")]
        content += ("<details><summary>By gold origin, by scheme and without NLAS "
                    "(95%, decide nothing)</summary>"
                    + table(["Comparison · subset", "Items", "Classes", "Δ macro F1",
                             "Δ accuracy", "Δ balanced accuracy"], rows)
                    + '<p class="note">⚑ marks a scheme with fewer than 30 items. Breakdowns '
                    "reuse the resamples of the whole test set; without NLAS has its own.</p>"
                    "</details>")
    sections.append(section(
        "deciding", "Deciding comparisons",
        f"Macro F1, paired bootstrap, Bonferroni across the {manifest['m']} components: each "
        f"interval at {level}. An improvement only if the interval excludes zero. Within a "
        "component the second comparison is read only after the first passes; a component "
        "not computed keeps its place in m. Against the zero-shot an interval below zero is "
        "the cost of the diagram, and one containing zero is no measurable difference, not "
        "equivalence. Accuracy and balanced accuracy are reported beside, at 95%.", content))

    # systems
    rows = []
    for r in main.to_dict("records"):
        rows.append(f"<tr><th>{E(label(r['system']))}<br><small><code>{E(r['system'])}</code> · "
                    f"{E(r['role'])} · schemes {E(r['schemes'])}</small></th>"
                    f"<td>{r['n_items']}</td>"
                    + "".join(cell(r[m], r[f"{m}_low"], r[f"{m}_high"]) for m in METRICS)
                    + "</tr>")
    sections.append(section(
        "systems", "Every system",
        "Each system on its own items, with 95% intervals. Exploratory systems decide "
        "nothing. The majority per scheme is an in-sample oracle.",
        table(["System", "Items", *METRICS.values()], rows)))

    # CQ level and supervision
    level_table = tables["cq_level"]
    content = ""
    if len(level_table):
        whole = level_table[level_table.scheme == "all"]
        rows = [f"<tr><th>{E(label(r['system']))}</th><td>{r['n']}</td>"
                + triple(r, "agreement", False) + triple(r, "brier", False, True) + "</tr>"
                for r in whole.to_dict("records")]
        content += table(["Combiner or entry", "Labelled pairs", "Agreement", "Brier ↓"], rows)
        rows = [f"<tr><th>{E(r['scheme'])} · {E(label(r['system']))}</th><td>{r['n']}</td>"
                + triple(r, "agreement", False) + triple(r, "brier", False, True) + "</tr>"
                for r in level_table[level_table.scheme != "all"].to_dict("records")]
        content += ("<details><summary>By scheme</summary>"
                    + table(["Scheme · system", "Pairs", "Agreement", "Brier ↓"], rows)
                    + "</details>")
        differences = tables["cq_level_differences"]
        rows = [f"<tr><th>{E(label(r['system']))} vs {E(label(r['reference']))}</th>"
                f"<td>{r['n']}</td>" + triple(r, "agreement") + triple(r, "brier")
                + "</tr>" for r in differences[differences.scheme == "all"].to_dict("records")]
        content += ("<details><summary>Paired differences (95%)</summary>"
                    + table(["Comparison", "Pairs", "Δ agreement", "Δ Brier"], rows)
                    + "</details>")
    human = tables["human_path"]
    if len(human):
        h = human.iloc[0]
        content += (f'<p class="note">Reference line, declared as such: the path of the human '
                    f"answers through the diagram 1.1 reaches a verdict on {h.usable_items} of "
                    f"{h.test_items} items; on them accuracy {h.accuracy:.3f}, macro F1 "
                    f"{h.macro_f1:.3f}.</p>")
    sections.append(section(
        "cq-level", "The level of each CQ",
        "Agreement of the hard combined answer with the binary human CQ labels (an answer "
        "outside the arcs disagrees) and Brier score of p_first; every pair of an item moves "
        "together in the resamples. Exploratory.", content))

    rows = []
    for combiner, supervision, mark in (("majority", "none", "(a)"),
                                        ("dawid_skene", "none", "(a)"),
                                        ("weighted", "cq", "(b)"), ("stacking", "cq", "(b)"),
                                        ("stacking", "gold", "(c)")):
        soft = "C1" if (combiner, supervision) == ("stacking", "cq") else \
            f"C1.{combiner}.{supervision}.soft"
        hard = "C1.hard" if soft == "C1" else soft.removesuffix(".soft") + ".hard"
        cq_row = (level_table[(level_table.system == f"C1.{combiner}.{supervision}")
                              & (level_table.scheme == "all")] if len(level_table) else [])
        got = cq_row.iloc[0] if len(cq_row) else {}
        rows.append(
            f"<tr><th>{mark} {E(combiner.replace('_', ' '))}, supervision "
            f"{E(supervision)}</th>"
            + "".join(cell(by_system.loc[n, "macro_f1"], by_system.loc[n, "macro_f1_low"],
                           by_system.loc[n, "macro_f1_high"]) if n in by_system.index
                      else '<td class="empty">—</td>' for n in (soft, hard))
            + cell(got.get("agreement"), got.get("agreement_low"), got.get("agreement_high"))
            + "</tr>")
    content = table(["Combination", "Macro F1 · soft", "Macro F1 · hard", "CQ agreement"], rows)
    explored = tables["exploratory"]
    supervision_rows = explored[explored.question.str.startswith("supervision")] \
        if len(explored) else explored
    content += table(["Comparison", "Δ macro F1 · 95%"], [
        f"<tr><th>{E(r['system'])} vs {E(r['reference'])}<br><small>{E(r['question'])}"
        f"</small></th>" + triple(r, "macro_f1") + "</tr>"
        for r in supervision_rows.to_dict("records")])
    sections.append(section(
        "supervision", "The supervision factor",
        "(a) no labels, (b) human CQ labels, (c) gold verdicts through the diagram; (b) and "
        "(c) share the stacking family and its centre. Similar verdicts: the CQ labels and the "
        "gold ask the same of the answers. (c) above (b) with lower CQ agreement: the gain "
        "bends the answers away from what their questions ask.", content))

    # routing curve
    curve = tables["routing_curve"]
    if len(curve):
        line = curve[curve.requested_share.isna()].sort_values(["explained_share", "tau"])
        asked = curve[curve.requested_share.notna()]
        figure = _figure(
            [("Macro F1", PALETTE[0], list(zip(line.explained_share, line.macro_f1,
                                               strict=True))),
             ("Accuracy", PALETTE[1], list(zip(line.explained_share, line.accuracy,
                                               strict=True)))],
            "share of verdicts explained by the diagram", "score",
            [(PALETTE[0], r.explained_share, r.macro_f1, r.macro_f1_low, r.macro_f1_high)
             for r in asked.itertuples() if r.attainable]
            + [(PALETTE[1], r.explained_share, r.accuracy, r.accuracy_low, r.accuracy_high)
               for r in asked.itertuples() if r.attainable])
        rows = [f"<tr><th>{r.requested_share:.0%}</th>"
                + (f"<td>{r.tau:+.4f}</td><td>{r.explained_share:.3f}</td>"
                   + cell(r.macro_f1, r.macro_f1_low, r.macro_f1_high)
                   + cell(r.accuracy, r.accuracy_low, r.accuracy_high)
                   if r.attainable else '<td colspan="4" class="empty">not attainable</td>')
                + "</tr>" for r in asked.itertuples()]
        content = figure + table(["Requested share", "τ", "Share reached", "Macro F1 · 95%",
                                  "Accuracy · 95%"], rows)
    else:
        content = '<p class="empty">C2 was not computed.</p>'
    sections.append(section(
        "routing", "The cost of staying explainable",
        "C2 with every threshold τ from −1 to 1: the diagram where P_d ≥ P_z − τ. Agreeing "
        "verdicts count as explained by the diagram. Thresholds are read on the pooled "
        "out-of-fold estimates and fixed across resamples. Descriptive.", content))

    # S-soft
    shown = [n for n in ("C4", "C4.path", "C4.nohistory", "A2.pruned",
                         f"A0:{manifest['reference_model']}@2.0") if n in by_system.index]
    content = metric_table(main[main.system.isin(shown)], label) if shown else ""
    pending_history = manifest.get("ssoft_pending_items", {}).get("count", 0)
    pending_control = manifest.get("ssoft_nohistory_pending_items", {}).get("count", 0)
    for run, count in (("history", pending_history), ("control without history", pending_control)):
        if count:
            item = "item" if count == 1 else "items"
            content += (f'<p class="note">Incomplete S-soft {run} run: {count} {item} with '
                        "pending mass; its systems and comparisons are not computed.</p>")
    if "C4" not in by_system.index and not (pending_history or pending_control):
        content += ('<p class="note">No S-soft run yet: C4, C4.path, C4.nohistory and their '
                    "comparisons are listed as missing; H4 is not computed and m stays "
                    f"{manifest['m']}.</p>")
    elif "C4" in by_system.index:
        content += (f'<p class="note">Items with pending mass (calls missing from the run): '
                    f"{pending_history}.</p>")
        ssoft_rows = explored[explored.system.str.startswith("C4")] if len(explored) else []
        if len(ssoft_rows):
            content += table(["Comparison", "Δ macro F1 · 95%"], [
                f"<tr><th>{E(label(r['system']))} vs {E(label(r['reference']))}<br><small>"
                f"{E(r['question'])}</small></th>" + triple(r, "macro_f1") + "</tr>"
                for r in ssoft_rows.to_dict("records")])
        changes = tables["ssoft_changes"]
        if len(changes):
            content += ("<details><summary>Answers that change on the most probable path"
                        "</summary>" + table(["Scheme · CQ", "Nodes", "Changed", "Share"], [
                            f"<tr><th>{E(r.scheme)} · {E(r.cq_id)}</th><td>{r.nodes}</td>"
                            f"<td>{r.changed}</td>" + cell(r.share) + "</tr>"
                            for r in changes.itertuples()]) + "</details>")
    sections.append(section(
        "ssoft", "S-soft",
        "The CQs asked along the diagram 2.0 with the earlier answers of the path in view, "
        "branches kept above mass 0.02, against A2.pruned on the answers of full run v3 with "
        "the same burden table and pruning.", content))

    # uncertainty
    abstained = tables["abstention"]
    content = ""
    if len(abstained):
        levels = [c.removeprefix("selective_accuracy_") for c in abstained.columns
                  if c.startswith("selective_accuracy_") and c[-1].isdigit()]
        rows = [f"<tr><th>{E(label(r['system']))}</th>"
                + triple(r, "augrc", False, True) + triple(r, "aurc", False, True)
                + "".join(triple(r, f"selective_accuracy_{v}", False) for v in levels)
                + cell(r.get("augrc_margin"), reverse=True) + "</tr>"
                for r in abstained.to_dict("records")]
        content += table(["System", "AUGRC ↓", "AURC ↓",
                          *[f"Accuracy at {v}%" for v in levels], "AUGRC by margin ↓"], rows)
        curves = tables["abstention_curves"]
        drawn = [n for n in ("C1", "C2", "C3", "C4", "A2", "A4",
                             f"zeroshot:{manifest['reference_model']}", "A5")
                 if len(curves) and n in set(curves.system)]
        if drawn:
            content += _figure(
                [(label(n), PALETTE[k % len(PALETTE)],
                  list(zip(curves[curves.system == n].coverage,
                           curves[curves.system == n].selective_risk, strict=True)))
                 for k, n in enumerate(drawn)], "coverage", "selective risk")
        kinds = tables["abstention_composition"]
        if len(kinds):
            own = kinds[kinds.kind == "good argumentation"]
            good = (own.assign(share=own.share.where(own.value == "yes", 0.0))
                    .pivot_table(index="system", columns="coverage", values="share",
                                 aggfunc="sum", sort=False))
            good = good.reindex(columns=sorted(good.columns, reverse=True))
            rows = [f"<tr><th>{E(label(system))}</th>"
                    + "".join(cell(good.loc[system].get(c)) for c in good.columns) + "</tr>"
                    for system in good.index]
            content += ("<details><summary>Composition of the abstained items</summary>"
                        '<p class="note">Share of Good Argumentation (gold) among the items '
                        "each system abstains on; abstention_composition.csv also has scheme, "
                        "corpus and gold origin.</p>"
                        + table(["System", *[f"at {c:.0%}" for c in good.columns]], rows)
                        + "</details>")
    conformal = tables["conformal"]
    if len(conformal):
        whole = conformal[conformal.kind == "all"]
        content += "<h3>Conformal sets</h3>" + table(
            ["System", "Coverage", "Mean size", "One verdict"],
            [f"<tr><th>{E(label(r.system))}</th>" + cell(r.coverage)
             + f"<td>{r.mean_size:.2f}</td>" + cell(r.singleton_share) + "</tr>"
             for r in whole.itertuples()])
        rows = [f"<tr><th>{E(label(r.system))} · {E(r.kind)}: {E(r.subset)}</th>"
                f"<td>{r.n_items}</td>" + cell(r.coverage) + f"<td>{r.mean_size:.2f}</td>"
                + cell(r.singleton_share) + "</tr>"
                for r in conformal[conformal.kind != "all"].itertuples()]
        content += ("<details><summary>By gold origin and by scheme</summary>"
                    + table(["System · subset", "Items", "Coverage", "Mean size",
                             "One verdict"], rows) + "</details>")
    sections.append(section(
        "uncertainty", "Uncertainty",
        "Abstention curves: items by decreasing confidence; AUGRC is the mean share of items "
        "given a wrong verdict instead of an abstention. Items without a verdict come last "
        "and count as errors. Conformal sets at α = "
        f"{manifest['conformal_alpha']:g}: thresholds from the gold of the other four folds. "
        "All exploratory.", content))

    # exploratory and missing
    rows = [f"<tr><th>{E(label(r['system']))} vs {E(label(r['reference']))}<br><small>"
            f"{E(r['question'])}</small></th><td>{r['n_items']}</td>"
            + triple(r, "macro_f1") + "</tr>" for r in explored.to_dict("records")]
    sections.append(section(
        "exploratory", "Exploratory comparisons",
        "Named in spec 09 or each exploratory system against the diagram on its own run. "
        "95% intervals; they support no claim.",
        table(["Comparison", "Items", "Δ macro F1 · 95%"], rows)))
    # additions after the results
    content = ""
    frequent = tables["frequent_classes"]
    if len(frequent):
        classes = manifest.get("frequent_classes", {})
        content += (f'<h3>Macro F1 over the frequent classes</h3><p class="note">The '
                    f"{len(classes.get('classes', []))} classes with at least "
                    f"{classes.get('minimum_gold_items', '')} gold verdicts in the test set; "
                    "the macro F1 over all classes moves with a single item of a class with "
                    "one or two.</p>")
        content += table(["System", "Items", "Classes", "Macro F1, frequent classes · 95%"], [
            f"<tr><th>{E(label(r['system']))}<br><small><code>{E(r['system'])}</code> · "
            f"{E(r['role'])}</small></th><td>{r['n_items']}</td><td>{r['n_classes']}</td>"
            + triple(r, "macro_f1_frequent", False) + "</tr>"
            for r in frequent.to_dict("records")])
        differences = tables["frequent_classes_differences"]
        content += ("<details><summary>Every comparison on the frequent classes (95%)"
                    "</summary>" + table(["Comparison", "Items", "Δ macro F1, frequent"], [
                        f"<tr><th>{E(label(r['system']))} vs {E(label(r['reference']))}"
                        f"<br><small>{E(r['kind'])}: {E(r['question'])}</small></th>"
                        f"<td>{r['n_items']}</td>" + triple(r, "macro_f1_frequent") + "</tr>"
                        for r in differences.to_dict("records")]) + "</details>")
    substituted = tables["substitution"]
    if len(substituted):
        curve = tables["substitution_curve"]
        content += ("<h3>Human CQ answers in place of the reference model's</h3>"
                    '<p class="note">The diagram 1.1 traversed with the hard answers of the '
                    "reference model, where the human binary answer replaces the model's: "
                    "on every labelled pair of one CQ, on all of them, and, in the figure, "
                    "on random shares of the disagreeing pairs (mean of "
                    f"{manifest.get('substitution', {}).get('draws', '')} draws).</p>")
        if len(curve):
            content += _figure(
                [(text, PALETTE[k], list(zip(curve.agreement, curve[metric], strict=True)))
                 for k, (metric, text) in enumerate((("macro_f1", "Macro F1"),
                                                     ("macro_f1_frequent",
                                                      "Macro F1, frequent classes"),
                                                     ("accuracy", "Accuracy")))],
                "agreement of the answers with the human CQ labels", "score")
        content += table(["Scheme · CQ", "Labelled", "Disagreeing", "Verdicts changed",
                          "Wrong → right", "Right → wrong", "Δ macro F1 · 95%",
                          "Δ macro F1, frequent · 95%", "Δ accuracy · 95%"], [
            f"<tr><th>{E(r['scheme'])} · {E(r['cq_id'])}</th><td>{r['labelled']}</td>"
            f"<td>{r['disagreeing']}</td><td>{r['verdicts_changed']}</td>"
            f"<td>{r['wrong_to_right']}</td><td>{r['right_to_wrong']}</td>"
            + triple(r, "macro_f1") + triple(r, "macro_f1_frequent") + triple(r, "accuracy")
            + "</tr>" for r in substituted.to_dict("records")])
    if content:
        sections.append(section(
            "additions", "Added after the results",
            "Analyses and systems named after the results of round 2, all exploratory: the "
            "macro F1 over the frequent classes, the human CQ answers in place of the "
            "model's, and the systems marked as added after the results in their "
            "description.", content))
    missing = "".join(f"<li><strong>{E(k)}</strong>: {E(v)}</li>"
                      for k, v in manifest.get("missing", {}).items())
    sections.append(section(
        "missing", "Not computed", "Named systems and comparisons without a result, and why.",
        f"<ul>{missing or '<li>None.</li>'}</ul>"))
    dq_content = dq_section(folder, manifest, read) if "dq" in manifest else None
    sections.append(section(
        "sources", "Scope and reproducibility",
        "Every value is read from a saved CSV of this evaluation; the page fits nothing.",
        "<details><summary>Input fingerprints (SHA-256)</summary><pre>"
        + E(json.dumps(inputs, indent=2)) + "</pre></details>"))

    if dq_content is not None:
        sections.append(dq_content)

    nav = [("overview", "Overview"), ("deciding", "Deciding"), ("systems", "Systems"),
           ("cq-level", "CQ level"), ("supervision", "Supervision"), ("routing", "Routing"),
           ("ssoft", "S-soft"), ("uncertainty", "Uncertainty"),
           ("exploratory", "Exploratory"), ("missing", "Missing"), ("sources", "Sources")]
    if len(tables["frequent_classes"]) or len(tables["substitution"]):
        nav.insert(nav.index(("missing", "Missing")), ("additions", "Additions"))
    if "dq" in manifest:
        nav.append(("dq", "D+Q"))
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Round 2 · {E(manifest['name'])}</title><style>{CSS}</style></head>"
        '<body><main><div class="eyebrow">Walton schemes · round 2</div>'
        "<h1>Combining, routing and learning from critical questions</h1><nav>"
        + "".join(f'<a href="#{key}">{E(text)}</a>' for key, text in nav)
        + '<button id="print" type="button">Print / Save PDF</button></nav>'
        + "".join(sections)
        + "<footer>Self-contained report · the CSV tables and the manifest remain the "
        "reference.</footer></main>"
        f"<script>{SCRIPT}</script></body></html>"
    )
    path = folder / "summary.html"
    path.write_text(document, encoding="utf-8")
    return path


def dq_section(folder, manifest, read):
    """The separate D+Q section, using only saved tables and manifest metadata."""
    tables = {name: _read_optional(folder / f"dq_{name}.csv", read) for name in (
        "deciding", "deciding_breakdowns", "main_table", "consistency", "departures",
        "cq_level", "cq_changes", "replication", "exploratory", "abstention",
        "abstention_curves", "routing_curve", "choices")}

    def values(frame, keys, metrics):
        if frame.empty:
            return '<p class="note">Not computed; see the missing inputs below.</p>'

        def formatted(row, metric):
            stem = ("explained" if metric == "explained_share"
                    else metric.removesuffix("_difference"))
            return cell(row.get(metric), row.get(f"{stem}_low"), row.get(f"{stem}_high"),
                        difference=metric.endswith("_difference"))

        rows = []
        for r in frame.to_dict("records"):
            rows.append("<tr>" + "".join(
                f'<td class="text">{E(r[k]) if k in r and pd.notna(r[k]) else ""}</td>'
                for k in keys)
                        + "".join(formatted(r, m) for m in metrics)
                        + "</tr>")
        return table([*keys, *metrics], rows)

    def part(identifier, title, content):
        return f'<h3 id="dq-{identifier}">{E(title)}</h3>' + content

    decide = tables["deciding"]
    primary = values(decide, ["hypothesis", "system", "reference", "outcome", "reading", "reason"],
                     ["macro_f1_difference"])
    primary += ('<p class="note">H_DQ is a family of one comparison, with a 95% paired '
                "interval. An improvement is robust only when the point estimate of D+Q 1.1 "
                "against the repeated zero-shot is positive. Zero inside the interval means "
                "no measurable difference, not equivalence.</p>")
    content = part("primary", "The primary comparison", primary)
    beside = values(decide, ["hypothesis"], ["accuracy_difference", "balanced_accuracy_difference"])
    beside += values(tables["deciding_breakdowns"], ["kind", "subset", "n_items", "flagged"],
                     ["macro_f1_difference", "accuracy_difference", "balanced_accuracy_difference"])
    beside += "<details><summary>Every D+Q and matched-control system</summary>" + values(
        tables["main_table"], ["system", "schemes", "n_items"], list(METRICS)) + "</details>"
    content += part("breakdowns", "Reported beside the primary comparison", beside)
    content += part("consistency", "Consistency and departures", values(
        tables["consistency"], ["system", "kind", "subset", "n_items", "n_paths", "n_consistent"],
        ["consistent", "stopped", "agreement_on_paths", "declared_mass", "declared_is_max",
         "declared_accuracy_consistent", "path_accuracy_consistent",
         "declared_accuracy_other", "path_accuracy_other"])
        + values(tables["departures"], ["system", "scheme", "cq_id", "answer", "reason",
                                        "gold_origin", "count", "n_items"], ["share"]))
    content += part("cq-level", "The level of each CQ", values(
        tables["cq_level"], ["kind", "system", "reference", "scheme", "n", "brier_n"],
        ["agreement", "brier", "agreement_difference", "brier_difference"])
        + '<p class="note">Brier uses the parsed p_logprob on the same item/CQ pairs in '
        "both sources, including mass outside the arcs. Missing probabilities are not "
        "imputed; brier_n gives their support. All pairs of an item move together.</p>"
        + values(tables["cq_changes"], ["system", "reference", "scheme", "cq_id", "n_items"],
                 ["changed", "first", "first_reference"]))
    content += part("exploratory", "Secondary and exploratory comparisons", values(
        tables["exploratory"], ["comparison", "system", "reference", "subset", "n_items"],
        ["macro_f1_difference"])
        + values(tables["replication"], ["system", "reference", "subset", "n_items"], ["agreement"])
        + '<p class="note">The 2.0 questions were written with full-run-v2 test errors in '
        "view. Comparisons using those questions can favour D+Q or S-soft. Secondary and "
        "exploratory intervals decide nothing.</p>")
    uncertainty = values(tables["abstention"], ["kind", "system", "reference", "n_items"],
                         ["augrc", "aurc", *[f"selective_accuracy_{round(100 * v)}"
                                             for v in (1.0, 0.95, 0.9, 0.8, 0.7)],
                          "augrc_difference"])
    curves = tables["abstention_curves"]
    if len(curves):
        reference = manifest["reference_model"]
        names = [n for n in curves.system.unique()
                 if n.split(":", 1)[-1].removesuffix("@2.0") == reference]
        uncertainty += _figure([
            (name, PALETTE[k % len(PALETTE)], list(zip(
                curves.loc[curves.system == name, "coverage"],
                curves.loc[curves.system == name, "selective_risk"], strict=True)))
            for k, name in enumerate(names)], "coverage", "selective risk")
    content += part("abstention", "Abstention", uncertainty)
    routing = tables["exploratory"]
    if "comparison" in routing:
        routing = routing[routing.comparison == "C2"]
    routing_content = values(routing, ["system", "reference", "n_items"],
                             ["macro_f1_difference", "explained_share"])
    curve = tables["routing_curve"]
    if len(curve):
        line = curve[curve.requested_share.isna()].sort_values(["explained_share", "tau"])
        routing_content += _figure([
            ("Macro F1", PALETTE[0], list(zip(line.explained_share, line.macro_f1, strict=True))),
            ("Accuracy", PALETTE[1], list(zip(line.explained_share, line.accuracy, strict=True)))
        ], "share of verdicts explained by C1@2.0", "score")
        routing_content += values(curve[curve.requested_share.notna()],
                                  ["requested_share", "tau", "attainable"],
                                  ["explained_share", "macro_f1", "accuracy"])
    routing_content += ('<p class="note">All routing comparisons here are exploratory. '
                        "C2@2.0 uses two sides and threshold zero; its curve varies the threshold. "
                        "C2.dq adds D+Q on schemes 1.1. C2.dq@2.0 adds D+Q and S-soft on "
                        "schemes 2.0. A verdict is explained when it equals C1, C4, or a "
                        "consistent D+Q verdict. Added confidences use training-item mean "
                        "imputation and a missing indicator. C1 is cross-fitted within every "
                        "training partition, fitting the original CQ labels for transfer "
                        "to 2.0.</p>")
    routing_content += "<details><summary>Routing and upstream hyperparameters</summary>" + values(
        tables["choices"], ["component", "fold", "parameter", "value", "training_items"], []
    ) + "</details>"
    content += part("routing", "Exploratory routing with D+Q and S-soft", routing_content)
    missing = manifest["dq"].get("missing", {})
    content += part("missing", "What was not computed", "<ul>" + "".join(
        f"<li><strong>{E(k)}</strong>: {E(v)}</li>" for k, v in missing.items()) + "</ul>")
    return section("dq", "D+Q and the direct verdict with the 2.0 menus",
                   "Gold scheme condition, fine label space, shared round-two folds and item "
                   "resamples. A missing or invalid verdict is wrong.",
                   content)

