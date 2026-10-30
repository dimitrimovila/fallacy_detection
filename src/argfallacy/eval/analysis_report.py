"""Render the analysis contract using only its saved CSVs and manifest."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from html import escape
from pathlib import Path

from ..schemes import SchemeError

FORMAT = "thesis_analyses_v1"
PREVIEW_ROWS = 25
# Minimum public columns, including pending tables. The manifest records full schemas.
SCHEMAS = {
    "main_table": (
        "evaluation round system scheme_version subset_kind subset origin "
        "n_items accuracy macro_f1 balanced_accuracy excluded_classes status"
    ),
    "comparisons": (
        "evaluation system reference family family_m source_table kind read level n_items status"
    ),
    "per_class": (
        "evaluation system scheme_version subset origin class gold_support "
        "prediction_count precision recall f1"
    ),
    "rare_classes": (
        "evaluation system rare_definition rare_classes supported_rare_classes "
        "mean_recall undefined"
    ),
    "integrity": "evaluation scheme_version measure count denominator denominator_basis status",
    "stage1": "evaluation run system scheme_version condition n_items accuracy macro_f1 status",
    "routing_curve": (
        "evaluation system scheme_version subset tau requested_share attainable"
        " n_items explained_share status"
    ),
    "confidence": (
        "evaluation system scheme_version source_table item_id confidence "
        "confidence_source p_label p_label_lower_bound"
    ),
    "cq_level": (
        "evaluation system scheme_version population origin scheme cq_id "
        "n_items n_cq_pairs annotated_items brier_n agreement brier status"
    ),
    "cq_level_differences": (
        "evaluation system reference scheme_version population n_items "
        "n_cq_pairs agreement_difference brier_difference status"
    ),
    "human_agreement": (
        "evaluation system scheme_version population n_items units pairs alpha raw_agreement status"
    ),
    "human_path": (
        "evaluation system reference scheme_version population n_items "
        "annotated_items usable_items accuracy status"
    ),
    "abstention": "evaluation system origin n_items augrc aurc status",
    "abstention_curves": (
        "evaluation system origin n_items k coverage selective_risk generalized_risk"
    ),
    "abstention_thresholds": (
        "evaluation system origin n_items threshold kept errors coverage "
        "selective_risk generalized_risk"
    ),
    "abstention_differences": "evaluation system reference origin n_items augrc_difference status",
    "abstention_composition": (
        "evaluation system origin n_items requested_coverage actual_cut kind "
        "value count share_rejected rejection_rate"
    ),
    "difficulty": "evaluation system origin n_items requested_coverage actual_cut spearman status",
    "human_direct": (
        "evaluation system reference scheme_version origin comparison n_items accuracy status"
    ),
    "uncertainty_disagreement": (
        "evaluation system scheme_version origin selection observed_items auroc"
        " mean_confidence disagreement_prevalence status"
    ),
    "conformal_items": (
        "evaluation system scheme_version item_id fold terminal_ids size gold_included threshold"
    ),
    "conformal": (
        "evaluation system scheme_version kind subset n_items coverage mean_size singleton_share"
    ),
    "source_conformal": (
        "evaluation system scheme_version kind subset n_items coverage mean_size "
        "singleton_share source_table source_sha256 status"
    ),
}

SECTIONS = [
    (
        "main",
        "Both rounds: main and subgroup verdict tables",
        ["main_table"],
        "Gold scheme, fine labels, gold verdict reference. Populations are each system's declared "
        "eligible items; origins, schemes, corpus and without-NLAS views are "
        "recomputed from predictions. "
        "Class support exclusions and Good Argumentation zero denominators are explicit. "
        "New breakdowns are exploratory at 95%.",
    ),
    (
        "comparisons",
        "Original deciding families and exploratory comparisons",
        ["comparisons"],
        "Paired common eligible populations. Original families remain m=5 (round one, 99%) and "
        "m=4 (round two, 98.75%). Preserve source read gates and conditional outcomes; a gated "
        "comparison is not a claim. H_DQ is separately m=1 at 95%. Other views are exploratory "
        "95% intervals. No measurable difference is not equivalence or success.",
    ),
    (
        "classes",
        "Per-class scores and C1 rare-class diagnosis",
        ["rare_classes", "per_class"],
        "Rare classes have gold support <20 in the full frozen test before subsetting. "
        "Mean rare recall uses supported classes on the common population. This differs from "
        "the post-results frequent-class cutoff: gold support ≥10. Unsupported classes, "
        "prediction counts and undefined precision/recall remain visible.",
    ),
    (
        "cq",
        "CQ correctness and historical human references",
        ["cq_level", "cq_level_differences", "human_agreement", "human_path"],
        "CQ correctness, verdict quality and explanation consistency are separate measures. "
        "C1 binary supervision requires positive gold-scheme judgement; P7 uses the broader "
        "gold-sheet population. Hard human agreement is withheld below 20 annotated items. "
        "All CQs of an item move together. Version 1.1 labels do not supervise rewritten 2.0 CQs. "
        "Observed S-soft paths exclude unvisited nodes. Historical cross-sheet verdict agreement "
        "is separate from human traversal under the gold scheme. Second-rater "
        "CQs and blind repeats "
        "remain independently pending.",
    ),
    (
        "routing",
        "C2 routing by origin",
        ["routing_curve"],
        "Exact side records are required. Every origin retains the pooled threshold grid and "
        "pooled thresholds selected at requested shares; there is no origin retuning. Ties can "
        "make a requested share unattainable by any single threshold. Rounded saved routing "
        "summaries cannot recover exact decisions.",
    ),
    (
        "selective",
        "Ordered and attainable-threshold curves; AUGRC and AURC",
        ["abstention", "abstention_differences", "abstention_curves", "abstention_thresholds"],
        "Ordered P11 AUGRC and AURC are discrete means over k=1..N; item-id ordering breaks "
        "confidence ties. A cut through a tie is not attainable by one confidence threshold. "
        "Threshold curves keep ties together, require a valid verdict and finite confidence, "
        "and include the empty selection: generalized risk zero, selective risk undefined. "
        "Missing confidence is not zero. Maximum operational coverage and missing counts "
        "are recorded. Fixed cuts use half-up rounding and rerank within paired item draws. "
        "Systems without confidence are reference points. Exactly four exploratory AUGRC "
        "pairs are permitted: C2 versus zero-shot Qwen; C4 versus A2.pruned; C1 versus A2; "
        "A5 versus zero-shot Qwen. None decides success.",
    ),
    (
        "composition",
        "Abstention composition and difficulty",
        ["abstention_composition", "difficulty"],
        "Counts, shares among rejected items and within-group rejection rates use explicit "
        "kept/rejected/all denominators. Missing attributes are unobserved. Difficulty uses "
        "Enrico's frozen levels 1–5; Spearman correlates difficulty with 1-confidence on rated "
        "operational items only. Constant inputs and unusable replicates are undefined.",
    ),
    (
        "human",
        "Optional direct human verdicts and disagreement",
        ["human_direct", "uncertainty_disagreement"],
        "Only explicitly configured answers are used. Gold remains the primary reference; "
        "direct/path and historical-rater comparisons use observed common items. Human "
        "readiness, partial rounds and the planned minimum of 200 answers are recorded "
        "independently of D+Q availability. Global eligible ranking precedes intersection "
        "with observed humans. AUROC gives half credit for ties; single-class samples are "
        "undefined. These are associations with observed disagreement, not a "
        "human label distribution.",
    ),
    (
        "conformal",
        "Conformal sets and summaries",
        ["source_conformal", "conformal", "conformal_items"],
        "C5 uses genuine terminal distributions or verified companion exports, alpha=0.10, "
        "calibration on the other folds, with the original terminal and empty-mass rules. "
        "Learned out-of-fold scores give approximate coverage. Aggregate summaries cannot "
        "reconstruct sets. Missing exact inputs remain pending independently of lighter analyses.",
    ),
    (
        "integrity",
        "Integrity and confidence provenance",
        ["integrity", "confidence"],
        "Invalid, missing, cannot-be-determined, not-applicable and absent-verdict counts "
        "retain their sample/CQ/item units. Confidence records retain evaluation, version, "
        "source table/run, p_label and its lower-bound flag. Recovery requires exact parsed "
        "verdict agreement. Path, soft and routing confidence is never replaced with "
        "declared-verdict confidence.",
    ),
    (
        "stage1",
        "Stage-one scheme recognition",
        ["stage1"],
        "Frozen-test scheme recognition is not predicted-scheme verdict evaluation. "
        "The latter remains deferred. Observed/missing rows and supported/excluded scheme "
        "classes are recorded; these point scores do not acquire bootstrap intervals.",
    ),
]

DQ_NOTE = (
    "Saved spec-10 results, with original serialization and source evaluation/table/hash/row keys. "
    "H_DQ compares declared DQ 2.0 with matched-menu direct verdict 2.0: m=1, 95%. "
    "A positive primary interval requires a positive point difference of DQ 1.1 against "
    "the same-server zero-shot control for a robust improvement. A confidence interval "
    "containing zero is no measurable difference, never equivalence or success. "
    "Read the saved outcome, reading and secondary_difference. Declared verdict, hard path "
    "and soft traversal remain distinct. Stopped paths count against consistency. "
    "CQ correctness, verdict quality and explanation consistency are separate; consistency "
    "is necessary but insufficient for faithful explanations. The 2.0 questions were revised "
    "with earlier test errors in view. CQ/departure/change/replication diagnostics, imported "
    "abstention and exploratory C2 extensions retain that limitation. Imported summaries "
    "are separate from new analysis curves and cannot supply exact routing exports."
)

CSS = """
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#203033;background:#f4f5f2}
body{margin:0}main{max-width:1500px;margin:auto;padding:2rem}header{border-bottom:4px solid #246b68}
h1{font-size:2.4rem}h2{color:#185957}h3{font-size:1.1rem}
nav{display:flex;gap:.7rem;flex-wrap:wrap;margin:1.5rem 0}
a{color:#155e75}section{margin:2rem 0;padding:1.5rem;background:white;border-radius:10px}
p{line-height:1.6;max-width:105ch}.note,.status{background:#edf2ee;padding:.8rem;white-space:pre-wrap}
.scroll{overflow:auto;max-height:65vh}table{border-collapse:collapse;font-size:.8rem;min-width:100%}
td,th{text-align:left;padding:.5rem;border:1px solid #dde3df;max-width:34rem;min-width:6rem}
td{overflow-wrap:anywhere}
th{background:#e7eeea;position:sticky;top:0}tbody tr:nth-child(even){background:#f8faf8}
pre{white-space:pre-wrap;overflow-wrap:anywhere}button,input{padding:.5rem;margin:.4rem}summary{cursor:pointer}
.tools{display:flex;align-items:center;flex-wrap:wrap}.pending{color:#80551b}code{overflow-wrap:anywhere}
"""

JS = """
for(const block of document.querySelectorAll('[data-file]')){
 const payload=block.querySelector('script[type="application/json"]'); if(!payload)continue;
 const data=JSON.parse(payload.textContent),body=block.querySelector('tbody');let page=0,query='';
 const curveFiles=['abstention_curves.csv','abstention_thresholds.csv','routing_curve.csv',
  'dq_abstention_curves.csv','dq_routing_curve.csv'];
 if(curveFiles.includes(block.dataset.file)){
  const routing=data.columns.includes('explained_share');
  const xKey=routing?'explained_share':'coverage',yKey=routing?'macro_f1':'selective_risk';
  const xi=data.columns.indexOf(xKey),yi=data.columns.indexOf(yKey);
  const keys=['evaluation','system','origin','subset'].filter(k=>data.columns.includes(k));
  const groups=new Map();
  for(const row of data.rows){
   if(xi<0||yi<0||row[xi]===''||row[yi]==='')continue;
   const x=Number(row[xi]),y=Number(row[yi]);if(!Number.isFinite(x+y))continue;
   const key=keys.map(k=>row[data.columns.indexOf(k)]).join(' · ')||'saved curve';
   if(!groups.has(key))groups.set(key,[]);groups.get(key).push([x,y]);
  }
  if(groups.size){
   const panel=document.createElement('figure'),selector=document.createElement('select');
   selector.setAttribute('aria-label','Select saved curve');
   const caption=document.createElement('figcaption');
   caption.textContent=block.dataset.file.includes('threshold')||routing?
    'Saved threshold points only; intermediate coverage cuts may be unattainable.':
    'Saved ordered curve; a cut through a confidence tie may be unattainable by a threshold.';
   for(const name of groups.keys()){
    const option=document.createElement('option');option.textContent=name;selector.append(option);
   }
   const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
   svg.setAttribute('viewBox','0 0 720 240');svg.setAttribute('role','img');
   svg.style.maxWidth='800px';svg.style.width='100%';
   function draw(){svg.replaceChildren();const values=groups.get(selector.value);
    svg.setAttribute('aria-label',`${selector.value}: ${xKey} versus ${yKey}`);
    const points=values.slice().sort((a,b)=>a[0]-b[0]);
    function node(tag,attrs,text){const el=document.createElementNS(ns,tag);
     for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);
     if(text!==undefined)el.textContent=text;svg.append(el);return el;
    }
    node('path',{d:'M50 20V200H700',fill:'none',stroke:'#203033'});
    node('text',{x:50,y:220,'font-size':12},xKey.replaceAll('_',' ')+' (0–1)');
    node('text',{x:50,y:14,'font-size':12},yKey.replaceAll('_',' ')+' (0–1)');
    const coords=points.map(([x,y])=>[50+650*x,200-180*y]);
    if(block.dataset.file.includes('curves'))node('polyline',{
     points:coords.map(p=>p.join(',')).join(' '),fill:'none',stroke:'#246b68','stroke-width':2});
    else for(const [cx,cy] of coords)node('circle',{cx,cy,r:2,fill:'#246b68'});
   }
   selector.onchange=draw;panel.append(selector,svg,caption);
   block.querySelector('.tools').before(panel);draw();
  }
 }
 const input=block.querySelector('input'),count=block.querySelector('[data-count]');
 function show(){const rows=data.rows.filter(r=>!query||r.join(' ').toLowerCase().includes(query));
  const last=Math.max(0,Math.ceil(rows.length/25)-1);page=Math.min(last,Math.max(0,page));
  body.replaceChildren(...rows.slice(page*25,(page+1)*25).map(r=>{
   const tr=document.createElement('tr');
   for(const value of r){const td=document.createElement('td');
    td.textContent=value;tr.append(td)}return tr}));
  count.textContent=`${rows.length} matching rows · page ${page+1}/${last+1}`;
 }
 input.addEventListener('input',()=>{query=input.value.toLowerCase();page=0;show()});
 block.querySelector('[data-prev]').onclick=()=>{page--;show()};
 block.querySelector('[data-next]').onclick=()=>{page++;show()};show();
}
"""


def require(ok, message):
    if not ok:
        raise SchemeError(message)


def label(value):
    return value.replace("_", " ").replace("sha256", "SHA-256")


def _table(folder, name, manifest, fingerprints):
    file = folder / name
    part = manifest["parts"].get(name.removesuffix(".csv"), {})
    contract = manifest["tables"].get(name)
    require(isinstance(contract, dict), f"{name}: missing saved table contract")
    state = part.get("status")
    pending = state in ("pending", "deferred") and bool(part.get("reason"))
    if not file.is_file():
        require(pending, f"{name}: missing computed table; manifest must declare pending")
        return (
            f'<article id="table-{escape(name)}" data-file="{name}" data-rows="0">'
            f'<h3>{escape(label(name))}</h3><p class="pending">{escape(str(part))}</p></article>'
        )
    require(not file.is_symlink(), f"{name}: table symlinks are not supported")
    raw = file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8")), strict=True)
        columns = next(reader)
        rows = list(reader)
    except (UnicodeError, StopIteration, csv.Error) as error:
        raise SchemeError(f"{name}: malformed saved CSV: {error}") from None
    require(len(set(columns)) == len(columns), f"{name}: duplicate columns")
    required = SCHEMAS.get(name.removesuffix(".csv"), "").split()
    require(
        set(required) <= set(columns),
        f"{name}: missing required columns {set(required) - set(columns)}",
    )
    require(columns == contract.get("columns"), f"{name}: saved schema mismatch")
    require(all(len(r) == len(columns) for r in rows), f"{name}: malformed row width")
    require(len(rows) == contract.get("rows"), f"{name}: saved row count mismatch")
    require(bool(rows) or pending or state == "undefined", f"{name}: empty computed table")
    for col in contract.get("numeric_columns", []):
        require(col in columns, f"{name}: missing numeric column {col}")
        for row in rows:
            value = row[columns.index(col)]
            if not value:
                continue
            try:
                number = float(value)
                require(
                    math.isfinite(number) or col in ("tau", "threshold"), f"{name}: nonfinite {col}"
                )
            except ValueError:
                raise SchemeError(f"{name}: invalid numeric value in {col}: {value!r}") from None
    require(digest == contract.get("sha256"), f"{name}: saved SHA-256 mismatch")
    fingerprints[name] = digest
    heading = f"<h3>{escape(label(name))}</h3>"
    source = (
        '<p class="note">'
        + escape(
            f"Saved table: {name} · {len(rows)} rows · status: {state}. "
            f"SHA-256: {digest}. Source evaluations, versions, references and populations are "
            "identified in the row keys and provenance manifest. Blank estimates are unavailable "
            "or undefined, never placeholder zeros. Intervals use the row level and"
            " metric-specific "
            "replicate/omission counts; counts have no interval unless explicitly saved."
        )
        + f' <a href="{name}">CSV</a></p>'
    )
    status = (
        "<details><summary>Table status and missingness</summary><pre>"
        + escape(json.dumps(part, indent=2))
        + "</pre></details>"
    )
    if not rows:
        content = '<p class="pending">' + escape(part.get("reason", str(state))) + "</p>"
    else:
        total_rows = len(rows)
        rows = rows[:PREVIEW_ROWS]
        headers = "".join(f'<th scope="col">{escape(label(c))}</th>' for c in columns)
        first = "".join(
            "<tr>" + "".join(f"<td>{escape(v)}</td>" for v in r) + "</tr>" for r in rows[:25]
        )
        payload = (
            json.dumps(dict(columns=columns, rows=rows), ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        content = (
            f'<p class="note">Preview: {len(rows)} of {total_rows} rows. '
            'Search covers this preview only. Use the complete CSV for filtering '
            'and analysis; no saved rows were removed.</p>'
            '<div class="tools"><label>Search rows <input type="search" aria-label="Search '
            + name
            + '"></label>'
            "<button data-prev>Previous</button><button data-"
            "next>Next</button><span data-count></span></div>"
            f'<div class="scroll"><table><thead><tr>{headers}</tr></thead>'
            f"<tbody>{first}</tbody></table></div>"
            f'<script type="application/json">{payload}</script>'
        )
    return (
        f'<article id="table-{name}" data-file="{name}" data-rows="{contract["rows"]}">'
        f"{heading}{source}{status}{content}</article>"
    )


def write_analysis_summary(folder):
    folder = Path(folder).resolve()
    manifest_file = folder / "manifest.json"
    raw = manifest_file.read_bytes()
    try:
        manifest = json.loads(raw)
    except ValueError as error:
        raise SchemeError(f"manifest.json: malformed JSON: {error}") from None
    require(manifest.get("report_format") == FORMAT, "unsupported analysis report format")
    require(
        isinstance(manifest.get("tables"), dict) and isinstance(manifest.get("parts"), dict),
        "analysis manifest: missing table contracts or statuses",
    )
    for name in manifest["tables"]:
        require(
            Path(name).name == name and name.endswith(".csv"),
            "analysis manifest: tables must be local CSV basenames",
        )
    required = {n + ".csv" for n in SCHEMAS}
    require(required <= set(manifest["tables"]), "analysis manifest: incomplete required tables")
    # Protect every recorded source by path/inode metadata only; never read its contents.
    target = folder / "summary.html"
    require(not target.is_symlink(), "report output overlap through symlink")
    protected = [folder / n for n in manifest["tables"]] + [manifest_file]
    protected += [Path(r["path"]) for r in manifest.get("inputs", [])]
    for source in protected:
        require(target != source.resolve(), f"report output overlap: {source}")
        if target.exists() and source.is_file():
            require(
                not target.samefile(source), f"report output overlap through shared file: {source}"
            )
    fingerprints = {}
    rendered = {
        n.removesuffix(".csv"): _table(folder, n, manifest, fingerprints)
        for n in manifest["tables"]
    }
    sections, nav = [], []
    for key, title, names, note in SECTIONS:
        nav.append(f'<a href="#{key}">{escape(title)}</a>')
        sections.append(
            f'<section id="{key}"><h2>{escape(title)}</h2><p>{escape(note)}</p>'
            + "".join(rendered[n] for n in names)
            + "</section>"
        )
    extras = [n for n in rendered if n not in SCHEMAS]
    for key, title, names, note in [
        (
            "dq",
            "D+Q, matched controls and explanation consistency",
            [n for n in extras if n.startswith("dq_")],
            DQ_NOTE,
        ),
        (
            "post-results",
            "Selected post-results frequent-class summaries",
            [n for n in extras if n.startswith("post_results_")],
            "Exploratory additions after the results: frequent classes have full-"
            "test gold support ≥10. "
            "Only selected saved frequent-class tables are imported. No substitution simulation or "
            "separate 2.0 aggregator experiment is run. The original deciding "
            "families stay unchanged.",
        ),
    ]:
        nav.append(f'<a href="#{key}">{escape(title)}</a>')
        sections.append(
            f'<section id="{key}"><h2>{escape(title)}</h2><p>{escape(note)}</p>'
            + "".join(rendered[n] for n in names)
            + "</section>"
        )
    require(
        set(extras) <= {n for n in extras if n.startswith(("dq_", "post_results_"))},
        "unknown analysis table format",
    )
    provenance = json.dumps(manifest, indent=2, ensure_ascii=False)
    sections.append(
        '<section id="provenance"><h2>Provenance, readiness and resources</h2>'
        "<p>95% exploratory percentile intervals use shared item draws unless a saved "
        "deciding level is explicit. Without NLAS uses generator 2. Supported classes "
        "only enter macro scores; undefined flags preserve the canonical zero convention. "
        "Source commits absent from the source remain unavailable; hashes identify files. "
        "Synthetic analysis resources do not establish fitted replay feasibility. "
        "This report opens only this directory’s manifest and saved tables.</p>"
        '<p><a href="manifest.json">Manifest JSON</a></p><details><summary>Complete '
        "source/version, denominator, confidence, missingness and resource record</summary>"
        f"<pre>{escape(provenance)}</pre></details></section>"
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Thesis analyses</title>'
        f"<style>{CSS}</style><body><main><header><p>Saved-result analysis · fine label space</p>"
        "<h1>Thesis analyses of both rounds</h1><p>Verdict quality, CQ correctness and explanation "
        "consistency, with explicit populations and independent readiness states.</p></header>"
        "<nav>"
        + "".join(nav)
        + '<a href="#provenance">Provenance and readiness</a></nav>'
        + "".join(sections)
        + f"<script>{JS}</script></main></body></html>"
    )
    # Detect concurrent mutations before replacing the report; CSVs remain untouched.
    require(manifest_file.read_bytes() == raw, "manifest changed during report regeneration")
    for name, digest in fingerprints.items():
        require(
            hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest,
            f"{name}: changed during report regeneration",
        )
    target.write_text(html, encoding="utf-8")
    return target
