"""CQ diagnostics and nominal agreement on the frozen test set."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..annotations import ANNOTATIONS_FILE, ANNOTATOR, ITEMS_FILE, load_annotations
from ..client.run import schemes_dir_of, schemes_hash
from ..parse import INVALID
from ..paths import SCHEMES_DIR
from ..schemes import SchemeError, load_all, traverse
from .prior import load_test_set

CBD = "cannot_be_determined"


def agreement(units) -> dict:
    """Nominal alpha with missing ratings; raw agreement over observed pairs.

    Singleton units contribute neither coincidences nor expected disagreement.
    Undefined alpha (no pairs or no variation) remains missing, including unanimity.
    """
    units = [[x for x in unit if x is not None and not pd.isna(x)] for unit in units]
    units = [u for u in units if len(u) >= 2]
    counts, disagreement, pairs, equal = {}, 0.0, 0, 0
    for unit in units:
        for value in unit:
            counts[value] = counts.get(value, 0) + 1
        different = sum(a != b for a, b in combinations(unit, 2))
        n_pairs = len(unit) * (len(unit) - 1) // 2
        disagreement += 2 * different / (len(unit) - 1)
        pairs += n_pairs
        equal += n_pairs - different
    n = sum(counts.values())
    expected = (n * n - sum(c * c for c in counts.values())) / (n - 1) if n > 1 else 0
    return {"units": len(units), "pairs": pairs,
            "alpha": 1 - disagreement / expected if expected else np.nan,
            "raw_agreement": equal / pairs if pairs else np.nan,
            "distribution": json.dumps(counts, sort_keys=True)}


def human_answers(annotations, items, schemes):
    """Identify a sheet by its complete CQ set, independently of its judged scheme."""
    notes = annotations[annotations.annotator == ANNOTATOR]
    output = []
    gold = items.set_index("item_id").gold_scheme
    for sheet, group in notes.groupby("sheet"):
        cqs = set(group.loc[group.field.str.startswith("CQ"), "field"])
        if not cqs:
            continue
        matches = [s for s, scheme in schemes.items() if set(scheme.cqs) == cqs]
        if len(matches) != 1:
            raise SchemeError(f"{sheet}: ambiguous CQ annotation scheme")
        sid = matches[0]
        rows = group[group.field.isin(cqs)].copy()
        rows = rows[rows.item_id.map(gold) == sid]
        rows["scheme"] = sid
        output.append(rows)
    if not output:
        return pd.DataFrame(columns=["item_id", "scheme", "cq_id", "human"])
    result = pd.concat(output).rename(columns={"field": "cq_id", "value": "human"})
    keys = ["item_id", "scheme", "cq_id"]
    if result.groupby(keys).human.nunique().gt(1).any():
        raise SchemeError("Conflicting human CQ ratings; repeats need explicit provenance")
    result = result.drop_duplicates(keys)
    result["human"] = result.human.replace({"idk": CBD})
    return result[[*keys, "human"]]


def tables(summary, items, annotations, schemes, entries, human_compatible=True):
    """Build P7/P8 tables without inference, raw responses or burden tables."""
    gold = items.set_index("item_id").gold_scheme
    rows = summary[(summary.stage == "stage2") & (summary.scheme_condition == "gold")].copy()
    rows = rows[rows.item_id.isin(gold.index) & rows.model.isin(entries)]
    if (rows.scheme != rows.item_id.map(gold)).any():
        raise SchemeError("CQ diagnostics: row scheme differs from the item's gold scheme")
    keys = ["item_id", "scheme", "cq_id", "model"]
    if rows.duplicated(keys).any():
        raise SchemeError("Duplicate model CQ answers")
    human = human_answers(annotations, items, load_all()) if human_compatible else pd.DataFrame(
        columns=["item_id", "scheme", "cq_id", "human"])
    rows = rows.merge(human, on=["item_id", "scheme", "cq_id"], how="left")
    stats, model_agreement, calibration = [], [], []
    grids = {(model, ident): g.set_index("cq_id").answer.to_dict()
             for (model, ident), g in rows.groupby(["model", "item_id"])}
    for sid, scheme in schemes.items():
        ids = items.loc[items.gold_scheme == sid, "item_id"].tolist()
        if not ids:
            continue
        for cq, question in scheme.cqs.items():
            subset = rows[(rows.scheme == sid) & (rows.cq_id == cq)]
            pivot = subset.pivot(index="item_id", columns="model", values="answer").reindex(
                index=ids, columns=entries)
            ratings = pivot.replace(INVALID, np.nan)
            complete = ratings.notna().all(axis=1)
            agree = agreement(ratings.to_numpy().tolist())
            model_agreement.append({"scheme": sid, "cq_id": cq, "items": len(ids),
                                    **agree, "complete_items": int(complete.sum()),
                                    "unanimous_share": (ratings[complete].nunique(axis=1)
                                                        == 1).mean()})
            for model in entries:
                g = subset[subset.model == model].set_index("item_id").reindex(ids)
                h = human[(human.scheme == sid) & (human.cq_id == cq)].set_index("item_id")
                g["human"] = h.human.reindex(ids)
                answer = g.answer.fillna("missing")
                annotated = g.human.notna()
                observed = g.answer.notna() & (g.answer != INVALID)
                reached, changed = [], []
                for ident in ids:
                    grid = grids.get((model, ident), {})
                    base = traverse(scheme, grid)
                    reached.append(cq in base.visited_cqs)
                    flipped = dict(grid)
                    if grid.get(cq) in question.answer_space:
                        flipped[cq] = next(a for a in question.answer_space if a != grid[cq])
                    changed.append(traverse(scheme, flipped).verdict != base.verdict)
                record = {"scheme": sid, "cq_id": cq, "text": question.text, "model": model,
                          "items": len(ids), "missing": int((answer == "missing").sum()),
                          "invalid": int((answer == INVALID).sum()),
                          "distribution": json.dumps(answer.value_counts().to_dict(),
                                                     sort_keys=True),
                          "human_distribution": json.dumps(
                              g.human.dropna().value_counts().to_dict(), sort_keys=True),
                          "annotated": int(annotated.sum()),
                          "human_agreement": ((g.answer == g.human)[annotated].mean()
                                              if annotated.sum() >= 20 else np.nan),
                          "reached": np.mean(reached), "weight_on_verdict": np.mean(changed)}
                positive, negative = question.answer_space
                binary = g.human.isin([positive, negative])
                for source in ("p_logprob", "p_verbal", "p_sample"):
                    # Every probability in the parsed summary is already oriented
                    # toward the first admitted answer, including verbal confidence.
                    column = f"{source}_{positive}"
                    p = pd.to_numeric(g[column], errors="coerce") if column in g else pd.Series(
                        np.nan, index=g.index)
                    if source == "p_verbal":
                        p = pd.to_numeric(g[source], errors="coerce")
                    if ((p.dropna() < 0) | (p.dropna() > 1)).any():
                        raise SchemeError("CQ probability outside [0, 1]")
                    valid = binary & observed & p.notna()
                    target = (g.human == positive).astype(float)
                    record[f"{source}_n"] = int(valid.sum())
                    record[f"{source}_brier"] = ((p - target)[valid] ** 2).mean()
                    calibration.extend({"model": model, "probability": source,
                                        "scheme": sid, "cq_id": cq, "item_id": i,
                                        "p": p[i], "target": target[i]} for i in g.index[valid])
                stats.append(record)
    bins = []
    if calibration:
        pool = pd.DataFrame(calibration).sort_values(["model", "probability", "p", "scheme",
                                                     "cq_id", "item_id"])
        for (model, source), group in pool.groupby(["model", "probability"]):
            for number, indices in enumerate(np.array_split(np.arange(len(group)), 10), 1):
                if len(indices):
                    g = group.iloc[indices]
                    bins.append({"model": model, "probability": source, "bin": number,
                                 "n": len(g), "mean_probability": g.p.mean(),
                                 "observed_positive": g.target.mean(), "min": g.p.min(),
                                 "max": g.p.max()})
    verdicts = annotations[(annotations.field == "verdict") &
                          annotations.annotator.isin(["enrico", "eleni", ANNOTATOR]) &
                          annotations.item_id.isin(gold.index)]
    human_rows, conflicts = {}, 0
    # Multiple sheets are not independent raters. Keep only unambiguous item/annotator
    # labels; report conflicts rather than selecting the label closest to the gold.
    terminals = set().union(*(s.terminals for s in load_all().values()))
    coarse = 0
    for (ident, annotator), g in verdicts.groupby(["item_id", "annotator"]):
        values = set(g.value)
        if len(values) != 1:
            conflicts += 1
        elif next(iter(values)) not in terminals:
            coarse += 1
        else:
            human_rows.setdefault(ident, {})[annotator] = next(iter(values))
    people = ["enrico", "eleni", ANNOTATOR]
    human_stats = []
    for group in [people, *combinations(people, 2)]:
        human_stats.append({"annotators": "+".join(group), **agreement(
            [[r.get(a) for a in group] for r in human_rows.values()])})
    metrics = pd.DataFrame(stats)
    if len(metrics):
        metrics["mean_weight_on_verdict"] = metrics.groupby(["scheme", "cq_id"])[
            "weight_on_verdict"].transform("mean")
    return {"cq_metrics": metrics, "cq_agreement": pd.DataFrame(model_agreement),
            "cq_calibration": pd.DataFrame(bins, columns=["model", "probability", "bin", "n",
                "mean_probability", "observed_positive", "min", "max"]),
            "human_agreement": pd.DataFrame(human_stats)}, {
                "conflicting_verdict_ratings_excluded": conflicts,
                "nonterminal_verdict_ratings_excluded": coarse,
                "human_cq_compatible": human_compatible,
                "missing": ["CQ ratings by the second human annotator",
                            "Identified blind repeat annotations"]}


def write_cq_report(run_dir, out, items_path=ITEMS_FILE, annotations_path=ANNOTATIONS_FILE):
    """Write into an existing evaluation folder, without replacing its verdict metrics."""
    from .evaluate import _commit

    run_dir, out = Path(run_dir), Path(out)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "summary.csv", keep_default_na=False)
    items = load_test_set(items_path)
    folder = schemes_dir_of(manifest)
    expected_hash = (manifest.get("schemes_version") or {}).get("content_sha256")
    if expected_hash and expected_hash != schemes_hash(folder):
        raise SchemeError("CQ diagnostics: scheme files differ from the recorded run version")
    entries = [e for e in manifest["config"]["models"] if e in set(summary.model)]
    if not entries:
        raise SchemeError("No model entries for CQ diagnostics")
    frames, status = tables(summary, items, load_annotations(annotations_path), load_all(folder),
                            entries, folder == SCHEMES_DIR)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if name == "human_agreement" and not status["human_cq_compatible"]:
            continue  # verdict agreement is independent of the revised CQ run
        frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    inputs = [run_dir / "summary.csv", run_dir / "manifest.json", Path(items_path),
              Path(annotations_path), Path(__file__), *folder.glob("*.yaml"),
              *SCHEMES_DIR.glob("*.yaml")]
    record = {"run": run_dir.name, "commit": _commit(), "schemes": folder.name,
              "date": datetime.now(UTC).isoformat(timespec="seconds"),
              "software_versions": {p: version(p) for p in ("numpy", "pandas", "matplotlib")},
              "entries": entries, **status,
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (out / "cq_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    lines = ["# CQ and human agreement", "", f"Source run: `{run_dir.name}`.", "",
             "Test items and gold scheme condition only. CQ missing/invalid predictions stay in "
             "the human agreement denominator and are omitted from nominal alpha. Human "
             "agreement per CQ requires 20 annotations. Singleton units do not enter alpha; "
             "constant ratings yield undefined alpha. Raw agreement weights observed pairs.", "",
             "Human NA is a separate answer category. Low human agreement may reflect different "
             "judgements of applicability; inspect `human_distribution` before interpreting it "
             "as an error on the binary question.", "",
             "CQ distributions, Brier scores, reached shares and flip sensitivity: "
             "`cq_metrics.csv`. "
             "Flip sensitivity is per model; average across models for the pooled weight. "
             "Model alpha and complete-case unanimity: `cq_agreement.csv`.", "",
             "`cq_calibration.csv` contains ten equal-count bins per model/probability, pooled "
             "over CQs; ties are split deterministically. Brier and calibration use only binary "
             "human answers and available probabilities, already oriented toward yes/positive "
             "by the parser. No probability is fabricated for IDK/NA.", "",
             "Human verdict alpha uses fine terminal labels, never the benchmark gold as an "
             "extra rater. Repeated identical labels across sheets are collapsed; conflicting "
             "and coarse labels are excluded and counted in `cq_manifest.json`.", "",
             "These annotators are not independent of the benchmark construction: verdict "
             "agreement describes this annotation set, not independent validation of its gold.", "",
             "Human CQ comparisons: " + ("original scheme annotations." if status[
                 "human_cq_compatible"] else "unavailable: annotations refer to different CQs."),
             "", "Still unavailable: " + "; ".join(status["missing"]) + ".", ""]
    lines += ["## Questions with the lowest model agreement", "",
              "Descriptive only; these questions have different supports and answer distributions.",
              "", "| Scheme | CQ | Items | Alpha | Unanimous share |",
              "|---|---|---:|---:|---:|"]
    for row in frames["cq_agreement"].sort_values("alpha").head(8).itertuples():
        lines.append(f"| {row.scheme} | {row.cq_id} | {row.items} | {row.alpha:.3f} | "
                     f"{row.unanimous_share:.3f} |")
    if status["human_cq_compatible"]:
        lines += ["", "## Human verdict agreement", "",
                  "| Annotators | Items with paired ratings | Alpha | Raw agreement |",
                  "|---|---:|---:|---:|"]
        for row in frames["human_agreement"].itertuples():
            lines.append(f"| {row.annotators} | {row.units} | {row.alpha:.3f} | "
                         f"{row.raw_agreement:.3f} |")
    else:
        lines += ["", "Human verdict agreement is run-independent; see the original-scheme "
                  "evaluation. It is not duplicated here.", ""]
    if len(frames["cq_calibration"]):
        calibration_plot(frames["cq_calibration"], out / "cq_calibration.svg")
        lines += ["## Reliability diagrams", "", "![Reliability](cq_calibration.svg)", ""]
    (out / "cq_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def calibration_plot(frame, path):
    """Export the pooled equal-count reliability panels as a standalone figure."""
    from matplotlib.figure import Figure

    groups = list(frame.groupby(["model", "probability"]))
    figure = Figure(figsize=(12, 3.4 * ((len(groups) + 2) // 3)), layout="constrained")
    axes = figure.subplots((len(groups) + 2) // 3, 3, squeeze=False).ravel()
    for axis, ((model, source), group) in zip(axes, groups, strict=False):
        axis.plot([0, 1], [0, 1], "--", color="0.6")
        axis.plot(group.mean_probability, group.observed_positive, "o-", color="#1769aa")
        axis.set(xlim=(0, 1), ylim=(0, 1), title=f"{model}: {source}",
                 xlabel="Mean probability", ylabel="Observed positive share")
    for axis in axes[len(groups):]:
        axis.set_visible(False)
    figure.savefig(path, format="svg")
