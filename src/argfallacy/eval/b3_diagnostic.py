"""Isolated, audited B3 transfer experiment; historical evaluations are read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from ..aggregate import good_terminal, load_burden, routes
from ..aggregate.b1 import aggregate_combined
from ..aggregate.b1_inputs import build_dataset
from ..aggregate.crossfit import B1CrossFitter
from ..aggregate.interface import decide
from ..aggregate.tree import TREE_GRID, b3_out_of_fold
from ..annotations import ANNOTATIONS_FILE, ITEMS_FILE, load_annotations
from ..client.run import schemes_dir_of
from ..paths import REPO_ROOT
from ..schemes import SchemeError, enumerate_paths, load_all, traverse
from .b1 import cq_labels
from .canonical import score_fallacies
from .evaluate import Scorer, _read_run, interval, resample_indices, weights
from .folds import assign_folds
from .prior import load_test_set
from .round2 import ORIGINS

B3 = "B3@2.0"
BASELINE = "B1.hard@2.0"
CONTEXT = ("A0:qwen3_8_27b", "A2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)

    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n",
                          encoding="utf-8")


@dataclass(frozen=True)
class DiagnosticConfig:
    name: str
    fit_run: str
    target_run: str
    reference_model: str
    seed: int
    resamples: int
    practical_delta: float
    protocol: str
    baseline_evaluation: str
    context_evaluation: str

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemeError("diagnostic configuration keys differ from the protocol")
        config = cls(**data)
        if Path(config.name).name != config.name or config.name in {".", ".."}:
            raise SchemeError("diagnostic name must be a single directory name")
        if config.resamples < 1 or not 0 <= config.practical_delta <= 1:
            raise SchemeError("invalid bootstrap count or practical delta")
        return config


def export_tree(tree, scheme):
    """Complete numerical model, including unvisited branches and raw leaf counts."""
    if tree.model is None:
        return {"scheme": scheme.scheme_id, "nodes": [], "classes": [], "rules": []}
    fitted = tree.model.tree_
    nodes, rules = [], []
    for node in range(fitted.node_count):
        feature = int(fitted.feature[node])
        leaf = fitted.children_left[node] == -1
        value = fitted.value[node][0].astype(float)
        nodes.append({
            "id": node, "left": int(fitted.children_left[node]),
            "right": int(fitted.children_right[node]), "leaf": bool(leaf),
            "feature": None if leaf else tree.names[feature],
            "feature_index": None if leaf else feature,
            "threshold": None if leaf else float(fitted.threshold[node]),
            "raw_support": int(fitted.n_node_samples[node]),
            "weighted_support": float(fitted.weighted_n_node_samples[node]),
            "distribution": dict(zip(tree.model.classes_, value / value.sum(), strict=True)),
            "raw_class_counts": dict(tree.raw_leaves.get(node, {})),
        })

    def walk(node, path):
        current = nodes[node]
        if current["leaf"]:
            cqs = [step["cq"] for step in path]
            rules.append({
                "leaf": node, "conditions": path,
                "verdict": decide(current["distribution"], 0, "").verdict,
                "raw_support": current["raw_support"],
                "raw_class_counts": current["raw_class_counts"],
                "repeated_cq": len(set(cqs)) < len(cqs),
                "uses_outside": any(step["kind"] == "outside" for step in path),
                "tiny_leaf": current["raw_support"] < 5,
                "semantic_status": "requires argumentation review; not independently validated",
            })
            return
        cq, kind = current["feature"].split()
        for child, operator in ((current["left"], "<="), (current["right"], ">")):
            step = {"cq": cq, "kind": kind, "operator": operator,
                    "threshold": current["threshold"],
                    "cq_text": scheme.cqs[cq].text,
                    "cq_note": scheme.cqs[cq].note,
                    "first_arc": scheme.answer_space(cq)[0]}
            walk(child, [*path, step])

    walk(0, [])
    exported = {"scheme": scheme.scheme_id, "version": "2.0",
                "features": list(tree.names), "classes": list(tree.model.classes_),
                "nodes": nodes, "rules": rules, "comparison_dtype": "float32"}
    manual_paths = list(enumerate_paths(scheme))
    for rule in rules:
        tested = {step["cq"] for step in rule["conditions"]}
        gate_flags = {}
        for cq in tested:
            contexts = [dict(steps[:n]) for steps, _ in manual_paths
                        for n, (visited, _) in enumerate(steps) if visited == cq]
            if contexts:
                gates = set.intersection(*(set(context) for context in contexts))
                missing = gates - tested
                if missing:
                    gate_flags[cq] = sorted(missing)
        rule["manual_graph_gates_not_tested"] = gate_flags
        rule["graph_gate_note"] = "routing differences are review flags, not semantic violations"
        rule["binary_corner_witnesses"] = 0
        rule["manual_corner_terminals"] = {}
    by_leaf = {rule["leaf"]: rule for rule in rules}
    corner_counts = {leaf: Counter() for leaf in by_leaf}
    for bits in product((0.0, 1.0), repeat=len(scheme.cq_order)):
        row = np.array([[bit, 0.0] for bit in bits]).ravel()
        _, _, leaf, _ = replay_tree(exported, row)
        answers = {cq: scheme.answer_space(cq)[0 if bit == 1 else 1]
                   for cq, bit in zip(scheme.cq_order, bits, strict=True)}
        terminal = traverse(scheme, answers).verdict
        by_leaf[leaf]["binary_corner_witnesses"] += 1
        corner_counts[leaf][terminal] += 1
    for leaf, counts in corner_counts.items():
        by_leaf[leaf]["manual_corner_terminals"] = dict(counts)
    return exported


def replay_tree(exported, row):
    """Follow full-precision exported thresholds on sklearn's float32 inputs."""
    if not exported["nodes"]:
        return None, {}, None, []
    row = np.asarray(row, dtype=np.float32)
    node, path = 0, []
    while not exported["nodes"][node]["leaf"]:
        current = exported["nodes"][node]
        value = float(row[current["feature_index"]])
        left = value <= current["threshold"]
        path.append({"node": node, "feature": current["feature"], "value": value,
                     "threshold": current["threshold"], "operator": "<=" if left else ">"})
        node = current["left"] if left else current["right"]
    distribution = exported["nodes"][node]["distribution"]
    return decide(distribution, 0, "").verdict, distribution, node, path


def score_tables(items, predictions, seed, resamples):
    """Paired canonical scoring with shared draws and fixed common-coverage selection."""
    ids = items.item_id.tolist()
    by_system = {s: [None if pd.isna(v) else v
                     for v in g.set_index("item_id").reindex(ids).verdict]
                 for s, g in predictions.groupby("system")}
    if any(len(g) != len(ids) or set(g.item_id) != set(ids)
           for _, g in predictions.groupby("system")):
        raise SchemeError("prediction population is incomplete or duplicated")
    common = np.array([a is not None and b is not None
                       for a, b in zip(by_system[B3], by_system[BASELINE], strict=True)])
    masks = {"all": np.ones(len(ids), bool), "common_coverage": common,
             "reachable_gold": items.reachable_gold.to_numpy(bool),
             "unreachable_gold": ~items.reachable_gold.to_numpy(bool),
             "without_nlas": items.source.str.lower().ne("nlas").to_numpy(),
             "gold_non_good": items.gold_fallacy.ne(good_terminal()).to_numpy()}
    for column in ("gold_scheme", "source", "origin"):
        for value in sorted(items[column].unique()):
            masks[f"{column}:{value}"] = items[column].eq(value).to_numpy()
    draws = weights(resample_indices(seed, len(ids), resamples, 1), len(ids))
    estimates, differences, classes = [], [], []
    for subset, mask in masks.items():
        if not mask.any():
            continue
        truth = items.loc[mask, "gold_fallacy"].tolist()
        own_weights = draws[:, mask]
        nonempty = own_weights.sum(axis=1) > 0
        observed = Scorer(truth, np.ones((1, len(truth))))
        sampled = Scorer(truth, own_weights[nonempty])
        scores = {}
        for system, full in by_system.items():
            predicted = [full[j] for j in np.flatnonzero(mask)]
            point = observed.metrics(predicted)
            boots = sampled.metrics(predicted)
            for scorer, values in ((observed, point), (sampled, boots)):
                f1, supported = scorer.class_f1(predicted)
                supported[:, scorer.good] = False
                values["macro_f1_without_good_class"] = np.divide(
                    (f1 * supported).sum(axis=1), supported.sum(axis=1),
                    out=np.full(len(f1), np.nan), where=supported.sum(axis=1) > 0)
            scores[system] = point, boots
            for metric in point:
                valid = boots[metric][np.isfinite(boots[metric])]
                low, high = interval(valid) if len(valid) else (None, None)
                estimates.append({"subset": subset, "system": system, "n_items": int(mask.sum()),
                                  "metric": metric, "estimate": float(point[metric][0]),
                                  "low": low, "high": high, "replicates": len(valid),
                                  "empty_draws": int((~nonempty).sum())})
            frame = pd.DataFrame({"item_id": items.loc[mask, "item_id"],
                                  "predicted": predicted})
            canonical = score_fallacies(frame, items)
            if abs(canonical.macro_f1 - point["macro_f1"][0]) > 1e-12:
                raise SchemeError("canonical scorer disagreement")
            if subset == "all":
                classes.extend({"system": system, **row}
                               for row in canonical.classes.to_dict("records"))
        for metric in scores[B3][0]:
            a, b = scores[B3], scores[BASELINE]
            deltas = a[1][metric] - b[1][metric]
            valid = deltas[np.isfinite(deltas)]
            low, high = interval(valid) if len(valid) else (None, None)
            differences.append({"subset": subset, "metric": metric,
                                "difference": float(a[0][metric][0] - b[0][metric][0]),
                                "low": low, "high": high, "replicates": len(valid),
                                "role": "primary" if subset == "all" and metric == "macro_f1"
                                else "descriptive"})
    return pd.DataFrame(estimates), pd.DataFrame(differences), pd.DataFrame(classes)


def _peak_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
                 "pagefile", "peak_pagefile")]

        memory = Memory()
        memory.cb = ctypes.sizeof(memory)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        read_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        read_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        read_memory.restype = wintypes.BOOL
        if read_memory(process(), ctypes.byref(memory), memory.cb):
            return memory.peak
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def run(config_path, *, out_root=None, runs_dir=None, progress=print):
    """Freeze provenance, fit the single approved family, and export its full audit."""
    config_path = Path(config_path).resolve()
    config = DiagnosticConfig.load(config_path)
    runs_dir = Path(runs_dir or REPO_ROOT / "runs")
    out = Path(out_root or REPO_ROOT / "evaluations") / config.name
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    protocol = REPO_ROOT / config.protocol
    shutil.copyfile(config_path, out / "configuration.yaml")
    shutil.copyfile(protocol, out / "protocol.md")
    protected = [ITEMS_FILE, ANNOTATIONS_FILE, config_path, protocol]
    for run_id in (config.fit_run, config.target_run):
        protected.extend(runs_dir / run_id / f for f in ("manifest.json", "summary.csv"))
    for folder in ("schemes/v1", "schemes/v2", "schemes/burden"):
        protected.extend((REPO_ROOT / folder).glob("*.yaml"))
    protected.extend(REPO_ROOT / f"prompts/stage2_{v}_answers.yaml" for v in ("v2", "v3"))
    baseline_folder = REPO_ROOT / config.baseline_evaluation
    context_folder = REPO_ROOT / config.context_evaluation
    for folder in (baseline_folder, context_folder):
        protected.extend(folder / f for f in ("manifest.json", "verdicts.csv", "main_table.csv"))
    protected.append(baseline_folder / "cq_combined.csv")
    input_hashes = {str(p.resolve()): digest(p) for p in protected}
    manifest = {"status": "running", "started_utc": datetime.now(UTC).isoformat(),
                "classification": "exploratory; adaptive reuse of benchmark",
                "input_sha256": input_hashes, "protocol_sha256": digest(protocol),
                "code_sha256": {str(p.relative_to(REPO_ROOT)): digest(p)
                                for p in (REPO_ROOT / "src/argfallacy").rglob("*.py")},
                "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
                "python": platform.python_version(), "grid": list(TREE_GRID),
                "thread_pools_before": threadpool_info(), "fit_threads": 1,
                "practical_delta": config.practical_delta,
                "independent_semantic_adjudication": False,
                "repeat_fit_stability": "not executed; five outer models only"}
    json_write(out / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            _fit_and_report(config, runs_dir, out, manifest, progress)
        changed = [p for p, sha in input_hashes.items() if digest(p) != sha]
        if changed:
            raise SchemeError(f"protected inputs changed: {changed}")
        manifest.update(status="complete", protected_inputs_unchanged=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.update(elapsed_seconds=time.perf_counter() - start,
                        peak_process_memory_bytes=_peak_memory(),
                        finished_utc=datetime.now(UTC).isoformat())
        manifest["output_sha256"] = {str(p.relative_to(out)): digest(p)
                                     for p in out.rglob("*")
                                     if p.is_file() and p.name != "manifest.json"}
        json_write(out / "manifest.json", manifest)
    return out


def _fit_and_report(config, runs_dir, out, manifest, progress):
    items = load_test_set(ITEMS_FILE).sort_values("item_id").reset_index(drop=True)
    if len(items) != 601:
        raise SchemeError("the approved population is 601 arguments")
    if not set(items.source.str.lower()) <= set(ORIGINS):
        raise SchemeError("unknown gold origin")
    datasets = []
    for run_id, expected in ((config.fit_run, "1.1"), (config.target_run, "2.0")):
        run_manifest, summary = _read_run(runs_dir, run_id)
        folder = schemes_dir_of(run_manifest)
        schemes = load_all(folder)
        if {str(yaml.safe_load(p.read_text(encoding="utf-8"))["version"])
            for p in folder.glob("*.yaml")} != {expected}:
            raise SchemeError(f"{run_id}: unexpected scheme version")
        stage = summary[(summary.stage == "stage2") & (summary.scheme_condition == "gold")]
        if stage.duplicated(["model", "item_id", "cq_id"]).any():
            raise SchemeError(f"{run_id}: duplicate CQ rows")
        expected_rows = sum(len(schemes[s].cq_order) for s in items.gold_scheme) * 5
        if len(stage[stage.item_id.isin(items.item_id)]) != expected_rows:
            raise SchemeError(f"{run_id}: incomplete five-model CQ grid")
        burden = load_burden(folder.name, schemes)
        datasets.append(build_dataset(
            schemes, items[["item_id", "gold_scheme", "gold_fallacy"]], summary,
            run_manifest, config.reference_model,
            {sid: routes(scheme, burden[sid]) for sid, scheme in schemes.items()}))
    fit_data, target = datasets
    labels = cq_labels(load_annotations(ANNOTATIONS_FILE), items, fit_data.schemes)
    if len(labels) != 2656 or fit_data.signals != target.signals:
        raise SchemeError("CQ supervision population or transfer signal ordering changed")
    folds = assign_folds(items, config.seed)
    saved = pd.read_csv(REPO_ROOT / config.baseline_evaluation / "verdicts.csv")
    old = saved[saved.system == BASELINE].set_index("item_id").reindex(items.item_id)
    if len(old) != len(items) or old.fold.isna().any():
        raise SchemeError("saved hard baseline does not cover all test arguments")
    if any(int(old.loc[i, "fold"]) != folds[i] for i in items.item_id):
        raise SchemeError("historical fold assignment differs")
    json_write(out / "folds.json", folds)
    source = B1CrossFitter(target, labels, config.seed, fit_data=fit_data)
    baseline, feature_frames = {}, []
    for fold in sorted(set(folds.values())):
        progress(f"Baseline: outer fold {fold + 1}/5", flush=True)
        train = sorted(i for i in folds if folds[i] != fold)
        held = sorted(i for i in folds if folds[i] == fold)
        inputs = source.fit(train)
        if not np.isfinite(inputs.p).all() or not np.isfinite(target.outside).all():
            raise SchemeError("non-finite target signals")
        baseline.update({i: a for i, a in aggregate_combined(target, inputs.p, "hard").items()
                         if i in held})
        own = inputs.rows(held).assign(fold=fold)
        feature_frames.append(own)
    feature_frame = pd.concat(feature_frames).sort_values(["item_id", "cq_id"])
    feature_frame.to_csv(out / "outer_features.csv", index=False)
    drift = [i for i in items.item_id if baseline[i].verdict !=
             (None if pd.isna(old.loc[i, "verdict"]) else old.loc[i, "verdict"])]
    manifest["baseline_verdict_drift"] = drift
    if drift:
        raise SchemeError(f"baseline verdict replay differs on {len(drift)} arguments")
    historical = pd.read_csv(REPO_ROOT / config.baseline_evaluation / "cq_combined.csv")
    # Historical CQ rows are written once for the shared soft/hard combination.
    historical = historical[historical.system == "B1@2.0"]
    paired = feature_frame.merge(historical, on=["item_id", "scheme", "cq_id"],
                                 suffixes=("_new", "_saved"), validate="one_to_one")
    if len(paired) != len(target.pairs):
        raise SchemeError("saved baseline CQ population differs")
    manifest["baseline_max_saved_signal_difference"] = {
        c: float(np.max(np.abs(paired[c + "_new"] - paired[c + "_saved"])))
        for c in ("p_first", "outside")}
    json_write(out / "manifest.json", manifest)
    audit_path = out / "fit_scores.jsonl"
    with audit_path.open("w", encoding="utf-8") as audit_file:
        def audit(event):
            audit_file.write(json.dumps(event) + "\n")
            audit_file.flush()
            if event["phase"] == "outer":
                progress(f"B3: fold {event['fold'] + 1}/5, {event['scheme']}", flush=True)

        result = b3_out_of_fold(source, folds, name=B3, audit=audit)
    with (out / "fitted_models.pkl").open("wb") as file:
        pickle.dump({"trees": result.models, "b1": source.models}, file)
    json_write(out / "upstream_choices.json", source.choices)
    json_write(out / "tree_choices.json", result.choices)
    exports, rules, paths = {}, [], []
    for fold, trees in result.models.items():
        train = sorted(i for i in folds if folds[i] != fold)
        inputs = source.fit(train)
        for sid, tree in trees.items():
            exported = export_tree(tree, target.schemes[sid])
            exports[f"{fold}:{sid}"] = exported
            rules.extend({"fold": fold, "scheme": sid, **rule} for rule in exported["rules"])
            own = items[(items.item_id.map(folds) == fold) & (items.gold_scheme == sid)]
            for item, row in zip(own.item_id, inputs.tree_features(own.item_id), strict=True):
                verdict, distribution, leaf, path = replay_tree(exported, row)
                aggregation = result.aggregations[item]
                if verdict != aggregation.verdict or any(
                    abs(distribution[k] - aggregation.distribution[k]) > 1e-12
                    for k in distribution
                ) or (tree.model is not None and leaf != int(tree.model.apply([row])[0])):
                    raise SchemeError(f"exported tree replay failed: {item}")
                paths.append({"item_id": item, "fold": fold, "scheme": sid,
                              "leaf": leaf, "path": path, "verdict": verdict})
    json_write(out / "trees.json", exports)
    json_write(out / "rules.json", rules)
    json_write(out / "oof_paths.json", paths)
    manifest["export_replay_verified_items"] = len(paths)
    prediction_rows = []
    for name, aggregations in ((B3, result.aggregations), (BASELINE, baseline)):
        for item, aggregation in aggregations.items():
            prediction_rows.append({"system": name, "item_id": item, "fold": folds[item],
                                    "verdict": aggregation.verdict,
                                    "confidence": aggregation.confidence,
                                    "abstention": aggregation.abstention,
                                    "distribution": json.dumps(aggregation.distribution),
                                    "explanation": aggregation.explanation})
    context = pd.read_csv(REPO_ROOT / config.context_evaluation / "verdicts.csv")
    for name in CONTEXT:
        rows = context[context.system == name].copy()
        if len(rows) != len(items) or set(rows.item_id) != set(items.item_id):
            raise SchemeError(f"incomplete contextual predictions: {name}")
        rows["system"] = name + "@2.0"
        prediction_rows.extend(rows.to_dict("records"))
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(out / "verdicts.csv", index=False)
    items["origin"] = items.source.str.lower().map(ORIGINS)
    items["reachable_gold"] = [row.gold_fallacy in target.schemes[row.gold_scheme].terminals
                               for row in items.itertuples()]
    items.loc[~items.reachable_gold].to_csv(out / "unreachable_gold.csv", index=False)
    progress("Scoring: paired bootstrap and descriptive breakdowns", flush=True)
    estimates, differences, classes = score_tables(items, predictions, config.seed,
                                                  config.resamples)
    estimates.to_csv(out / "metrics.csv", index=False)
    differences.to_csv(out / "differences.csv", index=False)
    classes.to_csv(out / "per_class.csv", index=False)
    ledger = items[["item_id", "gold_scheme", "gold_fallacy", "source", "origin",
                    "reachable_gold"]].copy()
    for name, aggregations in ((B3, result.aggregations), (BASELINE, baseline)):
        ledger[name] = ledger.item_id.map({i: a.verdict for i, a in aggregations.items()})
        ledger[name + "_correct"] = ledger[name] == ledger.gold_fallacy
    ledger.to_csv(out / "paired_items.csv", index=False)
    primary = differences[(differences.subset == "all") &
                          (differences.metric == "macro_f1")].iloc[0].to_dict()
    manifest["primary_result"] = primary
    manifest["practical_interest"] = primary["difference"] >= config.practical_delta
    manifest["positive_paired_interval"] = primary["low"] > 0
    _summary(out, manifest, estimates, rules)


def _summary(out, manifest, estimates, rules):
    rows = estimates[(estimates.subset == "all") & estimates.metric.isin(
        ["macro_f1", "accuracy", "coverage"])]
    table = rows.pivot(index="system", columns="metric", values="estimate")
    primary = manifest["primary_result"]
    lines = ["# B3 on schemes 2.0: exploratory diagnostic", "",
             "The protocol was frozen before these fits; the benchmark had already been explored.",
             "", "| System | Accuracy | Macro-F1 | Coverage |",
             "| --- | ---: | ---: | ---: |"]
    for system, row in table.iterrows():
        lines.append(f"| {system} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.coverage:.4f} |")
    lines += ["", f"Primary paired Macro-F1 difference: {primary['difference']:+.4f}, "
              f"IC95% [{primary['low']:+.4f}, {primary['high']:+.4f}].",
              f"Practical-interest threshold +0.03 reached: {manifest['practical_interest']}. "
              f"Lower interval above zero: {manifest['positive_paired_interval']}.", "",
              "Qwen A0/A2 are context only. All primary cases, including abstentions and "
              "unreachable gold, remain scored. Conditional common coverage and origin/schema/"
              "source sensitivities are in metrics.csv and differences.csv.", "",
              f"Full exported model replay verified on {manifest['export_replay_verified_items']} "
              f"OOF items. Exported rules: {len(rules)}; rules with repeated CQ: "
              f"{sum(r['repeated_cq'] for r in rules)}; with outside mass: "
              f"{sum(r['uses_outside'] for r in rules)}.", "",
              "Predictive improvement does not certify Walton validity. rules.json preserves "
              "every full-precision condition and its CQ text. Outside mass conflates NA and "
              "indeterminacy; numerical thresholds do not establish prerequisite answers. "
              "Independent semantic adjudication and repeat-fit stability were not performed.", "",
              "Bootstrap intervals condition on saved OOF predictions and omit retraining "
              "variability. Sparse classes, shared benchmark adaptation, transferred B1 "
              "coefficients, and diagram-dependent gold limit interpretation.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--out-root")
    args = parser.parse_args()
    print(run(args.config, out_root=args.out_root))


if __name__ == "__main__":
    main()
