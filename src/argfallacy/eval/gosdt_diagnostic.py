"""The fixed matched-input GOSDT/CART experiment, with resumable audit records."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from ..aggregate import load_burden, routes
from ..aggregate.b1 import aggregate_combined
from ..aggregate.b1_inputs import build_dataset
from ..aggregate.crossfit import B1CrossFitter
from ..aggregate.interface import macro_f1
from ..aggregate.sparse_tree import (
    GRID,
    analytic_tree,
    constant_mask,
    encode,
    fit_cart,
    fit_gosdt,
    objective,
    predicates,
    replay,
)
from ..annotations import ANNOTATIONS_FILE, ITEMS_FILE, load_annotations
from ..client.run import default_runs_dir, schemes_dir_of
from ..paths import REPO_ROOT
from ..schemes import SchemeError, enumerate_paths, load_all
from .b1 import cq_labels
from .b3_diagnostic import _peak_memory, digest, json_write, score_tables
from .evaluate import _read_run
from .folds import assign_folds
from .gosdt_process import SolverProcess, available_memory, require_memory
from .prior import load_test_set
from .round2 import ORIGINS

GOSDT = "GOSDT@2.0"
CART = "CART.binary@2.0"
MANUAL = "B1.hard@2.0"
HISTORICAL = "B3@2.0"


def configuration(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "seed": 20261019,
        "resamples": 10000,
        "practical_delta": 0.03,
        "depths": [2, 3, 4],
        "leaf_coefficients": [2, 5],
        "inner_folds": 3,
        "time_limit_seconds": 60,
        "memory_limit_bytes": 4_000_000_000,
        "grace_seconds": 5,
        "minimum_free_bytes": 3 * 1024**3,
    }
    required = set(expected) | {
        "name",
        "fit_run",
        "target_run",
        "reference_model",
        "protocol",
        "baseline_evaluation",
        "historical_evaluation",
    }
    if set(config) != required or any(config[k] != v for k, v in expected.items()):
        raise SchemeError("configuration differs from the frozen spec 12 design")
    if Path(config["name"]).name != config["name"] or config["name"] in {".", ".."}:
        raise SchemeError("experiment name must be a single directory component")
    return config


def runtime_probe(python, vendor, folder):
    x = np.array([[0, 0]] * 8 + [[0, 1]] * 4 + [[1, 0]] * 2, bool)
    y = np.array(["a"] * 8 + ["b"] * 4 + ["c"] * 2)
    process = SolverProcess(python, vendor, folder)
    result = process("runtime_probe", x, y, 2, 2, evaluation_x=x[::-1])
    if (
        not result.get("valid")
        or not result.get("evaluation_replay_verified")
        or result.get("runtime", {}).get("packages", {}).get("gosdt") != "1.0.4"
    ):
        raise SchemeError(f"GOSDT runtime probe failed: {result}")
    return result


def preflight(config, python, vendor, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "available_physical_bytes": available_memory(),
        "minimum_free_bytes": config["minimum_free_bytes"],
        "ready": False,
        "scope": "resource/runtime only; no experimental inputs fitted",
    }
    try:
        require_memory(config["minimum_free_bytes"])
        report["runtime"] = runtime_probe(python, vendor, folder)
        report["ready"] = True
    except (MemoryError, OSError, ValueError, SchemeError) as error:
        report["reason"] = str(error)
    json_write(folder / "preflight.json", report)
    return report


def _load(config):
    items = load_test_set(ITEMS_FILE).sort_values("item_id").reset_index(drop=True)
    if len(items) != 601 or not set(items.source.str.lower()) <= set(ORIGINS):
        raise SchemeError("unexpected test population/origins")
    datasets = []
    for name, version in ((config["fit_run"], "1.1"), (config["target_run"], "2.0")):
        manifest, summary = _read_run(default_runs_dir(), name)
        folder = schemes_dir_of(manifest)
        schemes = load_all(folder)
        versions = {
            str(yaml.safe_load(p.read_text(encoding="utf-8"))["version"])
            for p in folder.glob("*.yaml")
        }
        stage = summary[(summary.stage == "stage2") & (summary.scheme_condition == "gold")]
        own = stage[stage.item_id.isin(items.item_id)]
        expected = 5 * sum(len(schemes[s].cq_order) for s in items.gold_scheme)
        if (
            versions != {version}
            or len(own) != expected
            or own.duplicated(["model", "item_id", "cq_id"]).any()
        ):
            raise SchemeError(f"invalid version/grid in {name}")
        burden = load_burden(folder.name, schemes)
        datasets.append(
            build_dataset(
                schemes,
                items[["item_id", "gold_scheme", "gold_fallacy"]],
                summary,
                manifest,
                config["reference_model"],
                {sid: routes(scheme, burden[sid]) for sid, scheme in schemes.items()},
            )
        )
    original, target = datasets
    labels = cq_labels(load_annotations(ANNOTATIONS_FILE), items, original.schemes)
    if len(labels) != 2656 or original.signals != target.signals:
        raise SchemeError("CQ supervision or transfer ordering changed")
    return items, original, target, labels


def _snapshot(config_path, config, python, vendor):
    files = [ITEMS_FILE, ANNOTATIONS_FILE, Path(config_path), REPO_ROOT / config["protocol"]]
    for run in (config["fit_run"], config["target_run"]):
        files += [default_runs_dir() / run / f for f in ("manifest.json", "summary.csv")]
    for directory in ("schemes/v1", "schemes/v2", "schemes/burden", "labels", "prompts"):
        files += list((REPO_ROOT / directory).glob("*.yaml"))
    for directory in (config["baseline_evaluation"], config["historical_evaluation"]):
        files += [REPO_ROOT / directory / f for f in ("manifest.json", "verdicts.csv")]
    code = {
        str(p.relative_to(REPO_ROOT)): digest(p)
        for p in (REPO_ROOT / "src/argfallacy").rglob("*.py")
    }
    vendor = Path(vendor)
    runtime_files = [p for pattern in ("gosdt/*.py", "gosdt/*.pyd") for p in vendor.glob(pattern)]
    if not runtime_files:
        raise SchemeError("GOSDT vendor package is unavailable")
    return {
        "inputs": {str(p.resolve()): digest(p) for p in files},
        "code": code,
        "runtime": {str(p.resolve()): digest(p) for p in [Path(python), *runtime_files]},
        "driver_packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "pandas", "pyyaml", "threadpoolctl")
        },
        "config": config,
    }


def _checkpoint(out, manifest):
    manifest["outputs"] = {
        str(p.relative_to(out)): digest(p)
        for p in out.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    manifest["updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    temp = out / "manifest.tmp"
    json_write(temp, manifest)
    temp.replace(out / "manifest.json")


def select_candidate(records):
    def key(record):
        return (
            record["total_leaves"],
            record["actual_depth"],
            record["depth"],
            -record["coefficient"],
        )

    best = max(record["macro_f1"] for record in records)
    return min((r for r in records if r["macro_f1"] >= best - 1e-12), key=key)


def _fit(method, process, key, x, y, depth, coefficient, seed, evaluation_x):
    started = time.monotonic()
    if method == CART:
        result = fit_cart(x, y, depth, coefficient, seed, evaluation_x=evaluation_x)
    else:
        cost = objective(y, coefficient)
        tree, _ = analytic_tree(x, y, cost)
        if tree is not None:
            result = fit_gosdt(x, y, depth, coefficient, 60, evaluation_x=evaluation_x)
        else:
            result = process(key, x, y, depth, coefficient, evaluation_x=evaluation_x)
    # Wall time excludes a reused checkpoint, whose original worker timing is retained.
    result.setdefault("elapsed_seconds", time.monotonic() - started)
    return result


def group_fit(source, outer_train, held, scheme, fold, out, process, seed):
    """Only the supplied outer training context can label or tune this group."""
    metadata = source.data.items.set_index("item_id")
    eligible = sorted(
        i
        for i in outer_train
        if metadata.loc[i, "gold_scheme"] == scheme.scheme_id
        and metadata.loc[i, "gold_fallacy"] in scheme.terminals
    )

    def labels(ids):
        return metadata.loc[list(ids), "gold_fallacy"].to_numpy(str)

    group = f"{fold}_{scheme.scheme_id}"
    partitions = []
    if len(eligible) >= 2 and len(set(labels(eligible))) > 1:
        split = KFold(min(3, len(eligible)), shuffle=True, random_state=seed)
        for inner, (train_index, val_index) in enumerate(split.split(eligible)):
            train = [eligible[j] for j in train_index]
            validation = [eligible[j] for j in val_index]
            context = sorted(set(outer_train) - set(validation))
            training, testing = source.partition(context, validation)
            continuous_train = training.tree_features(train)
            continuous_val = testing.tree_features(validation)
            x, v = encode(continuous_train), encode(continuous_val)
            mask = constant_mask(x)
            partitions.append((train, validation, x[:, mask], v[:, mask], mask, context))
            np.savez_compressed(
                out / f"{group}_inner{inner}.npz",
                x=x,
                v=v,
                mask=mask,
                train=np.array(train),
                validation=np.array(validation),
                continuous_train=continuous_train,
                continuous_validation=continuous_val,
            )
    candidate_records, all_fits = {}, []
    incomplete = False
    for method in (GOSDT, CART):
        candidates = []
        for depth, coefficient in GRID:
            predictions, truth, leaves, depths = [], [], [], []
            valid = True
            for inner, (train, validation, x, v, mask, context) in enumerate(partitions):
                result = _fit(
                    method,
                    process,
                    f"{group}_inner{inner}_d{depth}_a{coefficient}",
                    x,
                    labels(train),
                    depth,
                    coefficient,
                    seed,
                    v,
                )
                own = {
                    "method": method,
                    "inner": inner,
                    "depth": depth,
                    "coefficient": coefficient,
                    "training_items": train,
                    "validation_items": validation,
                    "b1_context": context,
                    "mask": mask.tolist(),
                    "result": result,
                }
                if result.get("valid"):
                    predicted, _ = replay(result["tree"], v)
                    if (
                        not result.get("evaluation_replay_verified")
                        or predicted != result["evaluation_predictions"]
                    ):
                        raise SchemeError("inner native/exported prediction mismatch")
                    own["predictions"] = predicted
                    predictions.extend(predicted)
                    truth.extend(labels(validation).tolist())
                    leaves.append(result["leaves"])
                    depths.append(result["depth"])
                else:
                    valid = False
                all_fits.append(own)
                json_write(out / f"{group}_cv.json", all_fits)
            candidates.append(
                {
                    "depth": depth,
                    "coefficient": coefficient,
                    "complete": valid,
                    "macro_f1": macro_f1(truth, predictions) if valid else None,
                    "total_leaves": sum(leaves),
                    "actual_depth": max(depths, default=0),
                }
            )
            incomplete |= not valid
        candidate_records[method] = candidates
    if incomplete:
        return {
            "complete": False,
            "reason": "incomplete_inner_grid",
            "fits": all_fits,
            "candidates": candidate_records,
        }
    training, testing = source.partition(outer_train, held)
    continuous_train = (
        training.tree_features(eligible) if eligible else np.empty((0, len(scheme.cq_order) * 2))
    )
    continuous_held = testing.tree_features(held)
    x = encode(continuous_train) if eligible else np.empty((0, len(scheme.cq_order) * 5), bool)
    v = encode(continuous_held)
    mask = constant_mask(x)
    np.savez_compressed(
        out / f"{group}_outer.npz",
        x=x,
        v=v,
        mask=mask,
        train=np.array(eligible),
        held=np.array(held),
        continuous_train=continuous_train,
        continuous_held=continuous_held,
    )
    results = {}
    for method in (GOSDT, CART):
        choice = select_candidate(candidate_records[method])
        result = _fit(
            method,
            process,
            f"{group}_outer_d{choice['depth']}_a{choice['coefficient']}",
            x[:, mask],
            labels(eligible),
            choice["depth"],
            choice["coefficient"],
            seed,
            v[:, mask],
        )
        if "tree" not in result:
            return {
                "complete": False,
                "reason": "invalid_outer_fit",
                "fits": all_fits,
                "candidates": candidate_records,
                "failed_result": result,
            }
        predictions, paths = replay(result["tree"], v[:, mask])
        if result.get("valid") and (
            not result.get("evaluation_replay_verified")
            or predictions != result["evaluation_predictions"]
        ):
            raise SchemeError("outer native/exported prediction mismatch")
        results[method] = {
            "choice": choice,
            "model": result,
            "predictions": predictions,
            "paths": paths,
            "input_sha256": hashlib.sha256(v[:, mask].tobytes()).hexdigest(),
            "training_input_sha256": hashlib.sha256(x[:, mask].tobytes()).hexdigest(),
        }
    complete = all(r["model"].get("valid") for r in results.values())
    return {
        "complete": complete,
        "reason": None if complete else "invalid_outer_fit",
        "results": results,
        "candidates": candidate_records,
        "fits": all_fits,
        "training_items": eligible,
        "held_items": held,
        "outer_context": outer_train,
        "excluded_training_items": sorted(
            i
            for i in outer_train
            if metadata.loc[i, "gold_scheme"] == scheme.scheme_id and i not in set(eligible)
        ),
        "constant_mask": mask.tolist(),
        "predicates": predicates(scheme),
    }


def export_rules(group, scheme, fold):
    rules = []
    all_predicates = [
        p for p, keep in zip(group["predicates"], group["constant_mask"], strict=True) if keep
    ]
    manual_paths = list(enumerate_paths(scheme))
    for method, result in group["results"].items():

        def walk(node, conditions, method=method):
            if "verdict" in node:
                cqs = [c["cq"] for c in conditions]
                tested = set(cqs)
                missing = {}
                for cq in tested:
                    contexts = [
                        set(c for c, _ in steps[:j])
                        for steps, _ in manual_paths
                        for j, (c, _) in enumerate(steps)
                        if c == cq
                    ]
                    gates = set.intersection(*contexts) if contexts else set()
                    if gates - tested:
                        missing[cq] = sorted(gates - tested)
                rules.append(
                    {
                        "system": method,
                        "fold": fold,
                        "scheme": scheme.scheme_id,
                        "conditions": conditions,
                        "verdict": node["verdict"],
                        "raw_support": node["raw_support"],
                        "class_counts": node["class_counts"],
                        "repeated_cq": len(cqs) != len(tested),
                        "uses_outside": any(c["kind"] == "outside" for c in conditions),
                        "manual_gates_not_tested": missing,
                        "semantic_status": "review flags; not independent Walton adjudication",
                    }
                )
                return
            predicate = all_predicates[node["column"]]
            for branch, operator in (("true", ">"), ("false", "<=")):
                walk(node[branch], [*conditions, {**predicate, "operator": operator}])

        walk(result["model"]["tree"], [])
    return rules


def fit_reports(groups, out):
    """Flatten the saved audit; no refitting or model selection occurs here."""
    fits, candidates, selected = [], [], []
    for path in sorted(groups.glob("*.json")):
        if path.stem.endswith("_cv"):
            continue
        group = json.loads(path.read_text(encoding="utf-8"))
        if not group.get("complete"):
            continue
        for method, records in group["candidates"].items():
            candidates.extend({"group": path.stem, "system": method, **r} for r in records)
        records = [(r["method"], f"inner{r['inner']}", r["result"], r) for r in group["fits"]]
        records += [
            (method, "outer", r["model"], r["choice"]) for method, r in group["results"].items()
        ]
        for method, stage, result, configuration in records:
            native = result.get("native") or {}
            fits.append(
                {
                    "group": path.stem,
                    "system": method,
                    "stage": stage,
                    "configured_depth": configuration["depth"],
                    "coefficient": configuration["coefficient"],
                    **{
                        k: result.get(k)
                        for k in (
                            "status",
                            "valid",
                            "certified",
                            "depth",
                            "leaves",
                            "risk",
                            "objective",
                            "elapsed_seconds",
                        )
                    },
                    "lambda": result.get("cost", {}).get("lambda"),
                    "n_train": result.get("cost", {}).get("n"),
                    "lower_bound": native.get("lower"),
                    "upper_bound": native.get("upper"),
                    "incumbent_gap": native.get("incumbent_gap"),
                }
            )
        for method, result in group["results"].items():
            selected.append(
                {
                    "group": path.stem,
                    "system": method,
                    **result["choice"],
                    "selected_depth": result["model"]["depth"],
                    "selected_leaves": result["model"]["leaves"],
                }
            )
    for name, rows in (
        ("fit_statuses", fits),
        ("cv_scores", candidates),
        ("selected_configs", selected),
    ):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)


def restore_upstream(source, out, fold):
    """Restore only local artifacts already verified by the run manifest."""
    models, choices = out / f"upstream_fold{fold}.pkl", out / f"upstream_choices_fold{fold}.json"
    if models.exists():
        with models.open("rb") as file:
            source.models = pickle.load(file)
        source.choices = json.loads(choices.read_text(encoding="utf-8"))


def run(config_path, python, vendor, *, resume=False, progress=print):
    config = configuration(config_path)
    require_memory(config["minimum_free_bytes"])
    snapshot = _snapshot(config_path, config, python, vendor)
    out = REPO_ROOT / "evaluations" / config["name"]
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if manifest["snapshot"] != snapshot:
            raise SchemeError("resume inputs/code/runtime differ from the frozen checkpoint")
        for path, sha in manifest["outputs"].items():
            if digest(out / path) != sha:
                raise SchemeError(f"checkpoint hash differs: {path}")
        if manifest["status"] == "complete":
            return out
    else:
        out.mkdir(parents=True)
        manifest = {
            "snapshot": snapshot,
            "status": "running",
            "python": platform.python_version(),
            "primary": f"{GOSDT} minus {CART}",
            "grid": list(GRID),
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        (out / "protocol.md").write_bytes((REPO_ROOT / config["protocol"]).read_bytes())
        (out / "configuration.yaml").write_bytes(Path(config_path).read_bytes())
        _checkpoint(out, manifest)
    start = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            _experiment(config, out, manifest, python, vendor, progress)
        changed = [p for p, sha in snapshot["inputs"].items() if digest(p) != sha]
        changed += [p for p, sha in snapshot["code"].items() if digest(REPO_ROOT / p) != sha]
        if changed:
            raise SchemeError(f"protected input drift: {changed}")
        manifest["protected_inputs_unchanged"] = True
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(
            status="resource_stopped" if isinstance(error, MemoryError) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        manifest["last_segment_seconds"] = time.monotonic() - start
        manifest.setdefault("segments", []).append(
            {
                "elapsed_seconds": manifest["last_segment_seconds"],
                "driver_peak_working_set_bytes": _peak_memory(),
                "status": manifest["status"],
            }
        )
        _checkpoint(out, manifest)
    return out


def _experiment(config, out, manifest, python, vendor, progress):
    runtime_probe(python, vendor, out / "runtime")
    items, original, target, labels = _load(config)
    folds = assign_folds(items, config["seed"])
    json_write(out / "folds.json", folds)
    saved = pd.read_csv(REPO_ROOT / config["baseline_evaluation"] / "verdicts.csv")
    manual = saved[saved.system == MANUAL].set_index("item_id").reindex(items.item_id)
    if manual.fold.isna().any() or any(int(manual.loc[i, "fold"]) != folds[i] for i in folds):
        raise SchemeError("historical baseline coverage/folds differ")
    groups = out / "groups"
    groups.mkdir(exist_ok=True)
    process = SolverProcess(python, vendor, out / "fits")
    prediction_rows, rule_rows = [], []
    trees = {}
    for fold in sorted(set(folds.values())):
        require_memory(config["minimum_free_bytes"])
        train = sorted(i for i in folds if folds[i] != fold)
        held = sorted(i for i in folds if folds[i] == fold)
        source = B1CrossFitter(target, labels, config["seed"], fit_data=original)
        restore_upstream(source, out, fold)
        inputs = source.fit(train)
        baseline = aggregate_combined(target, inputs.p, "hard")
        for i in held:
            expected = None if pd.isna(manual.loc[i, "verdict"]) else manual.loc[i, "verdict"]
            if baseline[i].verdict != expected:
                raise SchemeError(f"B1 baseline verdict drift at {i}")
            prediction_rows.append(
                {"system": MANUAL, "item_id": i, "fold": fold, "verdict": baseline[i].verdict}
            )
        for sid, scheme in sorted(target.schemes.items()):
            own_held = [i for i in held if items.set_index("item_id").loc[i, "gold_scheme"] == sid]
            if not own_held:
                continue
            progress(f"Fold {fold + 1}/5, {sid}: matched six-candidate selection", flush=True)
            path = groups / f"{fold}_{sid}.json"
            if path.exists():
                group = json.loads(path.read_text(encoding="utf-8"))
            else:
                require_memory(2 * 1024**3)
                group = group_fit(
                    source, train, own_held, scheme, fold, groups, process, config["seed"]
                )
                json_write(path, group)
                _checkpoint(out, manifest)
            if not group["complete"]:
                manifest["scientific_primary_complete"] = False
                raise SchemeError(
                    f"{sid} fold {fold}: {group['reason']}; no automatic budget/grid change"
                )
            for method, result in group["results"].items():
                for i, verdict, conditions in zip(
                    own_held, result["predictions"], result["paths"], strict=True
                ):
                    prediction_rows.append(
                        {
                            "system": method,
                            "item_id": i,
                            "fold": fold,
                            "verdict": verdict,
                            "status": result["model"]["status"],
                            "certified": result["model"]["certified"],
                            "path": json.dumps(conditions),
                            "native_replay_verified": result["model"]["evaluation_replay_verified"],
                        }
                    )
                trees[f"{fold}:{sid}:{method}"] = {
                    "predicates": group["predicates"],
                    "mask": group["constant_mask"],
                    **result["model"],
                }
            rule_rows.extend(export_rules(group, scheme, fold))
            pd.DataFrame(prediction_rows).to_csv(out / "verdicts_partial.csv", index=False)
            json_write(out / "trees_partial.json", trees)
            json_write(out / f"upstream_choices_fold{fold}.json", source.choices)
            with (out / f"upstream_fold{fold}.pkl").open("wb") as file:
                pickle.dump(source.models, file)
            # Coefficients stay cached; dense predictions/aggregations can be reconstructed.
            source.release_predictions()
            _checkpoint(out, manifest)
            gc.collect()
        del source, inputs, baseline
        gc.collect()
    fit_reports(groups, out)
    history = pd.read_csv(REPO_ROOT / config["historical_evaluation"] / "verdicts.csv")
    history = history[history.system == HISTORICAL]
    if len(history) != 601 or set(history.item_id) != set(items.item_id):
        raise SchemeError("historical B3 context does not cover the same 601 arguments")
    prediction_rows.extend(history.to_dict("records"))
    predictions = pd.DataFrame(prediction_rows)
    replay_counts = {}
    for method in (GOSDT, CART):
        own = predictions[predictions.system == method]
        if (
            len(own) != 601
            or own.item_id.duplicated().any()
            or set(own.item_id) != set(items.item_id)
            or not own.native_replay_verified.eq(True).all()
        ):
            raise SchemeError(f"incomplete native held-out replay: {method}")
        replay_counts[method] = len(own)
    predictions.to_csv(out / "verdicts.csv", index=False)
    json_write(out / "trees.json", trees)
    json_write(out / "rules.json", rule_rows)
    items["origin"] = items.source.str.lower().map(ORIGINS)
    items["reachable_gold"] = [
        r.gold_fallacy in target.schemes[r.gold_scheme].terminals for r in items.itertuples()
    ]
    items.loc[~items.reachable_gold].to_csv(out / "unreachable_gold.csv", index=False)
    del original, target, labels
    gc.collect()
    progress("Scoring the fixed OOF predictions with 10,000 paired draws", flush=True)
    metrics, differences, per_class = score_tables(
        items,
        predictions,
        config["seed"],
        config["resamples"],
        primary_system=GOSDT,
        reference_system=CART,
    )
    for name, frame in (
        ("metrics", metrics),
        ("differences", differences),
        ("per_class", per_class),
    ):
        frame.to_csv(out / f"{name}.csv", index=False)
    primary = (
        differences[(differences.subset == "all") & (differences.metric == "macro_f1")]
        .iloc[0]
        .to_dict()
    )
    manifest.update(
        scientific_primary_complete=True,
        primary_result=primary,
        export_replay_verified_items=replay_counts,
        practical_interest=primary["difference"] >= config["practical_delta"],
    )
    table = metrics[
        (metrics.subset == "all") & metrics.metric.isin(["accuracy", "macro_f1", "coverage"])
    ]
    summary = table.pivot(index="system", columns="metric", values="estimate").to_csv()
    (out / "summary.md").write_text(
        "# Matched GOSDT/CART diagnostic\n\n"
        + "```csv\n"
        + summary
        + "```\n\n"
        + f"Primary Macro-F1 difference: {primary['difference']:.4f} "
        f"[{primary['low']:.4f}, {primary['high']:.4f}].\n\n"
        "All 601 arguments retained; historical/manual references are secondary. "
        "Intervals condition on fixed OOF predictions. Exact tree replay is verified; "
        "semantic rule flags are not independent Walton adjudication.\n",
        encoding="utf-8",
    )
