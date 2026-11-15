"""Compare two fallacy references on frozen item verdicts."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..labels import INVALID, SchemeError, collapse, label_space
from ..paths import FALLACIES_FILE, REPO_ROOT, SCHEME_LABELS_FILE, SCHEMES_ROOT
from ..schemes import load_all, load_vocabulary
from .analysis_bootstrap import Bootstrap, score_population
from .analysis_inputs import Inputs, require
from .canonical import NO_VERDICT, _gold_terminal, _score, predicted_fallacy_class, score_fallacies
from .evaluate import Scores, difference, interval
from .reference_sensitivity_sources import (
    EVALS,
    FINE,
    GOLD_SCHEME,
    QWEN,
    System,
    inventory,
    relative,
)

HISTORICAL = "historical"
INTEGRATED = "integrated"
REFERENCE_COLUMNS = {HISTORICAL: "gold_fallacy", INTEGRATED: "integrated_verdict"}
APP = "dumitru_app"
ORIGINAL = "original_dataset"
AUDIT_POPULATION = "audit_405"
AUDIT_STATUSES = {"absent", "specialized", "changed", "composite", "unchanged"}
ORIGINAL_LABEL_AUDIT = REPO_ROOT / "data/original_label_audit.csv"
AUDIT_COLUMNS = ["item_id", "gold_scheme", "gold_fallacy", "original_label_id", "status"]
METRICS = ("accuracy", "macro_f1")
SEED = 20261019
RESAMPLES = 10_000
BOOTSTRAP_BATCH = 512


def check_ids(frame, name):
    duplicate_ids = frame.loc[frame.item_id.duplicated(keep=False), "item_id"].unique().tolist()
    require(
        not duplicate_ids and frame.item_id.ne("").all(),
        f"{name}: duplicate IDs {duplicate_ids[:12]} or empty item ID",
    )


def validate_references(items, integrated, vocab):
    check_ids(items, "items.csv")
    check_ids(integrated, "Dumitru_test_set.csv")
    require(set(items.in_test_set) <= {"True", "False"}, "Invalid in_test_set values")
    test = items.loc[
        items.in_test_set.eq("True"), ["item_id", "gold_scheme", "gold_fallacy", "eleni_id"]
    ]
    left, right = set(test.item_id), set(integrated.item_id)
    require(
        len(test) == 601 and len(integrated) == 601 and left == right,
        (
            f"Test IDs differ (items.csv={len(test)}, Dumitru_test_set.csv={len(integrated)}). "
            f"Only historical {sorted(left - right)[:12]}. "
            f"Only integrated {sorted(right - left)[:12]}."
        ),
    )
    frame = test.merge(integrated, on="item_id", suffixes=("", "_copied"), validate="one_to_one")
    for name in ("gold_fallacy", "gold_scheme"):
        mismatch = frame.loc[frame[name].ne(frame[f"{name}_copied"]), "item_id"].tolist()
        require(
            not mismatch,
            f"Dumitru_test_set.csv: {name} differs on {len(mismatch)} IDs {mismatch[:12]}",
        )
    counts = Counter(frame.verdict_origin)
    require(
        counts == {APP: 406, ORIGINAL: 195}, f"Unexpected verdict_origin counts: {dict(counts)}"
    )
    app = frame.verdict_origin.eq(APP)
    changed = frame.gold_fallacy.ne(frame.integrated_verdict)
    bad_original = frame.loc[
        ~app & (changed | frame.original_verdict.ne(frame.gold_fallacy)), "item_id"
    ].tolist()
    require(
        not bad_original, f"Original labels differ on {len(bad_original)} IDs {bad_original[:12]}"
    )
    bad_app = frame.loc[app & frame.dumitru_direct.ne(frame.integrated_verdict), "item_id"].tolist()
    require(not bad_app, f"Integrated and app fields differ on {len(bad_app)} IDs {bad_app[:12]}")
    require(
        int((app & changed).sum()) == 155,
        f"Expected 155 app changes, found {int((app & changed).sum())}",
    )
    for label in REFERENCE_COLUMNS.values():
        for item_id, value in zip(frame.item_id, frame[label], strict=True):
            require(
                _gold_terminal(item_id, value, vocab) == value,
                f"{label}: noncanonical terminal {value!r} at {item_id}",
            )
    columns = [
        "item_id",
        "gold_scheme",
        "gold_fallacy",
        "integrated_verdict",
        "verdict_origin",
        "eleni_id",
    ]
    return (
        frame[columns].assign(label_changed=changed).sort_values("item_id").reset_index(drop=True)
    )


def read_references(inputs, vocab):
    items = inputs.csv(
        REPO_ROOT / "data/items.csv",
        ["item_id", "in_test_set", "gold_scheme", "gold_fallacy", "eleni_id"],
        dtype=str,
    )
    integrated = inputs.csv(
        REPO_ROOT / "data/Dumitru_test_set.csv",
        [
            "item_id",
            "gold_scheme",
            "gold_fallacy",
            "integrated_verdict",
            "verdict_origin",
            "original_verdict",
            "dumitru_direct",
        ],
        dtype=str,
    )
    return validate_references(items, integrated, vocab)


def audit_population(items, audit):
    """Select the items whose original label is absent or differs from the historical gold."""
    check_ids(audit, "original-label audit")
    require(set(audit.item_id) == set(items.item_id), "Original-label audit test IDs differ")
    matched = items.merge(audit, on="item_id", suffixes=("", "_audit"), validate="one_to_one")
    require(
        matched.gold_scheme.eq(matched.gold_scheme_audit).all()
        and matched.gold_fallacy.eq(matched.gold_fallacy_audit).all(),
        "Original-label audit differs from the frozen test schemes or gold",
    )
    require(set(matched.status) == AUDIT_STATUSES, "Unexpected original-label audit statuses")
    unchanged = matched[matched.status.eq("unchanged")]
    require(
        unchanged.original_label_id.eq(unchanged.gold_fallacy).all(),
        "An unchanged original label differs from the historical gold",
    )
    require(
        unchanged.integrated_verdict.eq(unchanged.gold_fallacy).all(),
        "The integrated reference changes an unchanged original label",
    )
    audited = matched[matched.status.ne("unchanged")]
    require(
        audited.verdict_origin.eq(APP).all(),
        "An absent or changed original label has no app verdict",
    )
    selected = set(audited.item_id)
    require(len(selected) == 405, f"Expected 405 audited items, found {len(selected)}")
    return selected


def historical_reproduction(catalog, items, vocab):
    checks = []
    cache = {}
    for reference in catalog.history:
        id_ = reference.system_id
        if id_ not in cache:
            rows = (
                catalog.systems[id_]
                .rows[["item_id", "predicted"]]
                .replace({"predicted": {INVALID: ""}})
            )
            score = score_fallacies(rows, items, vocabulary=vocab)
            cache[id_] = score.summary() | {
                "coverage": 1 - sum(score.not_a_class.values()) / score.n_items,
            }
        actual = cache[id_]
        require(
            actual["n_items"] == reference.n_items,
            f"{id_}: historical denominator mismatch in {reference.path}",
        )
        for metric, saved in reference.scores.items():
            got = actual[metric]
            require(
                abs(got - saved) <= reference.tolerance,
                (
                    f"{id_}: historical {metric} mismatch in {reference.path}. "
                    f"Saved {saved}, got {got}"
                ),
            )
            checks.append(
                {
                    "system_id": id_,
                    "source_file": relative(reference.path),
                    "source_key": reference.key,
                    "n_items": reference.n_items,
                    "metric": metric,
                    "saved": saved,
                    "reproduced": got,
                    "absolute_error": abs(got - saved),
                    "tolerance": reference.tolerance,
                }
            )
    require(bool(checks), "No canonical historical metric was verified")
    return pd.DataFrame(checks)


def align_predictions(system: System, space, items, vocab, menus):
    prediction = system.rows.drop(columns="gold_scheme", errors="ignore")
    frame = items.merge(prediction, on="item_id", how="left", validate="one_to_one", indicator=True)
    frame["eligible"] = frame.pop("_merge").eq("both")
    frame["predicted"] = frame.predicted.fillna("")
    try:
        kinds = {
            label: (None, INVALID)
            if label == INVALID
            else predicted_fallacy_class(label, space, vocab)
            for label in frame.predicted.unique()
        }
    except SchemeError as error:
        raise SchemeError(f"{system.id} in {system.source}: {error}") from error
    resolved = [kinds[label] for label in frame.predicted]
    frame["prediction_class"] = pd.Series([label for label, _ in resolved], dtype=object)
    frame["prediction_kind"] = [kind for _, kind in resolved]
    frame.loc[~frame.eligible, "prediction_kind"] = "outside_original_population"
    for reference, column in REFERENCE_COLUMNS.items():
        frame[f"{reference}_class"] = frame[column].map(lambda value: collapse(value, space, vocab))
        frame[f"correct_{reference}"] = (
            frame.prediction_class.eq(frame[f"{reference}_class"])
            .astype("boolean")
            .where(frame.eligible)
        )
    historical = frame.correct_historical.fillna(False)
    integrated = frame.correct_integrated.fillna(False)
    frame["transition"] = np.select(
        [~frame.eligible, historical & integrated, historical, integrated],
        ["outside_original_population", "correct_both", "correct_to_wrong", "wrong_to_correct"],
        default="wrong_both",
    )
    if system.version in menus:
        lookup = frame.gold_scheme.map(menus[system.version])
        for key, column in {**REFERENCE_COLUMNS, "prediction": "prediction_class"}.items():
            frame[f"{key}_outside_diagram_menu"] = [
                value not in terminal_ids if value is not None else pd.NA
                for value, terminal_ids in zip(frame[column], lookup, strict=True)
            ]
    for field, value in (("system_id", system.id), ("label_space", space)):
        frame.insert(0 if field == "system_id" else 1, field, value)
    return frame


def score_pair(frame, space, vocab):
    eligible = frame[frame.eligible]
    n = len(eligible)
    predictions = eligible.prediction_class.tolist()
    kinds = eligible.prediction_kind.tolist()
    labels = label_space(space, vocab)
    terminal_count = int(eligible.prediction_class.notna().sum())
    counts = Counter(eligible.transition)
    result = {
        "n_population": len(frame),
        "n_scored": n,
        "n_saved_rows": n,
        "n_outside_original_population": len(frame) - n,
        "n_verdicts": terminal_count,
        "n_no_verdict_or_abstention": int(eligible.prediction_kind.eq(NO_VERDICT).sum()),
        "n_invalid_or_nonclass": int(
            eligible.prediction_class.isna().sum() - eligible.prediction_kind.eq(NO_VERDICT).sum()
        ),
        "coverage": terminal_count / n if n else None,
        "coverage_of_population": terminal_count / len(frame),
        "n_reference_label_changes_scored": int(eligible.label_changed.sum()),
        **{
            key: counts[key]
            for key in ("correct_both", "wrong_both", "correct_to_wrong", "wrong_to_correct")
        },
    }
    for name in (*REFERENCE_COLUMNS, "prediction"):
        flag = f"{name}_outside_diagram_menu"
        result[f"n_{name}_outside_diagram_menu"] = (
            int(eligible[flag].eq(True).sum()) if flag in eligible else None
        )
    scores = {}
    for reference in REFERENCE_COLUMNS:
        score = _score(eligible[f"{reference}_class"].tolist(), predictions, kinds, labels)
        scores[reference] = score
        result.update(
            {
                f"n_correct_{reference}": int(eligible[f"correct_{reference}"].sum()),
                f"n_classes_{reference}": score.n_classes_gold,
                **{
                    f"{metric}_{reference}": getattr(score, metric) if n else None
                    for metric in METRICS
                },
            }
        )
        require(
            score.n_items == n and sum(score.not_a_class.values()) == n - terminal_count,
            "Canonical scorer disagrees with item audit on coverage",
        )
        if n:
            require(
                np.isclose(score.accuracy, result[f"n_correct_{reference}"] / n),
                "Canonical scorer disagrees with item correctness",
            )
    for metric in METRICS:
        result[f"{metric}_delta_pp"] = (
            100 * (result[f"{metric}_integrated"] - result[f"{metric}_historical"]) if n else None
        )
    return result, scores


def class_results(scores):
    tables = [
        score.classes.set_index("class").add_suffix(f"_{reference}")
        for reference, score in scores.items()
    ]
    result = tables[0].join(tables[1], validate="one_to_one")
    for metric in ("precision", "recall", "f1"):
        result[f"{metric}_delta_pp"] = 100 * (
            result[f"{metric}_integrated"] - result[f"{metric}_historical"]
        )
    result["support_delta"] = result.support_integrated - result.support_historical
    return result.reset_index().to_dict("records")


class PairedBootstrap:
    def __init__(self, vocab):
        self.bootstrap = Bootstrap(SEED, RESAMPLES, batch_size=BOOTSTRAP_BATCH)
        # Injective representatives let the project's fine bootstrap scorer count
        # already collapsed classes with the same accuracy and class supports.
        self.representatives = {
            collapse(label, "collapsed_symmetric", vocab): label
            for label in label_space(FINE, vocab)
        }

    def score(self, frame, space):
        predictions = frame.prediction_class.tolist()
        golds = {r: frame[f"{r}_class"].tolist() for r in REFERENCE_COLUMNS}
        if space != FINE:
            predictions = [self.representatives[p] if p is not None else None for p in predictions]
            golds = {r: [self.representatives[g] for g in labels] for r, labels in golds.items()}
        result = {}
        eligible = np.ones(len(frame), dtype=bool)
        for reference, gold in golds.items():
            observed, resampled = score_population(gold, predictions, eligible, self.bootstrap)
            result[reference] = Scores(
                {m: observed[m] for m in METRICS}, {m: resampled[m] for m in METRICS}
            )
        return result


def interval_columns(result, draws):
    for metric in METRICS:
        for reference, score in draws.items():
            require(
                np.isclose(score.observed[metric], result[f"{metric}_{reference}"], atol=1e-12),
                f"Paired scorer and canonical scorer disagree on {metric} / {reference}",
            )
        contrast = difference(draws[INTEGRATED], draws[HISTORICAL], metric)
        for bound in ("low", "high"):
            result[f"{metric}_delta_{bound}_pp"] = 100 * contrast[bound]


def rankings(main):
    full = main[main.cohort.eq("test") & main.role.ne("oracle")]
    gold = full[full.condition.eq(GOLD_SCHEME)]
    groups = [("gold_scheme_all_versions", gold)]
    groups += [(f"gold_scheme_{v}", part) for v, part in gold.groupby("scheme_version")]
    groups += list(full[~full.condition.eq(GOLD_SCHEME)].groupby("condition"))
    columns = [
        "system_id",
        "population",
        "label_space",
        "scheme_version",
        "n_population",
        "n_scored",
        "n_verdicts",
        "coverage",
    ]
    output = []
    for name, rows in groups:
        for _, part in rows.groupby(["population", "label_space"]):
            require(part.n_scored.eq(part.n_population).all(), f"Rank cohort {name} is incomplete")
            frame = part[columns].copy().assign(ranking_group=name, n_systems=len(part))
            for metric in METRICS:
                for reference in REFERENCE_COLUMNS:
                    field = f"{metric}_{reference}"
                    frame[field] = part[field]
                    frame[f"{field}_rank"] = (
                        part[field].round(12).rank(ascending=False, method="min").astype(int)
                    )
                frame[f"{metric}_rank_gain"] = (
                    frame[f"{metric}_historical_rank"] - frame[f"{metric}_integrated_rank"]
                )
            output.append(frame)
    return pd.concat(output, ignore_index=True)


def system_pairs():
    a0 = f"A0:{QWEN}"
    direct = f"zeroshot:{QWEN}"
    rerun = f"zeroshot.rerun:{QWEN}"
    pairs = [
        ("B1@1.1", f"{a0}@1.1"),
        ("B1@1.1", f"{direct}@1.1"),
        ("B2@1.1", f"{direct}@1.1"),
        ("B3@1.1", "B1.hard@1.1"),
        ("B3@1.1", f"{direct}@1.1"),
        ("B4@2.0", "A2.pruned@2.0"),
        ("B3@2.0", "B1.hard@2.0"),
        ("GOSDT.original@2.0", "CART.binary.original@2.0"),
        ("B1.annotation_exclusion@1.1", "B1@1.1"),
        (f"{a0}@2.0", f"{a0}@1.1"),
        ("A2@2.0", "A2@1.1"),
        (f"DQ:{QWEN}@1.1", f"{rerun}@1.1"),
        (f"DQ:{QWEN}@2.0", f"{direct}@2.0"),
        (f"{rerun}@1.1", f"{direct}@1.1"),
    ]
    for version in ("1.1", "2.0"):
        pairs += [
            (f"A5@{version}", f"{direct}@{version}"),
            (f"A5.answers_only@{version}", f"A5@{version}"),
        ]
    rank_follow_ups = [
        ("A5@1.1", "B2.dq@2.0"),
        ("A5@1.1", "A5.answers_only@2.0"),
        ("A5@1.1", "A5@2.0"),
    ]
    return [(a, b, "matched_contrast") for a, b in pairs] + [
        (a, b, "post_hoc_rank_contrast") for a, b in rank_follow_ups
    ]


def paired_comparisons(frames, scores, bootstrap, populations):
    output = []
    for a, b, purpose in system_pairs():
        if (a, FINE) not in frames or (b, FINE) not in frames:
            continue
        for population, ids in populations.items():
            left, right = (
                frame[frame.item_id.isin(ids)] for frame in (frames[a, FINE], frames[b, FINE])
            )
            common = set(left.loc[left.eligible, "item_id"]) & set(
                right.loc[right.eligible, "item_id"]
            )
            if not common:
                continue
            matched_ids = sorted(common)
            res = []
            for key, part in ((a, left), (b, right)):
                own = part[part.item_id.isin(common)]
                require(own.item_id.tolist() == matched_ids, "Paired draws must use the same IDs")
                if len(own) == int(part.eligible.sum()):
                    res.append(scores[key, FINE, population])
                else:
                    res.append(bootstrap.score(own, FINE))
            for metric in METRICS:
                row = {
                    "population": population,
                    "system_id": a,
                    "comparator_id": b,
                    "purpose": purpose,
                    "metric": metric,
                    "n_common_items": len(common),
                }
                deltas = {}
                for reference in REFERENCE_COLUMNS:
                    s, r = (result[reference] for result in res)
                    contrast = difference(s, r, metric)
                    deltas[reference] = s.resampled[metric] - r.resampled[metric]
                    row.update({f"{reference}_{k}_pp": 100 * v for k, v in contrast.items()})
                shift = deltas[INTEGRATED] - deltas[HISTORICAL]
                low, high = interval(shift)
                row.update(
                    {
                        "change_in_difference_pp": row["integrated_difference_pp"]
                        - row["historical_difference_pp"],
                        "change_low_pp": 100 * low,
                        "change_high_pp": 100 * high,
                    }
                )
                output.append(row)
    return pd.DataFrame(output)


def append_csv(frame, path):
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")


def evaluate(items, systems, vocab, menus, populations, staging):
    frames, draws = {}, {}
    main, schemes = [], []
    streamed = {"per_class.csv": 0, "verdicts.csv": 0}
    verdict_columns = []
    saved_columns = {column for system in systems.values() for column in system.rows}
    if any(system.version in menus for system in systems.values()):
        saved_columns.update(
            f"{key}_outside_diagram_menu" for key in (*REFERENCE_COLUMNS, "prediction")
        )
    bootstrap = PairedBootstrap(vocab)
    paired_systems = {system for a, b, _ in system_pairs() for system in (a, b)}
    for system in systems.values():
        for space in system.spaces:
            classes = []
            frame = align_predictions(system, space, items, vocab, menus)
            if not verdict_columns:
                verdict_columns = [c for c in frame if c != "eleni_id"]
                verdict_columns += sorted(saved_columns - set(frame.columns))
            append_csv(frame.reindex(columns=verdict_columns), staging / "verdicts.csv")
            streamed["verdicts.csv"] += len(frame)
            keep_paired = space == FINE and system.id in paired_systems
            if keep_paired:
                frames[system.id, space] = frame
            for population, ids in populations.items():
                part = frame[frame.item_id.isin(ids)]
                meta = {
                    "system_id": system.id,
                    "population": population,
                    "label_space": space,
                    "scheme_version": system.version,
                    "condition": system.condition,
                    "cohort": system.cohort,
                    "role": system.role,
                }
                result, scores = score_pair(part, space, vocab)
                if result["n_scored"]:
                    paired = bootstrap.score(part[part.eligible], space)
                    if keep_paired:
                        draws[system.id, space, population] = paired
                    interval_columns(result, paired)
                main.append(meta | result)
                classes.extend(meta | {"gold_scheme": "all"} | row for row in class_results(scores))
                for scheme, block in part.groupby("gold_scheme", sort=True):
                    result, scores = score_pair(block, space, vocab)
                    schemes.append(
                        meta | {"gold_scheme": scheme, "small_scheme": len(block) < 30} | result
                    )
                    classes.extend(
                        meta | {"gold_scheme": scheme} | row for row in class_results(scores)
                    )
            append_csv(pd.DataFrame(classes), staging / "per_class.csv")
            streamed["per_class.csv"] += len(classes)
    main = pd.DataFrame(main)
    pairs = paired_comparisons(frames, draws, bootstrap, populations)
    return {
        "main_table.csv": main,
        "per_scheme.csv": pd.DataFrame(schemes),
        "rankings.csv": rankings(main),
        "paired_comparisons.csv": pairs,
    }, streamed


def run(out: Path, notes: Path | None, original_label_audit: Path = ORIGINAL_LABEL_AUDIT):
    out = out.resolve()
    require(
        out.is_relative_to(EVALS.resolve()) and out != EVALS.resolve(),
        "Choose a new output folder under evaluations/",
    )
    require(not out.exists() and out.parent.is_dir(), f"Output must be a new directory: {out}")
    inputs = Inputs()
    for source in (FALLACIES_FILE, SCHEME_LABELS_FILE):
        inputs.add(source)
    for filename in (
        "reference_sensitivity.py",
        "reference_sensitivity_sources.py",
        "analysis_bootstrap.py",
        "canonical.py",
        "evaluate.py",
        "metrics.py",
        "folds.py",
        "prior.py",
        "compat.py",
    ):
        inputs.add(Path(__file__).with_name(filename))
    vocab = load_vocabulary(check_schemes=False)
    items = read_references(inputs, vocab)
    audit = inputs.csv(original_label_audit, AUDIT_COLUMNS, dtype=str)
    selected = audit_population(items, audit)
    populations = {"test_601": set(items.item_id), AUDIT_POPULATION: selected}
    items["in_audit_405"] = items.item_id.isin(selected)
    audit_record = {
        "original_label_audit": {
            "source": relative(original_label_audit),
            "criterion": "status differs from unchanged",
            "status_counts": dict(Counter(audit.status)),
            "n_labels_different_from_historical": int(
                items.loc[items.in_audit_405, "label_changed"].sum()
            ),
        }
    }
    menus = {}
    for version, folder in (("1.1", "v1"), ("2.0", "v2")):
        path = SCHEMES_ROOT / folder
        for file in path.glob("*.yaml"):
            inputs.add(file)
        menus[version] = {name: s.terminals for name, s in load_all(path).items()}
        require(set(items.gold_scheme) <= menus[version].keys(), f"{folder}: unknown gold scheme")
    catalog = inventory(items, inputs, notes)
    reproduction = historical_reproduction(catalog, items, vocab)
    print(f"Verified 601 reference pairs and {len(reproduction)} canonical historical metrics.")
    tables = {
        "systems.csv": pd.DataFrame(
            [
                {
                    "system_id": s.id,
                    "model": s.model,
                    "method": s.method,
                    "scheme_version": s.version,
                    "condition": s.condition,
                    "role": s.role,
                    "supervision": s.supervision,
                    "cohort": s.cohort,
                    "label_spaces": "|".join(s.spaces),
                    "n_saved_rows": len(s.rows),
                    "source_file": relative(s.source),
                    "source_key": s.source_key,
                }
                for s in catalog.systems.values()
            ]
        ),
        "source_map.csv": pd.DataFrame(catalog.sources),
        "unavailable.csv": pd.DataFrame(catalog.missing),
        "historical_reproduction.csv": reproduction,
        "compat_validation.csv": pd.DataFrame(catalog.compat),
        "reference_transitions.csv": (
            items.groupby(["verdict_origin", "gold_scheme", "gold_fallacy", "integrated_verdict"])
            .size()
            .rename("n_items")
            .reset_index()
        ),
    }
    scratch = (REPO_ROOT / "scratch").resolve()
    with TemporaryDirectory(prefix="reference_sensitivity_", dir=scratch) as directory:
        staging = Path(directory).resolve()
        require(staging.parent == scratch, "Unexpected staging directory")
        with threadpool_limits(limits=2, user_api="blas"):
            scored, counts = evaluate(items, catalog.systems, vocab, menus, populations, staging)
        for name, table in (tables | scored).items():
            table.to_csv(staging / name, index=False, lineterminator="\n")
            counts[name] = len(table)
        inputs.verify()
        out.mkdir()
        for name in counts:
            copyfile(staging / name, out / name)
    outputs = Inputs()
    for name in counts:
        outputs.add(out / name)
    manifest = {
        "name": out.name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "analysis": "offline reference sensitivity of fixed saved verdicts",
        "historical_reference": "data/items.csv:gold_fallacy on in_test_set",
        "integrated_reference": "data/Dumitru_test_set.csv:integrated_verdict",
        "populations": {name: len(ids) for name, ids in populations.items()},
        **audit_record,
        "reference_checks": {
            "same_unique_item_ids": len(items),
            "copied_historical_labels_and_schemes_match": True,
            "verdict_origin_counts": dict(Counter(items.verdict_origin)),
            "original_labels_match_historical": True,
            "changed_app_labels": int(items.label_changed.sum()),
        },
        "bootstrap": {
            "seed": SEED,
            "resamples": RESAMPLES,
            "batch_size": BOOTSTRAP_BATCH,
            "generator": 1,
            "unit": "item",
            "level": 0.95,
            "method": "paired percentile",
            "draw_population": "each fixed eligible population, sorted by item_id",
        },
        "n_system_conditions": len(catalog.systems),
        "canonical_historical_metric_checks": len(reproduction),
        "compat_task_checks": len(catalog.compat),
        "excluded_prior_ids_by_run": catalog.prior_mapping,
        "inputs_verified_unchanged": True,
        "inputs": [{"path": relative(p), "sha256": h} for p, h in sorted(inputs.files.items())],
        "outputs": {
            p.name: {"sha256": h, "rows": counts[p.name]} for p, h in sorted(outputs.files.items())
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(catalog.systems)} system conditions to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--research-notes", type=Path, help="Optional directory of saved human-CQ diagnostics."
    )
    parser.add_argument(
        "--original-label-audit",
        type=Path,
        default=ORIGINAL_LABEL_AUDIT,
        help="Original-label audit that selects the restricted population.",
    )
    args = parser.parse_args()
    run(args.out, args.research_notes, args.original_label_audit)


if __name__ == "__main__":
    main()
