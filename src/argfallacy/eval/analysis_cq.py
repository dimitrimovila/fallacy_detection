"""CQ populations, individual questions and paired item-clustered differences."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from ..aggregate.combiners import EPS
from ..aggregate.soft import answer_masses, source_of
from ..parse import INVALID
from ..schemes import load_all
from .analysis_bootstrap import contribution_ratios
from .analysis_inputs import probability, read_run, require, unique
from .analysis_selective import origin_masks
from .analysis_tables import metrics_record, table_frame
from .b1 import cq_labels
from .cq import CBD, human_answers

KEYS = ["item_id", "scheme", "cq_id"]
B1_POPULATION = "B1 binary; gold_scheme_rows with positive scheme judgement"
P7_POPULATION = "P7 gold-scheme sheet; any scheme judgement and all human answer categories"
NODE_POPULATION = "observed S-soft path occurrences; unvisited CQs excluded"


def cq_views(frame, items):
    for origin, mask in origin_masks(items):
        part = frame[frame.item_id.isin(items.loc[mask, "item_id"])]
        yield origin, "all", "all", part
        for scheme, group in part.groupby("scheme", sort=True):
            yield origin, scheme, "all", group
            for cq, rows in group.groupby("cq_id", sort=True):
                yield origin, scheme, cq, rows


def _scores(rows, items, bootstrap, difference=False):
    positions = pd.Index(items.item_id).get_indexer(rows.item_id)
    require((positions >= 0).all(), "unknown CQ item")
    target = rows.target.to_numpy(float)
    p = rows.p_first.to_numpy(float)
    finite = np.isfinite(p)
    require(
        not np.isinf(p).any() and ((p[finite] >= 0) & (p[finite] <= 1)).all(),
        "invalid CQ evaluation probabilities",
    )
    present = np.isfinite(target)
    probability_present = present & np.isfinite(p)
    if "brier_valid" in rows:
        probability_present &= rows.brier_valid.to_numpy(bool)
    squared = np.where(probability_present, (p - target) ** 2, 0)
    hard_present = rows.human.notna().to_numpy() & rows.human.ne("").to_numpy()
    correct = (rows.answer.to_numpy() == rows.human.to_numpy()) & hard_present
    if difference:
        q = rows.p_first_reference.to_numpy(float)
        probability_present &= np.isfinite(q)
        if "brier_valid_reference" in rows:
            probability_present &= rows.brier_valid_reference.to_numpy(bool)
        squared = np.where(probability_present, (p - target) ** 2 - (q - target) ** 2, 0)
        correct = correct.astype(float) - (
            (rows.answer_reference.to_numpy() == rows.human.to_numpy()) & hard_present
        )
    n_annotated = rows.loc[hard_present, "item_id"].nunique()
    result = {
        "n_items": rows.item_id.nunique(),
        "n_cq_pairs": len(rows[KEYS].drop_duplicates()),
        "n_observations": len(rows),
        "annotated_items": n_annotated,
        "annotated_cq_pairs": len(rows.loc[hard_present, KEYS].drop_duplicates()),
        "annotated_observations": int(hard_present.sum()),
        "score_unit": "path occurrence" if "path" in rows else "CQ pair",
        "brier_n": int(probability_present.sum()),
        "brier_items": rows.loc[probability_present, "item_id"].nunique(),
        "hard_agreement_withheld": n_annotated < 20,
        "distribution": json.dumps(
            rows.answer.fillna("missing").replace("", "missing").value_counts().to_dict(),
            sort_keys=True,
        ),
        "human_distribution": json.dumps(
            rows.loc[hard_present, "human"].value_counts().to_dict(), sort_keys=True
        ),
    }
    for name, numerator, denominator in [
        ("agreement", correct, hard_present),
        ("brier", squared, probability_present),
    ]:
        if name == "agreement" and n_annotated < 20:
            result.update(
                {
                    f"{name}_difference" if difference else name: np.nan,
                    f"{name}_low": np.nan,
                    f"{name}_high": np.nan,
                    f"{name}_replicates": 0,
                    f"{name}_omitted": bootstrap.resamples,
                }
            )
            continue
        estimate, spread = contribution_ratios(
            positions,
            np.asarray(numerator, float),
            denominator.astype(float),
            len(items),
            bootstrap,
        )
        record = metrics_record({name: estimate}, {name: spread})
        if difference:
            record[f"{name}_difference"] = record.pop(name)
        result.update(record)
    return result


def cq_scores(predictions, labels, items, bootstrap, population=B1_POPULATION):
    """Pure frame adapter used by both saved inputs and synthetic checks."""
    keys = [*KEYS, *(["path"] if "path" in predictions else [])]
    unique(predictions, keys, "CQ system")
    unique(labels, KEYS, "CQ labels")
    rows = predictions.merge(
        labels,
        on=KEYS,
        how="inner" if population == B1_POPULATION else "left",
        validate="many_to_one",
    )
    output = []
    for origin, scheme, cq, group in cq_views(rows, items):
        if group.empty:
            continue
        output.append(
            {
                "origin": origin,
                "scheme": scheme,
                "cq_id": cq,
                "population": population,
                "level": 0.95,
                "n_path_occurrences": len(group) if "path" in group else 0,
                **_scores(group, items, bootstrap),
                "status": "computed" if group.human.notna().any() else "pending human targets",
            }
        )
    return output


def cq_differences(left, right, labels, items, bootstrap):
    """Every metric uses common compatible item/CQ pairs, and shared item draws."""
    unique(left, KEYS, "left CQ system")
    unique(right, KEYS, "right CQ system")
    rows = left.merge(right, on=KEYS, suffixes=("", "_reference"), validate="one_to_one")
    rows = rows.merge(labels, on=KEYS, validate="many_to_one")
    output = []
    for origin, scheme, cq, group in cq_views(rows, items):
        if group.empty:
            continue
        output.append(
            {
                "origin": origin,
                "scheme": scheme,
                "cq_id": cq,
                "population": B1_POPULATION,
                "level": 0.95,
                "kind": "exploratory",
                "status": "computed",
                **_scores(group, items, bootstrap, True),
            }
        )
    return output


def entry_predictions(summary, items, schemes, model, settings, observed_nodes=False):
    """B1's conditional split and hard fallback; P7 signals stay unconditional."""
    frame = summary[
        (summary.model == model)
        & (summary.scheme_condition == "gold")
        & summary.item_id.isin(items.item_id)
    ].copy()
    frame = frame[frame.stage == ("ssoft" if observed_nodes else "stage2")]
    keys = [*KEYS, *(["path"] if observed_nodes else [])]
    unique(frame, keys, f"{model}/CQ")
    if not observed_nodes:
        expected = pd.DataFrame(
            [
                {"item_id": r.item_id, "scheme": r.gold_scheme, "cq_id": cq}
                for r in items.itertuples()
                for cq in schemes[r.gold_scheme].cqs
            ]
        )
        frame = expected.merge(frame, on=KEYS, how="left", validate="one_to_one")
    records = []
    for row in frame.to_dict("records"):
        scheme, cq = schemes[row["scheme"]], row["cq_id"]
        require(cq in scheme.cqs, f"unknown CQ: {cq}")
        for key, value in list(row.items()):
            if key in ("p_verbal", "p_logprob", "p_sample") or key.startswith(
                ("p_logprob_", "p_sample_")
            ):
                row[key] = probability(value, f"{model}/{cq}/{key}")
        row["answer"] = (
            None if pd.isna(row.get("answer")) or row.get("answer") == "" else row["answer"]
        )
        row["raw_answer"] = row["answer"]
        arcs = scheme.answer_space(cq)
        masses = answer_masses(scheme, cq, row, source_of(settings))
        inside = sum((masses or {}).get(a, 0.0) for a in arcs)
        row["p_first"] = masses.get(arcs[0], 0.0) / inside if inside else 0.5
        if masses is None:
            row["answer"] = None
        row["brier_valid"] = True
        records.append(row)
    return (
        pd.DataFrame(records)
        if records
        else frame.assign(p_first=np.nan, brier_valid=True, raw_answer=None)
    )


def p7_predictions(frame, schemes, source):
    out = frame.copy()
    out["answer"] = out.raw_answer
    out["p_first"] = [
        r.get(
            source
            if source == "p_verbal"
            else f"{source}_{schemes[r['scheme']].answer_space(r['cq_id'])[0]}",
            np.nan,
        )
        for r in out.to_dict("records")
    ]
    out["brier_valid"] = out.answer.notna() & out.answer.ne(INVALID) & out.answer.ne("")
    return out


def combined_predictions(frame, summary, reference, items, schemes, assignment):
    unique(frame, KEYS, "saved combined CQ rows")
    require(set(frame.item_id) <= set(items.item_id), "unknown combined CQ items")
    gold_scheme = items.set_index("item_id").gold_scheme
    lookup = (
        summary[
            (summary.model == reference)
            & (summary.stage == "stage2")
            & (summary.scheme_condition == "gold")
        ]
        .set_index(KEYS)
        .answer.to_dict()
    )
    out = frame.copy()
    answers = []
    for row in out.itertuples():
        require(row.scheme == gold_scheme[row.item_id], "combined CQ gold scheme mismatch")
        require(row.cq_id in schemes[row.scheme].cqs, "combined CQ version mismatch")
        p, outside = (
            probability(row.p_first, "combined p_first"),
            probability(row.outside, "combined outside"),
        )
        require(np.isfinite(p + outside), "missing combined CQ probability")
        try:
            fold = float(row.fold) if row.fold != "" else np.nan
        except (ValueError, TypeError):
            require(False, "invalid combined CQ fold")
        require(
            row.fold == "" or (np.isfinite(fold) and fold == assignment[row.item_id]),
            "combined CQ fold mismatch",
        )
        require(row.supervision == "none" or row.fold != "", "missing supervised CQ fold")
        arcs = schemes[row.scheme].answer_space(row.cq_id)
        answer = arcs[0] if p >= 0.5 else arcs[1]
        if abs(p - 0.5) <= EPS and lookup.get((row.item_id, row.scheme, row.cq_id)) == arcs[1]:
            answer = arcs[1]
        answers.append(CBD if outside > 0.5 else answer)
    out["answer"], out["brier_valid"] = answers, True
    return out


def dq_cq_tables(config, evaluation, items, registry, binary, broad, empty, bootstrap, inputs):
    levels, differences, statuses = [], [], {}
    if config.dq is None or config.dq != evaluation.path:
        return levels, differences, statuses
    for name, run_id in evaluation.manifest.get("dq", {}).get("systems", {}).items():
        if not name.startswith("DQ:"):
            continue
        if run_id not in config.parsed_runs:
            statuses[name] = "pending: explicit parsed D+Q source unavailable"
            continue
        _, version, summary = read_run(config.parsed_runs[run_id], run_id, items, registry, inputs)
        entry = name.split(":", 1)[1].removesuffix("@2.0")
        schemes = load_all(registry[version])
        system = evaluation.systems[name]
        own_items = items.loc[system.eligible]
        own = summary[
            (summary.stage == "dq")
            & (summary.model == entry)
            & (summary.scheme_condition == "gold")
            & (summary.cq_id != "")
            & summary.item_id.isin(own_items.item_id)
        ]
        expected = pd.DataFrame(
            [
                dict(item_id=r.item_id, scheme=r.gold_scheme, cq_id=cq)
                for r in own_items.itertuples()
                for cq in schemes[r.gold_scheme].cqs
            ]
        )
        require(
            set(own[KEYS].itertuples(index=False, name=None))
            <= set(expected[KEYS].itertuples(index=False, name=None)),
            "unknown D+Q CQ keys/version",
        )
        frame = expected.merge(own, on=KEYS, how="left", validate="one_to_one")
        frame["answer"] = frame.answer.replace("", None)
        frame["p_first"] = [
            probability(
                r.get("p_logprob_" + schemes[r["scheme"]].answer_space(r["cq_id"])[0], ""),
                "D+Q raw CQ probability",
            )
            for r in frame.to_dict("records")
        ]
        frame["brier_valid"] = frame.answer.notna() & frame.answer.ne(INVALID)
        meta = dict(
            evaluation=evaluation.name,
            round=evaluation.round,
            run=run_id,
            system=name,
            scheme_version=version,
            source_table="summary.csv:dq CQ rows",
            source_population="declared D+Q eligible items; compatible question version",
            probability="p_logprob; outside-arc mass retained; no imputation",
            supervision="none",
        )
        labels = binary if version == "1.1" else empty
        levels.extend(
            {**meta, **r}
            for r in cq_scores(
                frame,
                labels,
                items,
                bootstrap,
                B1_POPULATION
                if version == "1.1"
                else "D+Q 2.0 CQ vectors; compatible human targets pending",
            )
        )
        levels.extend(
            {**meta, **r}
            for r in cq_scores(
                frame, broad if version == "1.1" else empty, items, bootstrap, P7_POPULATION
            )
        )
        old_run = evaluation.manifest.get("runs", {}).get("answers")
        if version == "1.1" and old_run in config.parsed_runs:
            old_manifest, old_version, old = read_run(
                config.parsed_runs[old_run], old_run, items, registry, inputs
            )
            require(old_version == version, "D+Q/reference CQ version mismatch")
            baseline = p7_predictions(
                entry_predictions(old, items, schemes, entry, old_manifest["models"][entry]),
                schemes,
                "p_logprob",
            )
            differences.extend(
                {**meta, "reference": f"entry:{entry}", **r}
                for r in cq_differences(frame, baseline, labels, items, bootstrap)
            )
        statuses[name] = dict(
            status="computed" if version == "1.1" else "pending human targets",
            observed_cq_pairs=len(own),
            planned_cq_pairs=len(expected),
            version=version,
        )
    return levels, differences, statuses


def cq_tables(config, evaluations, items, registry, annotations, bootstrap, inputs):
    original = load_all(registry["1.1"])
    empty = pd.DataFrame(columns=[*KEYS, "human", "target"])
    if annotations is None:
        binary, broad = empty.copy(), empty.copy()
    else:
        binary = cq_labels(annotations, items, original).rename(columns={"answer": "human"})
        broad = human_answers(annotations, items, original)
        broad["target"] = [
            float(r.human == original[r.scheme].answer_space(r.cq_id)[0])
            if r.human in original[r.scheme].answer_space(r.cq_id)
            else np.nan
            for r in broad.itertuples()
        ]
    levels, differences, statuses = [], [], {}
    for evaluation in evaluations:
        combined_path = evaluation.path / "cq_combined.csv"
        combined = (
            inputs.csv(
                combined_path,
                ["system", "schemes", "supervision", *KEYS, "fold", "p_first", "outside"],
            )
            if combined_path.is_file()
            else None
        )
        systems = {}
        if combined is not None:
            unique(combined, ["system", "schemes", *KEYS], "saved combined CQ rows")
            require(
                set(combined.schemes.astype(str)) <= set(registry), "unknown combined CQ version"
            )
            for (name, version), group in combined.groupby(["system", "schemes"]):
                version = str(version)
                system = evaluation.systems.get(name)
                require(
                    system is not None and system.version == version,
                    f"{name}: combined CQ system/version mismatch",
                )
                schemes = load_all(registry[version])
                expected = {
                    (r.item_id, r.gold_scheme, cq)
                    for r in items.loc[system.eligible].itertuples()
                    for cq in schemes[r.gold_scheme].cqs
                }
                observed = set(group[KEYS].itertuples(index=False, name=None))
                require(observed == expected, f"{name}: combined CQ declared population mismatch")
                require(group.supervision.nunique() == 1, f"{name}: inconsistent CQ supervision")
        for role in ("answers", "answers_schemes_2", "ssoft", "ssoft_nohistory"):
            configured = evaluation.manifest.get("runs", {}).get(role) or []
            run_ids = [configured] if isinstance(configured, str) else configured
            for run_id in run_ids:
                if run_id not in config.parsed_runs:
                    statuses[f"{evaluation.name}/{run_id}"] = "pending: parsed run not configured"
                    continue
                path = config.parsed_runs[run_id]
                manifest, version, summary = read_run(path, run_id, items, registry, inputs)
                schemes = load_all(registry[version])
                nodes = role.startswith("ssoft")
                if nodes:
                    require(version == "2.0", "S-soft CQ version must be 2.0")
                    require(
                        manifest.get("config", {}).get("stages") == ["ssoft"], "not an S-soft run"
                    )
                    require(
                        bool(manifest.get("config", {}).get("history", True)) == (role == "ssoft"),
                        "S-soft history mismatch",
                    )
                    summary = inputs.csv(
                        path / "ssoft.csv",
                        [*KEYS, "model", "scheme_condition", "stage", "path", "answer"],
                    )
                    require(set(summary.stage) <= {"ssoft"}, "unsupported S-soft stage")
                    require(
                        set(summary.model) <= set(manifest.get("models", {})),
                        "unknown S-soft model",
                    )
                    require(
                        set(summary.item_id) <= set(items.attrs.get("all_item_ids", items.item_id)),
                        "unknown S-soft items",
                    )
                    own = summary[
                        summary.item_id.isin(items.item_id) & summary.scheme_condition.eq("gold")
                    ]
                    gold = items.set_index("item_id").gold_scheme
                    require(
                        all(gold[i] == s for i, s in zip(own.item_id, own.scheme, strict=True)),
                        "S-soft gold scheme mismatch",
                    )
                labels = binary if version == "1.1" else empty
                broad_labels = broad if version == "1.1" else empty
                entries = manifest.get("models", {})
                for entry, settings in entries.items():
                    name = f"{role}:{entry}" if nodes else f"entry:{entry}"
                    frame = entry_predictions(summary, items, schemes, entry, settings, nodes)
                    meta = {
                        "evaluation": evaluation.name,
                        "round": evaluation.round,
                        "run": run_id,
                        "system": name,
                        "scheme_version": version,
                        "supervision": "none",
                    }
                    if not nodes:
                        levels.extend(
                            {**meta, "probability": "p_first", **r}
                            for r in cq_scores(frame, labels, items, bootstrap)
                        )
                        if version == "1.1":
                            systems[name] = ("entry", entry)
                    for source in ("p_logprob", "p_verbal", "p_sample"):
                        if source == "p_sample" and int(settings.get("samples", 1)) <= 1:
                            continue
                        p7 = p7_predictions(frame, schemes, source)
                        levels.extend(
                            {**meta, "probability": source, **r}
                            for r in cq_scores(
                                p7,
                                broad_labels,
                                items,
                                bootstrap,
                                NODE_POPULATION if nodes else P7_POPULATION,
                            )
                        )
                # Compare compatible 1.1 systems only; 2.0 has no frozen human CQ targets.
                if version != "1.1" and not nodes and combined is not None:
                    own_combined = combined[combined.schemes.astype(str) == version]
                    for name, group in own_combined.groupby("system", sort=True):
                        frame = combined_predictions(
                            group,
                            summary,
                            evaluation.manifest["reference_model"],
                            items,
                            schemes,
                            evaluation.manifest["folds"]["assignment"],
                        )
                        levels.extend(
                            {
                                "evaluation": evaluation.name,
                                "round": evaluation.round,
                                "run": run_id,
                                "system": name,
                                "scheme_version": version,
                                "probability": "p_first",
                                **r,
                            }
                            for r in cq_scores(
                                frame,
                                empty,
                                items,
                                bootstrap,
                                "B1 2.0 full vectors; compatible human targets pending",
                            )
                        )
                if version == "1.1" and not nodes:
                    own_combined = (
                        combined[combined.schemes.astype(str) == version]
                        if combined is not None
                        else None
                    )

                    def prediction(
                        name,
                        systems=systems,
                        summary=summary,
                        schemes=schemes,
                        entries=entries,
                        own_combined=own_combined,
                        evaluation=evaluation,
                    ):
                        kind, value = systems[name]
                        if kind == "entry":
                            return entry_predictions(summary, items, schemes, value, entries[value])
                        return combined_predictions(
                            own_combined[own_combined.system == name],
                            summary,
                            evaluation.manifest["reference_model"],
                            items,
                            schemes,
                            evaluation.manifest["folds"]["assignment"],
                        )

                    if own_combined is not None:
                        for name, group in own_combined.groupby("system", sort=True):
                            systems[name] = ("combined", str(group.supervision.iloc[0]))
                            frame = prediction(name)
                            levels.extend(
                                {
                                    "evaluation": evaluation.name,
                                    "round": evaluation.round,
                                    "run": run_id,
                                    "system": name,
                                    "scheme_version": version,
                                    "supervision": group.supervision.iloc[0],
                                    "probability": "p_first",
                                    **r,
                                }
                                for r in cq_scores(frame, labels, items, bootstrap)
                            )
                    reference = f"entry:{evaluation.manifest['reference_model']}"
                    pairs = {
                        (name, reference)
                        for name in systems
                        if name != reference and reference in systems
                    }
                    pairs.update(
                        (a, b)
                        for a, b in combinations(sorted(systems), 2)
                        if systems[a][0] == systems[b][0] == "combined"
                        and systems[a][1] != systems[b][1]
                    )
                    for name, reference in sorted(pairs):
                        differences.extend(
                            {
                                "evaluation": evaluation.name,
                                "round": evaluation.round,
                                "system": name,
                                "reference": reference,
                                "scheme_version": version,
                                "probability": "p_first",
                                **r,
                            }
                            for r in cq_differences(
                                prediction(name), prediction(reference), labels, items, bootstrap
                            )
                        )
                del summary
        dq_levels, dq_differences, dq_status = dq_cq_tables(
            config, evaluation, items, registry, binary, broad, empty, bootstrap, inputs
        )
        levels.extend(dq_levels)
        differences.extend(dq_differences)
        statuses[evaluation.name] = {
            "binary_items": binary.item_id.nunique(),
            "binary_cq_pairs": len(binary),
            "p7_annotated_items": broad.item_id.nunique(),
            "p7_annotated_cq_pairs": len(broad),
            "binary_population": B1_POPULATION,
            "p7_population": P7_POPULATION,
            "annotation_version": "1.1",
            "rewritten_cq_targets": "pending: no compatible 2.0 annotations",
            "dq": dq_status,
        }
    columns = [
        "evaluation",
        "system",
        "scheme_version",
        "population",
        "origin",
        "scheme",
        "cq_id",
        "n_items",
        "n_cq_pairs",
        "annotated_items",
        "brier_n",
        "agreement",
        "brier",
        "status",
    ]
    return {
        "cq_level.csv": table_frame(levels, columns),
        "cq_level_differences.csv": table_frame(
            differences,
            [
                "evaluation",
                "system",
                "reference",
                "scheme_version",
                "population",
                "origin",
                "scheme",
                "cq_id",
                "n_items",
                "n_cq_pairs",
                "brier_n",
                "agreement_difference",
                "agreement_low",
                "agreement_high",
                "brier_difference",
                "brier_low",
                "brier_high",
                "status",
            ],
        ),
    }, statuses
