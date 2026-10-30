"""Invent a full-size saved-evaluation workload without reading research data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from argfallacy.aggregate import a0
from argfallacy.client.run import schemes_hash
from argfallacy.eval.analysis_dq import DQ_SCHEMAS
from argfallacy.eval.evaluate import Scorer
from argfallacy.eval.folds import assign_folds
from argfallacy.paths import SCHEMES_ROOT
from argfallacy.schemes import load_all

ENTRIES = ["qwen3_8_27b", "gemma4_31b", "gpt_oss_20b", "gpt_oss_120b", "k2_horizon_32b_high"]
FIRST = (
    [f"A0:{e}" for e in ENTRIES]
    + ["A1", "A2", "A3", "A4", "A5"]
    + [f"zeroshot:{e}" for e in ENTRIES]
    + ["majority"]
    + [f"ceiling:{e}" for e in ENTRIES]
)
SECOND = [
    "A0:qwen3_8_27b",
    "A2",
    "A4",
    "zeroshot:qwen3_8_27b",
    "A5",
    "A5.answers_only",
    "A3",
    "majority",
    "B1",
    "B2",
    "B3",
    "B4",
    "B1.hard",
    "A0:qwen3_8_27b@2.0",
    "A2.pruned",
    "B4.path",
    "B4.nohistory",
    "B1.majority.none.soft",
    "B1.majority.none.hard",
    "B1.dawid_skene.none.soft",
    "B1.dawid_skene.none.hard",
    "B1.weighted.cq.soft",
    "B1.weighted.cq.hard",
    "B1.stacking.gold.soft",
    "B1.stacking.gold.hard",
    "B1.annotation_exclusion",
    "B1.majority.none.soft@2.0",
    "B1.majority.none.hard@2.0",
    "B1.dawid_skene.none.soft@2.0",
    "B1.dawid_skene.none.hard@2.0",
    "B1.stacking.gold.soft@2.0",
    "B1.stacking.gold.hard@2.0",
    "B1@2.0",
    "B1.hard@2.0",
    "B2.A0",
    "B3.pooled",
    "A3.tree",
    "B1.stacking.gold_balanced.soft",
    "B1.stacking.gold_balanced.hard",
    "B3.pooled.balanced",
    "A2.prior",
    "B1.prior",
    "A3.combined",
]


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def build(root, batch_size=128):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    schemes = {v: load_all(SCHEMES_ROOT / folder) for v, folder in [("1.1", "v1"), ("2.0", "v2")]}
    scheme_names = sorted(schemes["1.1"])
    n, seed, resamples = 601, 20261019, 10000
    rng = np.random.default_rng(8351)
    items = pd.DataFrame(
        [
            dict(
                item_id=f"synthetic_{i:04}",
                source="LOGIC"
                if i < 240
                else "Reddit"
                if i < 255
                else "ElecDebate"
                if i < 260
                else "NLAS"
                if i < 570
                else "EthiX",
                gold_scheme=scheme_names[i % len(scheme_names)],
                gold_fallacy=sorted(schemes["1.1"][scheme_names[i % len(scheme_names)]].terminals)[
                    i % len(schemes["1.1"][scheme_names[i % len(scheme_names)]].terminals)
                ],
                in_test_set=True,
            )
            for i in range(n)
        ]
    )
    items.to_csv(root / "items.csv", index=False)
    folds = assign_folds(items, seed)
    parsed = {}
    cq_counts = {}
    for version, run_id in [("1.1", "full_v1"), ("2.0", "full_v2")]:
        directory = root / run_id
        directory.mkdir()
        parsed[run_id] = run_id
        rows = []
        for i, item in enumerate(items.itertuples()):
            for entry in ENTRIES:
                for cq in schemes[version][item.gold_scheme].cqs:
                    rows.append(
                        dict(
                            item_id=item.item_id,
                            scheme=item.gold_scheme,
                            stage="stage2",
                            scheme_condition="gold",
                            cq_id=cq,
                            model=entry,
                            answer="yes" if i % 3 else "no",
                            p_logprob_yes=0.65,
                            p_logprob_no=0.35,
                            p_logprob=0.65,
                            p_verbal=0.7,
                            n_samples_ok=1,
                            invalid_count=0,
                        )
                    )
                if version == "1.1":
                    rows.append(
                        dict(
                            item_id=item.item_id,
                            scheme=item.gold_scheme,
                            stage="stage1",
                            scheme_condition="gold",
                            cq_id="",
                            model=entry,
                            answer=item.gold_scheme,
                            n_samples_ok=1,
                            invalid_count=0,
                        )
                    )
        summary = pd.DataFrame(rows)
        cq_counts[run_id] = int(summary.stage.eq("stage2").sum())
        summary.to_csv(directory / "summary.csv", index=False)
        summary.assign(sample_index=0, parse_ok=True, finish_reason="stop").to_csv(
            directory / "answers.csv", index=False
        )
        dump(
            directory / "manifest.json",
            dict(
                run_id=run_id,
                schemes_version=dict(tag=version),
                models={e: dict(samples=1) for e in ENTRIES},
                config=dict(models=ENTRIES),
            ),
        )
    notes = []
    for i, item in enumerate(items.itertuples()):
        for field, value in {
            "scheme": item.gold_scheme,
            **{cq: "yes" if i % 2 else "no" for cq in schemes["1.1"][item.gold_scheme].cqs},
        }.items():
            notes.append(
                dict(
                    item_id=item.item_id,
                    annotator="dumitru",
                    sheet=item.gold_scheme,
                    row=i + 2,
                    field=field,
                    value=value,
                )
            )
        for person in ("dumitru", "enrico", "eleni"):
            notes.append(
                dict(
                    item_id=item.item_id,
                    annotator=person,
                    sheet=item.gold_scheme,
                    row=i + 2,
                    field="verdict",
                    value=item.gold_fallacy,
                )
            )
        notes.append(
            dict(
                item_id=item.item_id,
                annotator="enrico",
                sheet=item.gold_scheme,
                row=i + 2,
                field="difficulty",
                value=str(1 + i % 5),
            )
        )
    pd.DataFrame(notes).to_csv(root / "frozen.csv", index=False)
    for name, round_number, names in [("first", 1, FIRST), ("second", 2, SECOND)]:
        directory = root / name
        directory.mkdir()
        manifest = dict(
            name=name,
            round=round_number,
            seed=seed,
            resamples=resamples,
            items=n,
            items_without_nlas=int(items.source.ne("NLAS").sum()),
            folds=dict(k=5, assignment=folds),
            reference_model=ENTRIES[0],
            schemes=dict(tag="1.1", answers_schemes_2="v2"),
            runs=dict(
                answers="full_v1",
                **(dict(answers_schemes_2="full_v2") if round_number == 2 else {}),
            ),
            m=5 if round_number == 1 else 4,
            cq_label_exclusion=["example"],
            primary_configurations={},
            source_commit_status="synthetic",
            conformal_alpha=0.1,
        )
        rows, main, combined = [], [], []
        for k, system in enumerate(names):
            version = (
                "2.0"
                if system.endswith("@2.0")
                or system in ("B4", "B4.path", "B4.nohistory", "A2.pruned")
                else "1.1"
            )
            own = (
                items[items.gold_scheme.ne("example")]
                if system == "B1.annotation_exclusion"
                else items
            )
            main.append(
                dict(
                    system=system,
                    schemes=version,
                    n_items=len(own),
                    role="oracle"
                    if system == "majority" or system.startswith("ceiling:")
                    else "comparison",
                    description="invented saved result",
                )
            )
            for i, item in enumerate(own.itertuples()):
                menu = sorted(schemes[version][item.gold_scheme].terminals)
                prediction = (
                    item.gold_fallacy
                    if i % 5 and item.gold_fallacy in menu
                    else menu[k % len(menu)]
                )
                rows.append(
                    dict(
                        system=system,
                        item_id=item.item_id,
                        fold=folds[item.item_id],
                        verdict=prediction if i % 17 else "",
                        confidence=round(float(rng.random()), 3),
                    )
                )
                if round_number == 2 and system.startswith("B1"):
                    combined.extend(
                        dict(
                            system=system,
                            schemes=version,
                            supervision="none",
                            item_id=item.item_id,
                            scheme=item.gold_scheme,
                            cq_id=cq,
                            fold=folds[item.item_id],
                            p_first=0.65,
                            outside=0.0,
                        )
                        for cq in schemes[version][item.gold_scheme].cqs
                    )
        pd.DataFrame(rows).to_csv(directory / "verdicts.csv", index=False)
        pd.DataFrame(main).to_csv(directory / "main_table.csv", index=False)
        if combined:
            pd.DataFrame(combined).to_csv(directory / "cq_combined.csv", index=False)
        primary = (
            [("A" + str(i), "A0:" + ENTRIES[0]) for i in range(1, 6)]
            if round_number == 1
            else [
                ("B1", "A0:" + ENTRIES[0]),
                ("B1", "zeroshot:" + ENTRIES[0]),
                ("B2", "zeroshot:" + ENTRIES[0]),
                ("B3", "B1.hard"),
                ("B3", "zeroshot:" + ENTRIES[0]),
                ("B4", "A2.pruned"),
            ]
        )
        pd.DataFrame(
            [
                dict(
                    system=a,
                    reference=b,
                    level=1 - 0.05 / manifest["m"],
                    read=True,
                    computed=True,
                    read_after="",
                )
                for a, b in primary
            ]
        ).to_csv(directory / "deciding.csv", index=False)
        pd.DataFrame(
            [
                dict(system=a, reference=b, level=0.95)
                for a, b in (
                    [("A2", "A4")]
                    if round_number == 1
                    else [("A3.combined", "A3"), ("B1.hard", "B1"), ("B4.nohistory", "B4")]
                )
            ]
        ).to_csv(directory / "exploratory.csv", index=False)
        dump(directory / "manifest.json", manifest)
        if round_number == 2:
            pd.DataFrame(
                [
                    dict(
                        system="A2",
                        kind="total",
                        subset="all",
                        n_items=n,
                        coverage=0.9,
                        mean_size=2.0,
                        singleton_share=0.4,
                    )
                ]
            ).to_csv(directory / "conformal.csv", index=False)
    # Four explicit parsed spec-10 sources and eleven distinct imported systems.
    directory = root / "second"
    manifest = json.loads((directory / "manifest.json").read_text())
    mappings, run_mappings, predictions, main = {}, {}, [], []
    for role, version, stage, prefix in [
        ("dq", "1.1", "dq", "DQ"),
        ("dq_schemes_2", "2.0", "dq", "DQ"),
        ("zeroshot_schemes_2", "2.0", "zeroshot", "zeroshot"),
        ("zeroshot_rerun", "1.1", "zeroshot", "zeroshot.rerun"),
    ]:
        run_id = "invented_" + role
        d = root / run_id
        d.mkdir()
        parsed[run_id] = run_id
        run_mappings[role] = [run_id]
        suffix = "@2.0" if version == "2.0" else ""
        names = [prefix + ":" + ENTRIES[0] + suffix]
        if stage == "dq":
            names += ["DQ.path:" + ENTRIES[0] + suffix, "DQ.soft:" + ENTRIES[0] + suffix]
        rows = []
        for item in items.itertuples():
            scheme = schemes[version][item.gold_scheme]
            answer = sorted(scheme.terminals)[0]
            base = dict(
                item_id=item.item_id,
                scheme=item.gold_scheme,
                model=ENTRIES[0],
                stage=stage,
                scheme_condition="gold",
            )
            rows.append(
                dict(**base, cq_id="", answer=answer, p_label=0.81, p_label_lower_bound=False)
            )
            own = {cq: dict(answer="yes") for cq in scheme.cqs}
            if stage == "dq":
                rows.extend(
                    dict(
                        **base,
                        cq_id=cq,
                        answer="yes",
                        p_logprob_yes=0.65,
                        p_logprob_no=0.35,
                        p_logprob=0.65,
                    )
                    for cq in scheme.cqs
                )
            hard = a0(scheme, own)
            for system in names:
                predictions.append(
                    dict(
                        system=system,
                        item_id=item.item_id,
                        fold=folds[item.item_id],
                        verdict=hard.verdict if system.startswith("DQ.path") else answer,
                        confidence=hard.confidence
                        if system.startswith("DQ.path")
                        else 0.61
                        if system.startswith("DQ.soft")
                        else 0.81,
                        abstention=0.0,
                        explanation="invented",
                    )
                )
        pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
        cq_counts[run_id] = sum(bool(r["cq_id"]) for r in rows)
        folder = SCHEMES_ROOT / ("v1" if version == "1.1" else "v2")
        dump(
            d / "manifest.json",
            dict(
                run_id=run_id,
                schemes_version=dict(tag=version, content_sha256=schemes_hash(folder)),
                prompt_versions=[stage + "_v1"],
                models={ENTRIES[0]: dict(samples=1)},
                config=dict(models=[ENTRIES[0]], stages=[stage]),
            ),
        )
        for system in names:
            mappings[system] = run_id
            main.append(
                dict(
                    system=system,
                    schemes=version,
                    role="comparison",
                    n_items=n,
                    description="invented saved DQ result",
                )
            )
    for system, version in [("B2.dq", "1.1"), ("B2@2.0", "2.0"), ("B2.dq@2.0", "2.0")]:
        mappings[system] = dict(
            answers="full_v1" if version == "1.1" else "full_v2",
            fit_answers="full_v1",
            zeroshot="saved_zero_not_configured"
            if version == "1.1"
            else "invented_zeroshot_schemes_2",
        )
        if system != "B2@2.0":
            mappings[system]["dq"] = "invented_dq" if version == "1.1" else "invented_dq_schemes_2"
        if system == "B2.dq@2.0":
            # Mapping only; no S-soft parsed path workload is claimed by this benchmark.
            manifest["runs"]["ssoft"] = ["synthetic_ssoft_not_configured"]
            mappings[system]["ssoft"] = manifest["runs"]["ssoft"]
        main.append(
            dict(
                system=system,
                schemes=version,
                role="exploratory",
                n_items=n,
                description="invented routing prediction",
            )
        )
        predictions.extend(
            dict(
                system=system,
                item_id=i.item_id,
                fold=folds[i.item_id],
                verdict=sorted(schemes[version][i.gold_scheme].terminals)[0],
                confidence=0.6,
                abstention=0.0,
                explanation="invented",
            )
            for i in items.itertuples()
        )
    manifest["zeroshot_sources"] = {ENTRIES[0]: "saved_zero_not_configured"}
    manifest["dq"] = dict(
        runs=run_mappings,
        systems=mappings,
        items=n,
        folds=folds,
        family=dict(name="H_DQ", m=1, level=0.95),
    )
    dump(directory / "manifest.json", manifest)
    for filename, columns in DQ_SCHEMAS.items():
        pd.DataFrame(columns=columns).to_csv(directory / filename, index=False)
    pd.DataFrame(predictions).to_csv(directory / "dq_verdicts.csv", index=False)
    for row in main:
        chosen = [p["verdict"] for p in predictions if p["system"] == row["system"]]
        metrics = Scorer(items.gold_fallacy.tolist(), np.ones((1, n))).metrics(chosen)
        for metric, value in metrics.items():
            for suffix in ("", "_low", "_high"):
                row[metric + suffix] = value[0]
    pd.DataFrame(main).to_csv(directory / "dq_main_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(
        [
            dict(
                hypothesis="H_DQ",
                system="DQ:" + ENTRIES[0] + "@2.0",
                reference="zeroshot:" + ENTRIES[0] + "@2.0",
                level=0.95,
                computed=True,
                outcome="no measurable difference",
                reading="not applicable",
                macro_f1_difference=0,
                macro_f1_low=-0.1,
                macro_f1_high=0.1,
            )
        ]
    ).to_csv(directory / "dq_deciding.csv", index=False)
    config = dict(
        version=1,
        items="items.csv",
        annotations="frozen.csv",
        output="out",
        seed=seed,
        resamples=resamples,
        bootstrap_batch_size=batch_size,
        evaluations=dict(first=dict(path="first", round=1), second=dict(path="second", round=2)),
        parsed_runs=parsed,
        direct_verdict=dict(answers=None, samples=None),
        dq="second",
    )
    path = root / "analysis.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, dict(
        items=n,
        resamples=resamples,
        base_systems=[len(FIRST), len(SECOND)],
        imported_dq_systems=len(main),
        parsed_cq_rows=cq_counts,
        limitations="No new human inputs, exact routing replay, genuine conformal exports, "
        "or S-soft path occurrences. Includes frozen synthetic human and difficulty "
        "ratings, five CQ entries, combined CQ vectors, all verdict system identities, "
        "four spec-10 runs and imported DQ tables. No fitting or model calls.",
    )
