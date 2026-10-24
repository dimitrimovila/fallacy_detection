"""The samples of the human annotation, and the sheet of the direct verdict.

Two samples of the test set are drawn together, with one seed, so that they are disjoint:
the CQ annotation on schemes 2.0 and the direct verdict.  Only the sheet of the direct
verdict is written; it is read by an annotation app outside the repository and shows,
for each item, the text, the name of its gold scheme and the verdicts of that scheme's
diagram as the zero-shot prompt lists them.  Nothing else about the item is in the sheet.

The sheet comes in rounds: round r holds, for every scheme that still has items, its
items ``size*(r-1)+1`` to ``size*r`` in a per-scheme order drawn with the seed, shuffled
again within the round.  A stop at the end of any round leaves the same number of items
for every scheme not yet exhausted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .annotations import ANNOTATIONS_FILE, ANNOTATOR, ITEMS_FILE, gold_scheme_rows, load_annotations
from .eval.evaluate import _commit
from .eval.prior import load_test_set
from .paths import REPO_ROOT
from .prompts import verdicts
from .schemes import SchemeError, load_all

SHEET_FILE = "direct_verdict_sheet.jsonl"
SAMPLES_FILE = "samples.csv"
MANIFEST_FILE = "manifest.json"
SHEET_FIELDS = ("position", "item_id", "scheme_name", "text", "verdicts")
DIRECT = "direct"
CQ_V2 = "cq_v2"
FINE_VERDICT_ANNOTATOR = "eleni"


@dataclass(frozen=True)
class SampleConfig:
    seed: int
    date: str
    round_size: int
    cq_v2_per_scheme: dict[str, int] = field(default_factory=dict)
    never_annotated: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> SampleConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cq_v2 = data["cq_v2"]
        return cls(seed=int(data["seed"]), date=str(data["date"]),
                   round_size=int(data["round_size"]),
                   cq_v2_per_scheme={str(k): int(v) for k, v in cq_v2["per_scheme"].items()},
                   never_annotated={str(k): int(v)
                                    for k, v in (cq_v2.get("never_annotated") or {}).items()})


@dataclass(frozen=True)
class Drawn:
    direct: pd.DataFrame
    """item_id, scheme, round, position: in the order of the sheet."""
    cq_v2: pd.DataFrame
    """item_id, scheme: sorted by scheme and item."""


def pool_a(items: pd.DataFrame, annotations: pd.DataFrame, schemes) -> set[str]:
    """The test items with a full 1.1 vector under the gold scheme.

    An item is in the pool when a row of an ``Annotazione Dumitru`` sheet whose CQs are
    those of the item's gold scheme judges the scheme to be the gold one and answers every
    CQ of it (``na`` and ``idk`` included).  A row that judges another scheme, ``none`` or
    ``idk`` does not count, even with every CQ filled in.  On the frozen data this gives
    476 items: 438 reach a verdict along the diagram, 35 of cause to effect stop at
    ``na`` and 3 stop at ``idk``.
    """
    return set(gold_scheme_rows(items, annotations, schemes, complete=True).item_id)


def eleni_items(items: pd.DataFrame, annotations: pd.DataFrame, schemes) -> set[str]:
    """The test items with a fine verdict (a terminal, not a coarse label) of ``eleni``."""
    terminals = set().union(*(s.terminals for s in schemes.values()))
    rows = annotations[(annotations.annotator == FINE_VERDICT_ANNOTATOR)
                       & (annotations.field == "verdict") & annotations.value.isin(terminals)]
    return set(rows.item_id) & set(items.item_id)


def _take(rng: np.random.Generator, candidates: list[str], n: int) -> list[str]:
    return sorted(candidates[i] for i in rng.choice(len(candidates), size=n, replace=False))


def draw(config: SampleConfig, items: pd.DataFrame, annotations: pd.DataFrame,
         schemes) -> Drawn:
    """Both samples.  A stratum with fewer items than requested stops the draw."""
    unknown = sorted((set(config.cq_v2_per_scheme) | set(config.never_annotated)) - set(schemes))
    if unknown:
        raise SchemeError(f"unknown schemes in the configuration: {unknown}")
    gold = items.set_index("item_id").gold_scheme
    pool = pool_a(items, annotations, schemes)
    fine = eleni_items(items, annotations, schemes)
    annotated = set(annotations.loc[annotations.annotator == ANNOTATOR, "item_id"])

    strata = {}
    for scheme, n in sorted(config.cq_v2_per_scheme.items()):
        strata[("pool", scheme)] = (sorted(i for i in pool - fine if gold[i] == scheme), n)
    for scheme, n in sorted(config.never_annotated.items()):
        strata[("never_annotated", scheme)] = (
            sorted(i for i in gold.index if gold[i] == scheme and i not in annotated), n)
    short = [f"{scheme} ({where}): {len(c)} available, {n} requested"
             for (where, scheme), (c, n) in strata.items() if len(c) < n]
    if short:
        raise SchemeError("not enough items: " + "; ".join(short))

    rng = np.random.default_rng(config.seed)
    cq_v2 = [(i, scheme) for (_, scheme), (c, n) in strata.items() for i in _take(rng, c, n)]
    taken = {i for i, _ in cq_v2}

    direct_ids = sorted(pool - taken)
    by_scheme = {}
    for scheme in sorted({gold[i] for i in direct_ids}):
        ids = [i for i in direct_ids if gold[i] == scheme]
        by_scheme[scheme] = [ids[k] for k in rng.permutation(len(ids))]
    size = config.round_size
    rows, number = [], 0
    while any(len(ids) > size * number for ids in by_scheme.values()):
        chunk = [(i, s) for s, ids in by_scheme.items()
                 for i in ids[size * number:size * (number + 1)]]
        number += 1
        rows.extend((i, s, number) for i, s in (chunk[k] for k in rng.permutation(len(chunk))))
    direct = pd.DataFrame(rows, columns=["item_id", "scheme", "round"])
    direct["position"] = range(1, len(direct) + 1)
    cq_frame = pd.DataFrame(cq_v2, columns=["item_id", "scheme"]).sort_values(
        ["scheme", "item_id"], ignore_index=True)
    return Drawn(direct=direct, cq_v2=cq_frame)


def sheet_lines(direct: pd.DataFrame, items: pd.DataFrame, schemes,
                vocabulary: Mapping | None = None) -> list[str]:
    """One JSON object per item of the direct sample, in the order of position."""
    table = items.set_index("item_id")
    lines = []
    for row in direct.itertuples(index=False):
        scheme = schemes[row.scheme]
        record = {"position": int(row.position), "item_id": row.item_id,
                  "scheme_name": scheme.name, "text": table.text[row.item_id],
                  "verdicts": [{"id": v.terminal, "label": v.shown, "definition": v.definition}
                               for v in verdicts(scheme, vocabulary)]}
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    return lines


def _samples_csv(drawn: Drawn) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["item_id", "sample", "scheme", "round", "position"])
    for row in drawn.direct.itertuples(index=False):
        writer.writerow([row.item_id, DIRECT, row.scheme, row.round, row.position])
    for row in drawn.cq_v2.itertuples(index=False):
        writer.writerow([row.item_id, CQ_V2, row.scheme, "", ""])
    return buffer.getvalue()


def write_samples(config: SampleConfig, out: str | Path, items_path: str | Path = ITEMS_FILE,
                  annotations_path: str | Path = ANNOTATIONS_FILE) -> dict:
    """Write the sheet, samples.csv and manifest.json into ``out``; return the manifest.

    ``out`` must be outside the repository: the texts come from corpora with their own terms.
    """
    out = Path(out).resolve()
    if out.is_relative_to(REPO_ROOT.resolve()):
        raise SchemeError(f"{out} is inside the repository; write the sheets outside it")
    items, annotations = load_test_set(items_path), load_annotations(annotations_path)
    schemes = load_all()
    drawn = draw(config, items, annotations, schemes)
    files = {SHEET_FILE: "".join(sheet_lines(drawn.direct, items, schemes)),
             SAMPLES_FILE: _samples_csv(drawn)}
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, content in files.items():
        data = content.encode("utf-8")
        (out / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    manifest = {
        "seed": config.seed,
        "date": config.date,
        "git_commit": _commit(),
        "counts": {name: {s: int(n) for s, n in frame.scheme.value_counts().sort_index().items()}
                   for name, frame in ((DIRECT, drawn.direct), (CQ_V2, drawn.cq_v2))},
        "rounds": int(drawn.direct["round"].max()),
        "sha256": digests,
    }
    (out / MANIFEST_FILE).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest
