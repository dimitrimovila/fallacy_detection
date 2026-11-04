"""The input files and the splits.

Two CSV files are read, both prepared outside this package:

* ``items.csv``: ``item_id``, ``text``, ``scheme``, ``gold``, ``source`` (the column names
  can be remapped in the configuration, ``data.columns``);
* ``folds.csv``: ``item_id``, ``outer_fold`` (an integer from 0 to ``n_outer_folds - 1``).

Scheme and gold labels are compared in lower case. Every item of a boundary must have an
outer fold; items of other schemes or other classes are ignored.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import Boundary


class DataError(ValueError):
    """Raised when the input files do not match the documented format."""


@dataclass(frozen=True)
class Item:
    item_id: str
    text: str
    scheme: str
    gold: str
    source: str


@dataclass(frozen=True)
class LabeledItem:
    """An item of a boundary, with ``label`` 1 for the positive class and 0 otherwise."""

    item_id: str
    text: str
    source: str
    gold: str
    label: int
    outer_fold: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        raise DataError(f"{path}: file not found")
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_items(path: Path, columns: dict[str, str]) -> list[Item]:
    """The items are read with the column mapping of the configuration."""
    rows = _read_csv(path)
    if rows:
        missing = [c for c in columns.values() if c not in rows[0]]
        if missing:
            raise DataError(f"{path}: missing columns {missing}")
    items, seen = [], set()
    for n, row in enumerate(rows, start=2):
        item_id = (row[columns["item_id"]] or "").strip()
        text = (row[columns["text"]] or "").strip()
        if not item_id:
            raise DataError(f"{path}:{n}: empty item_id")
        if item_id in seen:
            raise DataError(f"{path}:{n}: duplicate item_id {item_id!r}")
        seen.add(item_id)
        items.append(
            Item(
                item_id=item_id,
                text=text,
                scheme=(row[columns["scheme"]] or "").strip().lower(),
                gold=(row[columns["gold"]] or "").strip().lower(),
                source=(row[columns["source"]] or "").strip() or "unknown",
            )
        )
    return items


def load_folds(path: Path, n_outer_folds: int) -> dict[str, int]:
    """The outer fold of every listed item is read and range-checked."""
    rows = _read_csv(path)
    if rows and not {"item_id", "outer_fold"} <= set(rows[0]):
        raise DataError(f"{path}: the columns item_id and outer_fold are required")
    folds: dict[str, int] = {}
    for n, row in enumerate(rows, start=2):
        item_id = (row["item_id"] or "").strip()
        try:
            fold = int(row["outer_fold"])
        except (TypeError, ValueError) as exc:
            raise DataError(f"{path}:{n}: outer_fold is not an integer") from exc
        if not 0 <= fold < n_outer_folds:
            raise DataError(f"{path}:{n}: outer_fold {fold} outside 0..{n_outer_folds - 1}")
        if item_id in folds:
            raise DataError(f"{path}:{n}: duplicate item_id {item_id!r}")
        folds[item_id] = fold
    return folds


def boundary_items(
    items: list[Item], folds: dict[str, int], boundary: Boundary
) -> list[LabeledItem]:
    """The items of one boundary are selected and labelled; texts must not be empty."""
    wanted = {boundary.positive, boundary.negative}
    selected = [i for i in items if i.scheme == boundary.scheme and i.gold in wanted]
    missing = [i.item_id for i in selected if i.item_id not in folds]
    if missing:
        raise DataError(
            f"{boundary.id}: {len(missing)} items have no outer fold, e.g. {missing[:5]}"
        )
    empty = [i.item_id for i in selected if not i.text]
    if empty:
        raise DataError(f"{boundary.id}: empty texts, e.g. {empty[:5]}")
    out = [
        LabeledItem(
            item_id=i.item_id,
            text=i.text,
            source=i.source,
            gold=i.gold,
            label=int(i.gold == boundary.positive),
            outer_fold=folds[i.item_id],
        )
        for i in selected
    ]
    counts = Counter(i.label for i in out)
    if counts[0] == 0 or counts[1] == 0:
        raise DataError(
            f"{boundary.id}: both classes are needed; found {counts[1]} positive and "
            f"{counts[0]} negative items"
        )
    return out


def outer_split(
    items: list[LabeledItem], fold: int
) -> tuple[list[LabeledItem], list[LabeledItem]]:
    """The outer training set (all other folds) and the outer test set (this fold)."""
    train = [i for i in items if i.outer_fold != fold]
    test = [i for i in items if i.outer_fold == fold]
    return train, test


def inner_split(
    train: list[LabeledItem], fraction: float, seed: int
) -> tuple[list[LabeledItem], list[LabeledItem]]:
    """A stratified validation share is drawn from the outer training set.

    Strata are (label, source) pairs, so that the inner validation set keeps the mix of
    sources of each class. Each stratum contributes ``round(fraction * size)`` items; when a
    class would end up with no validation item although it has at least two training items,
    one item of that class is moved to validation. The draw depends only on ``seed`` and on
    the item ids, not on the order of the rows.
    """
    rng = random.Random(seed)
    strata: dict[tuple[int, str], list[LabeledItem]] = defaultdict(list)
    for item in sorted(train, key=lambda i: i.item_id):
        strata[(item.label, item.source)].append(item)
    val_ids: set[str] = set()
    for key in sorted(strata):
        members = strata[key][:]
        rng.shuffle(members)
        k = int(round(fraction * len(members)))
        val_ids.update(m.item_id for m in members[:k])
    for label in (0, 1):
        members = sorted((i for i in train if i.label == label), key=lambda i: i.item_id)
        if len(members) >= 2 and not any(m.item_id in val_ids for m in members):
            val_ids.add(rng.choice(members).item_id)
        if members and all(m.item_id in val_ids for m in members):
            val_ids.discard(members[0].item_id)
    inner_train = [i for i in train if i.item_id not in val_ids]
    inner_val = [i for i in train if i.item_id in val_ids]
    return inner_train, inner_val


def class_source_table(items: list[LabeledItem]) -> dict[str, dict[int, int]]:
    """Counts per source and label, used by the report and by the per-source analysis."""
    table: dict[str, dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0})
    for item in items:
        table[item.source][item.label] += 1
    return dict(sorted(table.items()))
