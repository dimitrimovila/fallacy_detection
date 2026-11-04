"""The outer and inner splits of one boundary, shared by the ACT stage and the cost estimate."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ActConfig
from .data import LabeledItem, inner_split, outer_split


@dataclass(frozen=True)
class FoldSplit:
    outer_fold: int
    inner_train: list[LabeledItem]
    inner_val: list[LabeledItem]
    test: list[LabeledItem]


def fold_splits(items: list[LabeledItem], act: ActConfig) -> list[FoldSplit]:
    """One split per outer fold; the inner split seed is ``split_seed + outer_fold``."""
    out = []
    for fold in sorted({i.outer_fold for i in items}):
        train, test = outer_split(items, fold)
        inner_train, inner_val = inner_split(train, act.validation_fraction,
                                             act.split_seed + fold)
        out.append(FoldSplit(fold, inner_train, inner_val, test))
    return out
