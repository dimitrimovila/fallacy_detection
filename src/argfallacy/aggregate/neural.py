"""Small pooled neural networks, with label-isolated nested model selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..eval.folds import assign_folds
from ..schemes import SchemeError
from .interface import TIE, Aggregation, decide, macro_f1
from .neural_inputs import NeuralInputs


@dataclass(frozen=True)
class NetworkConfig:
    hidden: int = 8
    weight_decay: float = 0.1
    learning_rate: float = 0.01
    epochs: int = 200
    patience: int = 30
    tolerance: float = 1e-6

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.hidden, self.epochs, self.patience)):
            raise SchemeError("hidden, epochs and patience must be positive integers")
        values = [self.weight_decay, self.learning_rate, self.tolerance]
        if not np.isfinite(values).all() or min(values) < 0 or self.learning_rate == 0:
            raise SchemeError("invalid neural optimization parameters")


class CQNetwork(nn.Module):
    def __init__(self, inputs: int, hidden: int, classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, classes)
        )

    def forward(self, x):
        return self.layers(x)


@dataclass
class FittedNetwork:
    model: CQNetwork
    classes: tuple[str, ...]
    names: tuple[str, ...]
    config: NetworkConfig
    seed: int
    epochs_run: int

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != len(self.names) or not np.isfinite(x).all():
            raise SchemeError("invalid neural prediction matrix")
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.as_tensor(x, dtype=torch.float32)).softmax(-1).numpy()

    def predict(self, inputs: NeuralInputs) -> list[Aggregation]:
        if inputs.names != self.names:
            raise SchemeError("neural checkpoint feature schema mismatch")
        return [
            decide(
                dict(zip(self.classes, map(float, row), strict=True)),
                0.0,
                "Pooled neural CQ classifier; no diagram path explanation.",
            )
            for row in self.probabilities(inputs.x)
        ]

    def save(self, path: str | Path):
        torch.save(
            {
                "format": 1,
                "state": self.model.state_dict(),
                "classes": self.classes,
                "names": self.names,
                "config": asdict(self.config),
                "seed": self.seed,
                "epochs_run": self.epochs_run,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FittedNetwork:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format"] != 1:
            raise SchemeError("unsupported neural checkpoint format")
        config = NetworkConfig(**payload["config"])
        with torch.random.fork_rng(devices=[]):
            model = CQNetwork(len(payload["names"]), config.hidden, len(payload["classes"]))
        model.load_state_dict(payload["state"])
        return cls(
            model,
            tuple(payload["classes"]),
            tuple(payload["names"]),
            config,
            payload["seed"],
            payload["epochs_run"],
        )


def fit_network(
    x: np.ndarray, y: Sequence[str], names: Sequence[str], config: NetworkConfig, seed: int
) -> FittedNetwork:
    """Fit only supplied rows; class weights and stopping use training data alone."""
    if (
        x.ndim != 2
        or not len(x)
        or len(y) != len(x)
        or x.shape[1] != len(names)
        or not np.isfinite(x).all()
    ):
        raise SchemeError("invalid or empty neural training matrix")
    classes = tuple(sorted(set(y)))
    targets = np.array([classes.index(label) for label in y])
    counts = np.bincount(targets, minlength=len(classes))
    weights = torch.tensor(len(y) / (len(classes) * counts), dtype=torch.float32)
    # Preserve the caller's random state; CPU full-batch fits are repeatable in one environment.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CQNetwork(x.shape[1], config.hidden, len(classes))
        features = torch.as_tensor(x, dtype=torch.float32)
        target = torch.as_tensor(targets, dtype=torch.long)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        best, stale, epochs_run = float("inf"), 0, 0
        if len(classes) > 1:
            model.train()
            for epoch in range(config.epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(features), target)
                if not torch.isfinite(loss):
                    raise SchemeError("nonfinite neural training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
                epochs_run = epoch + 1
                value = float(loss.detach())
                if best - value > config.tolerance:
                    best, stale = value, 0
                else:
                    stale += 1
                if stale >= config.patience:
                    break
    return FittedNetwork(model, classes, tuple(names), config, seed, epochs_run)


def select_network(
    inputs: NeuralInputs,
    targets: Mapping[str, str],
    train: Sequence[int],
    candidates: Sequence[NetworkConfig],
    seed: int,
    split_seed: int,
    inner_folds: int,
) -> tuple[NetworkConfig, list[dict]]:
    """Inner out-of-fold predictions select the first best candidate by macro F1."""
    if len(train) < inner_folds or inner_folds < 2 or not candidates:
        raise SchemeError("neural selection needs candidates and at least two inner folds")
    inner_items = pd.DataFrame(
        {
            "item_id": [inputs.item_ids[i] for i in train],
            "gold_scheme": [inputs.scheme_ids[i] for i in train],
        }
    )
    folds = assign_folds(inner_items, split_seed, inner_folds)
    truth = [targets[inputs.item_ids[i]] for i in train]
    best, selected, records = -float("inf"), candidates[0], []
    for candidate_index, candidate in enumerate(candidates):
        predictions = {}
        for fold in range(inner_folds):
            fit = [i for i in train if folds[inputs.item_ids[i]] != fold]
            valid = [i for i in train if folds[inputs.item_ids[i]] == fold]
            model = fit_network(
                inputs.x[fit],
                [targets[inputs.item_ids[i]] for i in fit],
                inputs.names,
                candidate,
                seed + fold,
            )
            for index, row in zip(valid, model.probabilities(inputs.x[valid]), strict=True):
                predictions[index] = decide(
                    dict(zip(model.classes, row, strict=True)), 0.0, ""
                ).verdict
        score = macro_f1(truth, [predictions[i] for i in train])
        records.append({"candidate": candidate_index, **asdict(candidate), "macro_f1": score})
        if score > best + TIE:
            best, selected = score, candidate
    return selected, records


def neural_out_of_fold(
    inputs: NeuralInputs,
    targets: Mapping[str, str],
    folds: Mapping[str, int],
    candidates: Sequence[NetworkConfig],
    seed: int,
    split_seed: int,
    inner_folds: int = 3,
    checkpoint_dir: Path | None = None,
    progress=None,
):
    """Return predictions, candidate scores, and fitted outer models, indexed by fold."""
    ids = inputs.item_ids
    if set(ids) != set(targets) or set(ids) != set(folds) or len(set(folds.values())) < 2:
        raise SchemeError("neural targets/folds must cover exactly the input IDs")
    outputs, records, fitted = {}, [], {}
    for fold in sorted(set(folds.values())):
        train = [i for i, item in enumerate(ids) if folds[item] != fold]
        test = [i for i, item in enumerate(ids) if folds[item] == fold]
        if progress:
            progress(f"seed {seed}, outer fold {fold}: selecting and fitting")
        # Pass only training labels to selection, so accidental test-label reads fail.
        train_targets = {ids[i]: targets[ids[i]] for i in train}
        selected, choices = select_network(
            inputs, train_targets, train, candidates, seed, split_seed, inner_folds
        )
        model = fit_network(
            inputs.x[train], [train_targets[ids[i]] for i in train], inputs.names, selected, seed
        )
        for row in choices:
            records.append(
                {
                    "seed": seed,
                    "fold": fold,
                    **row,
                    "selected": all(row[k] == v for k, v in asdict(selected).items()),
                    "outer_epochs_run": model.epochs_run,
                }
            )
        for i, probabilities in zip(test, model.probabilities(inputs.x[test]), strict=True):
            outputs[ids[i]] = decide(
                dict(zip(model.classes, map(float, probabilities), strict=True)),
                0.0,
                "Pooled neural CQ classifier; no diagram path explanation.",
            )
        if checkpoint_dir is not None:
            model.save(checkpoint_dir / f"seed_{seed}_fold_{fold}.pt")
        fitted[fold] = model
    return outputs, records, fitted
