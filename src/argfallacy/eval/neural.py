"""Isolated evaluation and read-only preflight of pooled neural CQ classifiers."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..aggregate.neural_inputs import build_neural_inputs
from ..client.run import schemes_dir_of
from ..labels import label_space
from ..paths import FALLACIES_FILE, REPO_ROOT
from ..schemes import SchemeError, load_all
from .folds import assign_folds
from .prior import load_test_set


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(config_path):
    """Check complete data contracts without importing PyTorch or fitting a model."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    required = {
        "run",
        "items",
        "entries",
        "fold_seed",
        "folds",
        "inner_folds",
        "seeds",
        "threads",
        "candidates",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise SchemeError(f"neural configuration requires exactly {sorted(required)}")
    for key in ("folds", "inner_folds", "threads"):
        minimum = 1 if key == "threads" else 2
        if type(config[key]) is not int or config[key] < minimum:
            raise SchemeError(f"invalid neural {key}")
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in [*seeds, config["fold_seed"]])
    ):
        raise SchemeError("neural seeds must be distinct nonnegative integers")
    if not isinstance(config["entries"], list) or not config["entries"]:
        raise SchemeError("neural entries must be an explicit list")
    if not isinstance(config["candidates"], list) or not config["candidates"]:
        raise SchemeError("neural candidates must be an explicit nonempty list")
    # Validate settings independently of the optional training dependency.
    allowed = {"hidden", "weight_decay", "learning_rate", "epochs", "patience", "tolerance"}
    for candidate in config["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != allowed:
            raise SchemeError(f"each neural candidate requires {sorted(allowed)}")
        for key in ("hidden", "epochs", "patience"):
            if type(candidate[key]) is not int or candidate[key] < 1:
                raise SchemeError(f"invalid neural candidate {key}")
        for key in ("weight_decay", "learning_rate", "tolerance"):
            v = candidate[key]
            if not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise SchemeError(f"invalid neural candidate {key}")
        if candidate["learning_rate"] == 0:
            raise SchemeError("learning_rate must be positive")
    run_dir = REPO_ROOT / config["run"]
    manifest_path, summary_path = run_dir / "manifest.json", run_dir / "summary.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scheme_dir = schemes_dir_of(manifest)
    schemes = load_all(scheme_dir)
    items_path = REPO_ROOT / config["items"]
    items = load_test_set(items_path).sort_values("item_id").reset_index(drop=True)
    bad = items.loc[~items.gold_fallacy.isin(label_space()), ["item_id", "gold_fallacy"]]
    if not bad.empty:
        raise SchemeError(f"noncanonical neural target labels: {bad.to_dict('records')}")
    # Preserve the literal answer 'na'; empty numeric cells are handled by the encoder.
    summary = pd.read_csv(summary_path, keep_default_na=False, dtype={"item_id": str})
    inputs = build_neural_inputs(schemes, items, summary, manifest, config["entries"])
    own = summary[
        summary.stage.eq("stage2")
        & summary.scheme_condition.eq("gold")
        & summary.item_id.isin(inputs.item_ids)
    ]
    if absent := set(config["entries"]) - set(own.model):
        raise SchemeError(f"neural entries with no input rows: {sorted(absent)}")
    folds = assign_folds(items, config["fold_seed"], config["folds"])
    if len(set(folds.values())) != config["folds"] or any(
        sum(f != fold for f in folds.values()) < config["inner_folds"]
        for fold in set(folds.values())
    ):
        raise SchemeError("too few neural items for the requested nested folds")
    files = [
        Path(config_path).resolve(),
        manifest_path,
        summary_path,
        items_path,
        FALLACIES_FILE,
        *sorted(scheme_dir.glob("*.yaml")),
    ]
    hashes = {str(path.resolve()): _digest(path) for path in files}
    report = {
        "status": "preflight_only",
        "items": len(items),
        "schemes": len(set(items.gold_scheme)),
        "entries": sorted(config["entries"]),
        "features": len(inputs.names),
        "feature_audit": inputs.audit,
        "classes": items.gold_fallacy.value_counts().to_dict(),
        "scheme_folder": scheme_dir.name,
        "planned_fits": len(seeds)
        * config["folds"]
        * (len(config["candidates"]) * config["inner_folds"] + 1),
        "maximum_parameters_per_candidate": [
            (len(inputs.names) + 1) * c["hidden"] + (c["hidden"] + 1) * items.gold_fallacy.nunique()
            for c in config["candidates"]
        ],
        "input_hashes": hashes,
    }
    return config, items, inputs, folds, report


def run(config_path, out_dir=None, check_only=False):
    config, items, inputs, folds, report = prepare(config_path)
    if check_only:
        return report
    if out_dir is None:
        raise SchemeError("neural evaluation requires an explicit new --out directory")
    try:
        import torch

        from ..aggregate.neural import NetworkConfig, neural_out_of_fold
    except ImportError as error:
        raise SchemeError("neural training requires the optional neural dependencies") from error
    from .canonical import score_fallacies

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir()
    candidates = [NetworkConfig(**c) for c in config["candidates"]]
    source_paths = [
        *sorted((REPO_ROOT / "src" / "argfallacy").rglob("*.py")),
        REPO_ROOT / "pyproject.toml",
    ]
    manifest = {
        **report,
        "status": "running",
        "configuration": config,
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_hashes": {str(p.relative_to(REPO_ROOT)): _digest(p) for p in source_paths},
        "condition": "reference_scheme",
        "scope": "exploratory_point_metrics",
    }
    _json(out / "manifest.json", manifest)
    _json(out / "features.json", list(inputs.names))
    pd.DataFrame({"item_id": inputs.item_ids, "fold": [folds[i] for i in inputs.item_ids]}).to_csv(
        out / "folds.csv", index=False
    )
    targets = dict(zip(items.item_id, items.gold_fallacy, strict=True))
    previous_threads = torch.get_num_threads()
    rows, choices, metrics = [], [], []
    try:
        torch.set_num_threads(config["threads"])
        for seed in config["seeds"]:
            predictions, selected, _ = neural_out_of_fold(
                inputs,
                targets,
                folds,
                candidates,
                seed,
                config["fold_seed"],
                config["inner_folds"],
                checkpoints,
                lambda msg: print(msg, flush=True),
            )
            choices.extend(selected)
            system = f"N1.seed{seed}"
            for item in inputs.item_ids:
                pred = predictions[item]
                rows.append(
                    {
                        "system": system,
                        "item_id": item,
                        "fold": folds[item],
                        "gold": targets[item],
                        "verdict": pred.verdict,
                        "confidence": pred.confidence,
                        "abstention": pred.abstention,
                        "distribution": json.dumps(pred.distribution, sort_keys=True),
                        "explanation": pred.explanation,
                    }
                )
            score = score_fallacies(
                pd.DataFrame(
                    {
                        "item_id": inputs.item_ids,
                        "predicted": [predictions[i].verdict for i in inputs.item_ids],
                    }
                ),
                items,
            )
            metrics.append({"system": system, **score.summary()})
        pd.DataFrame(rows).to_csv(out / "verdicts.csv", index=False)
        pd.DataFrame(choices).to_csv(out / "choices.csv", index=False)
        pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
        _json(out / "candidates.json", [asdict(c) for c in candidates])
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.set_num_threads(previous_threads)
        _json(out / "manifest.json", manifest)
    return out
