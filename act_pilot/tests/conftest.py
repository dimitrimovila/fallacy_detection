from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from fakes import write_synthetic  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = PACKAGE_ROOT / "configs" / "pilot.yaml"
MANUAL = PACKAGE_ROOT / "manual_rewrites.yaml"
SCHEMES_V2 = PACKAGE_ROOT.parent / "schemes" / "v2"
# the two boundaries of configs/pilot.yaml
POPULAR = "popular_good_vs_ad_populum"
HOMINEM = "ad_hominem_poisoning_vs_abusive"


def make_config(tmp_path: Path, *, steps: int = 3, seeds=(0, 1), **act_overrides) -> Path:
    """A copy of configs/pilot.yaml pointing at synthetic data, with a short ACT loop."""
    items, folds = write_synthetic(tmp_path / "data")
    raw = yaml.safe_load(PILOT_CONFIG.read_text(encoding="utf-8"))
    raw["data"] = {"items": str(items), "folds": str(folds)}
    raw["manual_rewrites"] = str(MANUAL)
    raw["schemes_dir"] = str(SCHEMES_V2)
    raw["output_dir"] = str(tmp_path / "out")
    raw["content_validity"] = str(tmp_path / "content_validity.csv")
    raw["llm"]["max_workers"] = 4
    raw["llm"]["backoff_s"] = 0.0
    raw["act"].update({"steps": steps, "seeds": list(seeds), **act_overrides})
    raw["evaluation"]["bootstrap"] = 500
    raw["evaluation"]["lexical"]["rulefit_max_rules"] = 10
    raw["evaluation"]["lexical"]["rulefit_n_estimators"] = 20
    raw["evaluation"]["lexical"]["rulefit_cv"] = False
    raw["evaluation"]["lexical"]["max_features"] = 300
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return make_config(tmp_path)
