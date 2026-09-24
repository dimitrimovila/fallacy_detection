"""The scorer: predictions and gold labels in, metrics out.

Two modes, kept apart.  The canonical mode (:mod:`.canonical`) applies the rules
of the project and produces every number of the thesis.  The compat mode
(:mod:`.compat`) counts like the scoring code of the prior runs, only to show
that the counting reproduces their files.
"""

from .baselines import majority_by
from .canonical import (
    NO_VERDICT,
    SCHEME_SPACE,
    Score,
    gold_scheme_mismatches,
    predicted_fallacy_class,
    predicted_scheme_class,
    score_fallacies,
    score_schemes,
)
from .compat import compat_classwise, compat_metrics, compat_per_source
from .prior import PriorRun, load_test_set, prior_items, read_prior_runs, score_prior

__all__ = [
    "Score", "score_fallacies", "score_schemes", "gold_scheme_mismatches",
    "predicted_fallacy_class", "predicted_scheme_class", "NO_VERDICT", "SCHEME_SPACE",
    "majority_by",
    "compat_metrics", "compat_classwise", "compat_per_source",
    "PriorRun", "read_prior_runs", "load_test_set", "prior_items", "score_prior",
]
