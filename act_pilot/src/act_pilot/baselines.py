"""Lexical baselines, a source-only baseline and the lexical predictability of an answer.

All of them are fitted on the outer training folds and predict the outer test fold, so
their out-of-fold predictions are comparable item by item with those of the questions.

* TF-IDF + CART: the depth is chosen by an inner stratified cross-validation on the outer
  training fold (balanced accuracy), with balanced class weights.
* TF-IDF + RuleFit: :class:`imodels.RuleFitClassifier` (MIT licence) on a dense TF-IDF
  matrix with at most ``max_features`` columns.
* Source only: each source predicts the majority class it has in the outer training fold.
  It is not a lexical baseline and does not enter the acceptance criterion; it measures how
  much of the boundary is explained by where the items come from.
* Lexical predictability: a TF-IDF + logistic regression predicts the binary *answer* of the
  LLM to a question (not the gold label); its out-of-fold AUC says how far the answer can be
  recovered from words alone. The most weighted terms come from a fit on all items.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import LexicalConfig
from .data import LabeledItem


@dataclass
class BaselineResult:
    name: str
    predictions: list[int]
    details: list[dict] = field(default_factory=list)
    description: str = ""


def _vectorizer(cfg: LexicalConfig, min_df: int | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, cfg.ngram_max),
        min_df=cfg.min_df if min_df is None else min_df,
        max_features=cfg.max_features,
        sublinear_tf=True,
    )


def _fit_vectorizer(cfg: LexicalConfig, texts: Sequence[str]):
    try:
        vec = _vectorizer(cfg)
        return vec, vec.fit_transform(texts)
    except ValueError:
        # Very small training sets may have no term in min_df documents.
        vec = _vectorizer(cfg, min_df=1)
        return vec, vec.fit_transform(texts)


def _folds(items: Sequence[LabeledItem]) -> list[int]:
    return sorted({i.outer_fold for i in items})


def _majority(labels: Sequence[int]) -> int:
    counts = Counter(labels)
    return 1 if counts[1] >= counts[0] else 0


def tfidf_cart(items: Sequence[LabeledItem], cfg: LexicalConfig) -> BaselineResult:
    """Out-of-fold predictions of TF-IDF + CART, with the depth tuned in the inner folds."""
    preds = np.zeros(len(items), dtype=int)
    details = []
    for fold in _folds(items):
        tr = [k for k, i in enumerate(items) if i.outer_fold != fold]
        te = [k for k, i in enumerate(items) if i.outer_fold == fold]
        x_tr = [items[k].text for k in tr]
        y_tr = [items[k].label for k in tr]
        n_splits = min(cfg.inner_folds, min(Counter(y_tr).values()))
        pipe = Pipeline(
            [
                ("tfidf", _vectorizer(cfg, min_df=1)),
                ("tree", DecisionTreeClassifier(class_weight="balanced",
                                                random_state=cfg.random_state)),
            ]
        )
        if n_splits >= 2 and len(set(y_tr)) == 2:
            search = GridSearchCV(
                pipe,
                {"tree__max_depth": list(cfg.cart_depths)},
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=cfg.random_state),
                scoring="balanced_accuracy",
            )
            search.fit(x_tr, y_tr)
            model, depth = search.best_estimator_, search.best_params_["tree__max_depth"]
        else:
            depth = cfg.cart_depths[0]
            pipe.set_params(tree__max_depth=depth)
            model = pipe.fit(x_tr, y_tr)
        preds[te] = model.predict([items[k].text for k in te])
        details.append({"outer_fold": fold, "max_depth": int(depth)})
    return BaselineResult("tfidf_cart", preds.tolist(), details,
                          "TF-IDF + CART, depth chosen by inner cross-validation")


def _rulefit(cfg: LexicalConfig):
    from imodels import RuleFitClassifier

    return RuleFitClassifier(
        max_rules=cfg.rulefit_max_rules,
        n_estimators=cfg.rulefit_n_estimators,
        cv=cfg.rulefit_cv,
        random_state=cfg.random_state,
    )


def tfidf_rulefit(items: Sequence[LabeledItem], cfg: LexicalConfig) -> BaselineResult:
    """Out-of-fold predictions of TF-IDF + RuleFit, and the top rules of a fit on all items."""
    preds = np.zeros(len(items), dtype=int)
    details = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        for fold in _folds(items):
            tr = [k for k, i in enumerate(items) if i.outer_fold != fold]
            te = [k for k, i in enumerate(items) if i.outer_fold == fold]
            y_tr = np.array([items[k].label for k in tr])
            vec, x_tr = _fit_vectorizer(cfg, [items[k].text for k in tr])
            if len(set(y_tr)) < 2:
                preds[te] = int(y_tr[0])
                continue
            model = _rulefit(cfg)
            names = [f"tfidf[{t}]" for t in vec.get_feature_names_out()]
            model.fit(x_tr.toarray(), y_tr, feature_names=names)
            x_te = vec.transform([items[k].text for k in te]).toarray()
            preds[te] = np.asarray(model.predict(x_te)).astype(int)
            details.append({"outer_fold": fold, "n_features": len(names)})
        description = "TF-IDF + RuleFit"
        try:
            vec, x_all = _fit_vectorizer(cfg, [i.text for i in items])
            model = _rulefit(cfg)
            names = [f"tfidf[{t}]" for t in vec.get_feature_names_out()]
            model.fit(x_all.toarray(), np.array([i.label for i in items]), feature_names=names)
            rules = model._get_rules()
            rules = rules[rules.coef != 0].sort_values("importance", ascending=False).head(5)
            top = "; ".join(f"{r.rule} ({r.coef:+.2f})" for r in rules.itertuples())
            description += f"; top rules on all items: {top}"
        except Exception as exc:  # the description is informative only
            description += f"; rules not available ({type(exc).__name__})"
    return BaselineResult("tfidf_rulefit", preds.tolist(), details, description)


def source_only(items: Sequence[LabeledItem]) -> BaselineResult:
    """Each source predicts its majority class in the outer training fold."""
    preds = np.zeros(len(items), dtype=int)
    details = []
    for fold in _folds(items):
        tr = [i for i in items if i.outer_fold != fold]
        overall = _majority([i.label for i in tr])
        by_source: dict[str, list[int]] = {}
        for i in tr:
            by_source.setdefault(i.source, []).append(i.label)
        rule = {s: _majority(v) for s, v in by_source.items()}
        for k, item in enumerate(items):
            if item.outer_fold == fold:
                preds[k] = rule.get(item.source, overall)
        details.append({"outer_fold": fold, "rule": rule})
    return BaselineResult("source_only", preds.tolist(), details,
                          "majority class of each source in the training folds")


@dataclass
class LexicalPredictability:
    auc: float
    positive_terms: list[str]
    negative_terms: list[str]
    yes_rate: float


def lexical_predictability(
    items: Sequence[LabeledItem], target: Sequence[int], cfg: LexicalConfig
) -> LexicalPredictability:
    """How well TF-IDF + logistic regression recovers ``target`` (an answer) from the text."""
    y = np.asarray(target, dtype=int)
    yes_rate = float(y.mean()) if len(y) else float("nan")
    if len(set(y.tolist())) < 2:
        return LexicalPredictability(float("nan"), [], [], yes_rate)
    scores = np.zeros(len(items))
    for fold in _folds(items):
        tr = [k for k, i in enumerate(items) if i.outer_fold != fold]
        te = [k for k, i in enumerate(items) if i.outer_fold == fold]
        y_tr = y[tr]
        if len(set(y_tr.tolist())) < 2:
            scores[te] = float(y_tr[0])
            continue
        vec, x_tr = _fit_vectorizer(cfg, [items[k].text for k in tr])
        lr = LogisticRegression(class_weight="balanced", max_iter=2000)
        lr.fit(x_tr, y_tr)
        scores[te] = lr.predict_proba(vec.transform([items[k].text for k in te]))[:, 1]
    auc = float(roc_auc_score(y, scores))
    vec, x_all = _fit_vectorizer(cfg, [i.text for i in items])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x_all, y)
    terms = vec.get_feature_names_out()
    order = np.argsort(lr.coef_[0])
    k = cfg.top_terms
    positive = [str(terms[j]) for j in order[::-1][:k]]
    negative = [str(terms[j]) for j in order[:k]]
    return LexicalPredictability(auc, positive, negative, yes_rate)
