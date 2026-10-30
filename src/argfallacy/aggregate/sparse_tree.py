"""Matched Boolean trees with an explicit class-balanced misclassification cost."""

from __future__ import annotations

import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

GRID = tuple((depth, coefficient) for depth in (2, 3, 4) for coefficient in (5, 2))


def predicates(scheme):
    return [
        {
            "cq": cq,
            "kind": kind,
            "operator": ">",
            "threshold": threshold,
            "cq_text": scheme.cqs[cq].text,
            "cq_note": scheme.cqs[cq].note,
            "first_arc": scheme.answer_space(cq)[0],
            "version": "2.0",
        }
        for cq in scheme.cq_order
        for kind, thresholds in (("p_first", (0.25, 0.50, 0.75)), ("outside", (0.10, 0.50)))
        for threshold in thresholds
    ]


def encode(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("expected paired CQ probability/outside columns")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("CQ probabilities must be finite and within [0,1]")
    columns = [
        values[:, 2 * q + offset] > threshold
        for q in range(values.shape[1] // 2)
        for offset, thresholds in ((0, (0.25, 0.50, 0.75)), (1, (0.10, 0.50)))
        for threshold in thresholds
    ]
    return np.column_stack(columns) if columns else np.empty((len(values), 0), bool)


def constant_mask(x):
    return np.any(x != x[0], axis=0) if len(x) else np.zeros(x.shape[1], bool)


def objective(y, coefficient):
    classes, counts = np.unique(np.asarray(y), return_counts=True)
    n, k = len(y), len(classes)
    if not n:
        return {
            "classes": [],
            "counts": [],
            "matrix": [],
            "lambda": None,
            "nominal_lambda": None,
            "n": 0,
        }
    matrix = np.zeros((k, k))
    if k > 1:
        matrix[:] = 1 / (k * counts)
        np.fill_diagonal(matrix, 0)
    return {
        "classes": classes.tolist(),
        "counts": counts.tolist(),
        "n": n,
        "matrix": matrix.astype(np.float32).astype(float).tolist(),
        "lambda": float(np.float32(coefficient / n)),
        "nominal_lambda": coefficient / n,
    }


def sample_weights(y, cost):
    classes = cost["classes"]
    if len(classes) == 1:
        return np.ones(len(y))
    matrix = np.asarray(cost["matrix"])
    index = {c: j for j, c in enumerate(classes)}
    return np.array([cost["n"] * matrix[(index[c] + 1) % len(classes), index[c]] for c in y])


def root_leaf(y, cost):
    if not len(y):
        return {"verdict": None}
    matrix = np.asarray(cost["matrix"])
    counts = np.asarray(cost["counts"])
    losses = matrix @ counts
    return {"verdict": cost["classes"][int(np.argmin(losses))]}


def analytic_tree(x, y, cost):
    if not len(y):
        return root_leaf(y, cost), "empty_training"
    if len(cost["classes"]) == 1:
        return root_leaf(y, cost), "single_class"
    if not x.shape[1]:
        return root_leaf(y, cost), "no_features"
    matrix, counts = np.asarray(cost["matrix"]), np.asarray(cost["counts"])
    masses = [counts[j] * matrix[(j + 1) % len(counts), j] for j in range(len(counts))]
    if cost["lambda"] >= sorted(masses, reverse=True)[1]:
        return root_leaf(y, cost), "analytic_root"
    return None, None


def replay(tree, x):
    answers, paths = [], []
    for row in np.asarray(x, dtype=bool):
        node, path = tree, []
        while "verdict" not in node:
            value = bool(row[node["column"]])
            path.append({"column": node["column"], "value": value})
            node = node["true"] if value else node["false"]
        answers.append(node["verdict"])
        paths.append(path)
    return answers, paths


def annotate(tree, x, y, cost):
    """Attach training support and calculate the common effective objective."""
    classes = cost["classes"]
    matrix = np.asarray(cost["matrix"])
    index = {c: j for j, c in enumerate(classes)}
    y = np.asarray(y)
    w = sample_weights(y, cost) if len(y) else np.array([])

    def visit(node, ids):
        node["raw_support"] = len(ids)
        node["class_counts"] = {c: int(np.sum(y[ids] == c)) for c in classes}
        node["weighted_counts"] = {c: float(w[ids][y[ids] == c].sum()) for c in classes}
        if "verdict" in node:
            loss = sum(matrix[index[node["verdict"]], index[y[i]]] for i in ids) if len(ids) else 0
            node["training_risk"] = float(loss)
            return float(loss), 1, 0
        mask = x[ids, node["column"]]
        a, b = visit(node["true"], ids[mask]), visit(node["false"], ids[~mask])
        return a[0] + b[0], a[1] + b[1], 1 + max(a[2], b[2])

    risk, leaves, depth = visit(tree, np.arange(len(y)))
    predicted, _ = replay(tree, x)
    if len(y):
        nominal = np.mean([np.mean(np.asarray(predicted, object)[y == c] != c) for c in classes])
    else:
        nominal = None
    return {
        "risk": risk,
        "nominal_balanced_risk": nominal,
        "leaves": leaves,
        "depth": depth,
        "objective": risk + cost["lambda"] * leaves if len(y) else None,
    }


def fit_cart(x, y, depth, coefficient, seed, evaluation_x=None):
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y)
    cost = objective(y, coefficient)
    tree, status = analytic_tree(x, y, cost)
    unpruned, native_predictions = None, None
    if tree is None:
        weights = sample_weights(y, cost)
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=seed,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0,
            ccp_alpha=0,
        ).fit(x, y, sample_weight=weights)
        native = model.tree_
        matrix = np.asarray(cost["matrix"])

        def prune(node, ids):
            counts = np.array([np.sum(y[ids] == c) for c in cost["classes"]])
            losses = matrix @ counts
            cls = int(np.argmin(losses))
            leaf = {"verdict": cost["classes"][cls], "native_node": int(node)}
            stop = float(losses[cls]) + cost["lambda"]
            if native.children_left[node] == -1:
                return leaf, stop
            column = int(native.feature[node])
            mask = x[ids, column].astype(bool)
            # sklearn left means false (<= 0.5); JSON true/false is explicit.
            true, a = prune(native.children_right[node], ids[mask])
            false, b = prune(native.children_left[node], ids[~mask])
            if stop <= a + b + 1e-12:
                return leaf, stop
            return {"column": column, "true": true, "false": false, "native_node": int(node)}, a + b

        tree, _ = prune(0, np.arange(len(y)))
        unpruned = {"depth": model.get_depth(), "leaves": model.get_n_leaves()}
        if evaluation_x is not None:

            def materialize(node):
                i = node["native_node"]
                if "verdict" in node:
                    native.children_left[i] = native.children_right[i] = -1
                    native.feature[i] = native.threshold[i] = -2
                    native.value[i, 0, :] = 0
                    native.value[i, 0, cost["classes"].index(node["verdict"])] = 1
                else:
                    materialize(node["true"])
                    materialize(node["false"])

            materialize(tree)
            native_predictions = model.predict(evaluation_x).tolist()
        status = "greedy_pruned"
    result = {
        "tree": tree,
        "cost": cost,
        "status": status,
        "valid": bool(len(y)),
        "certified": False,
        "unpruned": unpruned,
        **annotate(tree, x, y, cost),
    }
    if evaluation_x is not None:
        exported, _ = replay(tree, evaluation_x)
        if native_predictions is None:
            native_predictions = [tree["verdict"]] * len(evaluation_x)
        if exported != native_predictions:
            raise ValueError("pruned sklearn held-out replay mismatch")
        result.update(evaluation_predictions=native_predictions, evaluation_replay_verified=True)
    return result


def fit_gosdt(x, y, depth, coefficient, seconds, evaluation_x=None):
    """Optional native dependency, imported only for a nontrivial solver fit."""
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y)
    cost = objective(y, coefficient)
    tree, status = analytic_tree(x, y, cost)
    native = None
    error, native_predictions = None, None
    if tree is None:
        from gosdt import GOSDTClassifier

        model = GOSDTClassifier(
            regularization=cost["lambda"],
            depth_budget=depth,
            time_limit=seconds,
            balance=False,
            similar_support=False,
            worker_limit=1,
            model_limit=1,
            uncertainty_tolerance=0,
            upperbound_guess=None,
            rule_list=False,
            non_binary=False,
        )
        try:
            model.fit(x, y, cost_matrix=np.asarray(cost["matrix"]))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if hasattr(model, "result_"):
            r = model.result_
            native = {
                "status": str(r.status),
                "model": r.model,
                "lower": float(r.lowerbound),
                "upper": float(r.upperbound),
                "model_loss": float(r.model_loss),
                "seconds": float(r.time),
            }
        if native and getattr(model, "trees_", []):

            def convert(node):
                if "prediction" in node:
                    return {"verdict": cost["classes"][node["prediction"]]}
                return {
                    "column": node["feature"],
                    "true": convert(node["true"]),
                    "false": convert(node["false"]),
                }

            tree = convert(json.loads(native["model"])[0])
            if replay(tree, x)[0] != model.predict(x).tolist():
                raise ValueError("GOSDT exported replay mismatch")
            if evaluation_x is not None:
                native_predictions = model.predict(evaluation_x).tolist()
                if replay(tree, evaluation_x)[0] != native_predictions:
                    raise ValueError("GOSDT held-out replay mismatch")
            if float(model.config_.regularization) != cost["lambda"]:
                raise ValueError("native regularization differs")
            status = native["status"]
    if tree is None:
        tree, status = root_leaf(y, cost), "solver_failed"
    measured = annotate(tree, x, y, cost)
    valid = bool(len(y)) and status != "solver_failed"
    certified = status in {"single_class", "no_features", "analytic_root"}
    if native and valid:
        j, lb, ub = measured["objective"], native["lower"], native["upper"]
        if not np.isfinite([j, lb, ub]).all() or j < lb - 1e-6 or j > ub + 1e-6:
            valid = False
            error = "inconsistent native objective/bounds"
        native["incumbent_gap"] = max(0, j - lb)
        certified = valid and native["status"] == "Status.CONVERGED" and abs(j - lb) <= 1e-6
    if measured["depth"] > depth:
        raise ValueError("exported tree exceeds depth budget")
    result = {
        "tree": tree,
        "cost": cost,
        "status": status,
        "valid": valid,
        "certified": certified,
        "native": native,
        "error": error,
        **measured,
    }
    if evaluation_x is not None:
        exported, _ = replay(tree, evaluation_x)
        if native_predictions is None:
            native_predictions = [tree["verdict"]] * len(evaluation_x)
        if valid and native_predictions != exported:
            raise ValueError("GOSDT analytic held-out replay mismatch")
        result.update(
            evaluation_predictions=native_predictions,
            evaluation_replay_verified=valid and native_predictions == exported,
        )
    return result
