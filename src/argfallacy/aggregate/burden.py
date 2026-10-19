"""The burden of proof of each critical question, and where it sends the mass outside the arcs.

The table of a version of the diagrams sits in ``schemes/burden/<folder>.yaml``,
outside the folder of the diagrams, so the hash of the schemes in the manifests
does not change.  It gives one type per CQ: P (premise), E (exception), G (gate),
D (two fallacies); the arcs follow from the diagram.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from ..paths import SCHEMES_ROOT
from ..schemes import Scheme, SchemeError
from .diagram import good_terminal, pass_arc, reaches_good

PREMISE, EXCEPTION, GATE, TWO_FALLACIES = "P", "E", "G", "D"
TYPES = (PREMISE, EXCEPTION, GATE, TWO_FALLACIES)

ABSTAIN, SPLIT = "abstain", "split"
"""Two routes that are not an arc: the mass leaves the diagram, or halves on the arcs."""

POLICIES = ("burden", "stop", "branch", "charitable")
"""``burden`` follows the table; the other three apply one rule to every CQ."""

BURDEN_DIR = SCHEMES_ROOT / "burden"


def load_burden(
    folder: str, schemes: Mapping[str, Scheme], root: str | Path = BURDEN_DIR
) -> dict[str, dict[str, str]]:
    """The types of the CQs of ``schemes/<folder>/``, checked against the diagrams.

    Every CQ of every scheme has one type and the table names no other CQ; a gate
    has both arcs reaching ``good_argumentation``; two fallacies have both arcs on
    a fallacy.
    """
    path = Path(root) / f"{folder}.yaml"
    if not path.is_file():
        raise SchemeError(f"the diagrams of schemes/{folder}/ have no burden table: {path} "
                          f"does not exist, so A2 and A4 cannot route the answers outside "
                          f"the arcs")
    table = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    good = good_terminal()
    problems = [f"{name}: not a scheme of schemes/{folder}/" for name in sorted(set(table)
                                                                            - set(schemes))]
    for scheme_id, scheme in sorted(schemes.items()):
        types = table.get(scheme_id) or {}
        problems += [f"{scheme_id} {cq}: no type" for cq in scheme.nodes if cq not in types]
        for cq, kind in types.items():
            if cq not in scheme.nodes:
                problems.append(f"{scheme_id} {cq}: not a CQ of the diagram")
            elif kind not in TYPES:
                problems.append(f"{scheme_id} {cq}: type {kind!r}, not one of {list(TYPES)}")
            elif kind == GATE and not all(reaches_good(scheme, cq).values()):
                problems.append(f"{scheme_id} {cq}: a gate, but not both arcs reach {good}")
            elif kind == TWO_FALLACIES and not all(
                    edge.is_terminal and edge.target != good
                    for edge in scheme.nodes[cq].values()):
                problems.append(f"{scheme_id} {cq}: two fallacies, but not both arcs end on "
                                f"a fallacy")
    if problems:
        raise SchemeError(f"{path}: " + "; ".join(problems))
    return {scheme_id: dict(table[scheme_id]) for scheme_id in sorted(schemes)}


def routes(scheme: Scheme, types: Mapping[str, str], policy: str = "burden") -> dict[str, str]:
    """Where the mass outside the arcs of each node goes: an arc, ABSTAIN or SPLIT.

    With ``burden``: a premise abstains, an exception takes the pass arc, a gate the
    arc to its follow-up questions, two fallacies split.  ``stop`` abstains at every
    CQ, ``branch`` splits at every CQ, ``charitable`` takes the pass arc at every CQ
    and splits at the node with two fallacies.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {list(POLICIES)}")
    out = {}
    for node in scheme.nodes:
        kind = types[node]
        if policy == "stop":
            out[node] = ABSTAIN
        elif policy == "branch" or kind == TWO_FALLACIES:
            out[node] = SPLIT
        elif policy == "charitable" or kind == EXCEPTION:
            out[node] = pass_arc(scheme, node)
        elif kind == PREMISE:
            out[node] = ABSTAIN
        else:
            standing = pass_arc(scheme, node)
            (out[node],) = [answer for answer in scheme.nodes[node] if answer != standing]
    return out
