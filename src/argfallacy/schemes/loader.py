"""Loading and structural validation of argumentation-scheme decision diagrams.

The diagrams live in ``schemes/*.yaml`` as data, not as code, so that a single
generic interpreter can execute any of them and so that the encoding itself is a
citable artefact.

IMPORTANT — YAML 1.1 gotcha
---------------------------
PyYAML implements YAML 1.1, in which the bare tokens ``yes``, ``no``, ``on`` and
``off`` parse as booleans.  Every answer key in the scheme files is therefore
*quoted*.  :func:`load_scheme` asserts this: an unquoted key surfaces as a Python
``bool`` and raises, rather than silently producing a graph keyed on ``True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from ..paths import FALLACIES_FILE, LABELS_DIR, SCHEMES_DIR


class SchemeError(ValueError):
    """Raised when a scheme file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class CriticalQuestion:
    id: str
    text: str
    answer_space: tuple[str, ...]
    flags: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class Edge:
    """One outgoing arc of a decision node."""

    answer: str
    target: str
    is_terminal: bool
    reconvergence: bool = False


@dataclass
class Scheme:
    scheme_id: str
    name: str
    schema: str
    identification_question: str
    variables: dict[str, str]
    cqs: dict[str, CriticalQuestion]
    cq_order: list[str]
    start: str
    nodes: dict[str, dict[str, Edge]]
    flags: tuple[str, ...] = ()
    note: str | None = None

    # -- basic accessors ---------------------------------------------------
    def answer_space(self, cq_id: str) -> tuple[str, ...]:
        if cq_id not in self.cqs:
            raise SchemeError(f"{self.scheme_id}: unknown CQ {cq_id!r}")
        return self.cqs[cq_id].answer_space

    @property
    def terminals(self) -> set[str]:
        return {e.target for edges in self.nodes.values()
                for e in edges.values() if e.is_terminal}

    @property
    def has_good_argumentation(self) -> bool:
        return "good_argumentation" in self.terminals

    def in_degree(self) -> dict[str, int]:
        """How many arcs enter each node/terminal.  >1 means reconvergence."""
        counts: dict[str, int] = {}
        for edges in self.nodes.values():
            for e in edges.values():
                counts[e.target] = counts.get(e.target, 0) + 1
        return counts

    def max_depth(self) -> int:
        """Number of decision nodes on the longest path (the graph is a DAG)."""
        memo: dict[str, int] = {}

        def depth(node: str) -> int:
            if node in memo:
                return memo[node]
            best = 1
            for e in self.nodes[node].values():
                if not e.is_terminal:
                    best = max(best, 1 + depth(e.target))
            memo[node] = best
            return best

        return depth(self.start)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def load_scheme(path: str | Path) -> Scheme:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    default_space = tuple(raw.get("default_answer_space", ["yes", "no"]))
    if any(isinstance(a, bool) for a in default_space):
        raise SchemeError(f"{path.name}: quote the yes/no tokens (YAML 1.1 booleans)")

    cqs: dict[str, CriticalQuestion] = {}
    order: list[str] = []
    for entry in raw["critical_questions"]:
        cq = CriticalQuestion(
            id=entry["id"],
            text=" ".join(entry["text"].split()),
            answer_space=tuple(entry.get("answer_space", default_space)),
            flags=_as_tuple(entry.get("flags")),
            note=entry.get("note"),
        )
        cqs[cq.id] = cq
        order.append(cq.id)

    nodes: dict[str, dict[str, Edge]] = {}
    for node_id, arcs in raw["graph"]["nodes"].items():
        if isinstance(node_id, bool):
            raise SchemeError(f"{path.name}: node id parsed as bool — quote it")
        edges: dict[str, Edge] = {}
        for answer, spec in arcs.items():
            if isinstance(answer, bool):
                raise SchemeError(
                    f"{path.name}: answer key of node {node_id} parsed as a YAML 1.1 "
                    f"boolean ({answer!r}); write it as \"yes\" / \"no\""
                )
            if "terminal" in spec:
                target, is_terminal = spec["terminal"], True
            elif "node" in spec:
                target, is_terminal = spec["node"], False
            else:
                raise SchemeError(f"{path.name}: arc {node_id}/{answer} has no target")
            edges[answer] = Edge(
                answer=answer,
                target=target,
                is_terminal=is_terminal,
                reconvergence=bool(spec.get("reconvergence", False)),
            )
        nodes[node_id] = edges

    scheme = Scheme(
        scheme_id=raw["scheme_id"],
        name=raw["name"],
        schema=raw["schema"].strip(),
        identification_question=" ".join(raw["identification_question"].split()),
        variables=raw.get("variables", {}),
        cqs=cqs,
        cq_order=order,
        start=raw["graph"]["start"],
        nodes=nodes,
        flags=_as_tuple(raw.get("scheme_flags")),
        note=raw.get("scheme_note"),
    )
    _check_integrity(scheme, path.name)
    return scheme


def _check_integrity(scheme: Scheme, filename: str) -> None:
    problems: list[str] = []

    if scheme.start not in scheme.nodes:
        problems.append(f"start node {scheme.start!r} is not defined")

    for node_id, edges in scheme.nodes.items():
        if node_id not in scheme.cqs:
            problems.append(f"node {node_id!r} has no matching critical question")
            continue
        space = set(scheme.cqs[node_id].answer_space)
        drawn = set(edges)
        if drawn != space:
            problems.append(
                f"node {node_id}: drawn arcs {sorted(drawn)} != answer space {sorted(space)}"
            )
        for e in edges.values():
            if not e.is_terminal and e.target not in scheme.nodes:
                problems.append(f"arc {node_id}/{e.answer} points to unknown node {e.target!r}")

    # every CQ that is not a node is dead weight; every node must be reachable
    reachable: set[str] = set()
    stack = [scheme.start]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for e in scheme.nodes[node].values():
            if not e.is_terminal:
                stack.append(e.target)
    for node_id in scheme.nodes:
        if node_id not in reachable:
            problems.append(f"node {node_id!r} is unreachable from {scheme.start!r}")
    for cq_id in scheme.cqs:
        if cq_id not in scheme.nodes:
            problems.append(f"CQ {cq_id!r} is listed but never asked in the diagram")

    # the graph must be acyclic: every traversal, path enumeration and depth
    # computation assumes it, and a back-arc would otherwise surface as a
    # RecursionError somewhere far from the file that caused it
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {n: WHITE for n in scheme.nodes}

    def visit(node: str, trail: list[str]) -> None:
        colour[node] = GREY
        for e in scheme.nodes[node].values():
            if e.is_terminal or e.target not in colour:
                continue
            if colour[e.target] == GREY:
                cycle = trail + [node, e.target]
                problems.append("cycle in the graph: " + " -> ".join(cycle))
            elif colour[e.target] == WHITE:
                visit(e.target, trail + [node])
        colour[node] = BLACK

    if scheme.start in colour:
        visit(scheme.start, [])

    if problems:
        raise SchemeError(f"{filename}:\n  - " + "\n  - ".join(problems))


def load_all(directory: str | Path = SCHEMES_DIR) -> dict[str, Scheme]:
    directory = Path(directory)
    schemes = {}
    for path in sorted(directory.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        scheme = load_scheme(path)
        schemes[scheme.scheme_id] = scheme
    return schemes


def load_vocabulary(
    directory: str | Path = LABELS_DIR,
    check_schemes: bool = True,
) -> dict[str, Any]:
    """Read the fallacy vocabulary and check the constraints it declares.

    ``check_schemes`` also cross-checks the ``schemes`` list of every terminal
    against the diagrams in ``schemes/``.  It is off on the hot path (a single
    label lookup) and on everywhere else, since it is the only check that can
    catch a diagram and the vocabulary drifting apart.
    """
    directory = Path(directory)
    path = directory / FALLACIES_FILE.name
    vocab = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_vocabulary(vocab, path.name)
    if check_schemes:
        check_vocabulary_against_schemes(vocab, load_all(), path.name)
    return vocab


def _check_vocabulary(vocab: dict[str, Any], filename: str) -> None:
    """Internal consistency of the vocabulary file, without reading the diagrams."""
    problems: list[str] = []

    terminals = set(vocab["terminals"])
    coarse = set(vocab.get("coarse_labels", {}))
    out_of_scope = set(vocab.get("out_of_scope", {}))
    families = set(vocab.get("families", {}))
    known = terminals | coarse | families | out_of_scope

    # normalize_label tries the four in a fixed order, so an id in two of them
    # would resolve by position rather than by meaning
    groups = {"terminals": terminals, "coarse_labels": coarse,
              "families": families, "out_of_scope": out_of_scope}
    for name, ids in groups.items():
        for other, other_ids in groups.items():
            if name < other and (clash := ids & other_ids):
                problems.append(f"id(s) {sorted(clash)} declared in both {name} and {other}")

    # an invalid output must be exactly one string, and must not also be an idk marker
    seen_raw: set[str] = set()
    idk = {" ".join(str(m).strip().lower().split()) for m in vocab.get("idk_markers", [])}
    for raw in vocab.get("invalid_outputs", {}):
        key = " ".join(str(raw).strip().lower().split())
        if key in seen_raw:
            problems.append(f"invalid output {raw!r} is listed twice")
        if key in idk:
            problems.append(f"{raw!r} is both an invalid output and an idk marker")
        seen_raw.add(key)

    # an alias must not point at two ids once the light normalization is applied
    seen: dict[str, str] = {}
    for raw, target in vocab.get("aliases", {}).items():
        key = " ".join(str(raw).strip().lower().split())
        if key in seen and seen[key] != target:
            problems.append(f"alias {key!r} points at both {seen[key]!r} and {target!r}")
        seen[key] = target
        if target not in known:
            problems.append(f"alias {raw!r} points at unknown id {target!r}")

    for cid, spec in vocab.get("coarse_labels", {}).items():
        for tid in spec.get("refines_to", []):
            if tid not in terminals:
                problems.append(f"coarse label {cid!r} refines to unknown terminal {tid!r}")

    for tid, spec in vocab["terminals"].items():
        fam = spec.get("family")
        if fam is not None and fam not in families:
            problems.append(f"terminal {tid!r} declares unknown family {fam!r}")

    for space, mapping in vocab.get("spaces", {}).items():
        for src, dst in mapping.items():
            if src not in terminals:
                problems.append(f"space {space!r} maps unknown terminal {src!r}")
            if dst not in terminals and dst not in families:
                problems.append(f"space {space!r} maps {src!r} to unknown id {dst!r}")

    if problems:
        raise SchemeError(f"{filename}:\n  - " + "\n  - ".join(problems))


def check_vocabulary_against_schemes(
    vocab: dict[str, Any],
    schemes: dict[str, Scheme],
    filename: str = "fallacies.yaml",
) -> None:
    """Terminals and their ``schemes`` list must match what the diagrams produce."""
    problems: list[str] = []

    produced: dict[str, set[str]] = {}
    for sid, scheme in schemes.items():
        for terminal in scheme.terminals:
            produced.setdefault(terminal, set()).add(sid)

    declared = set(vocab["terminals"])
    if declared != set(produced):
        for tid in sorted(declared - set(produced)):
            problems.append(f"terminal {tid!r} is in the vocabulary but no diagram reaches it")
        for tid in sorted(set(produced) - declared):
            problems.append(f"terminal {tid!r} is reached by a diagram but not in the vocabulary")

    for tid, spec in vocab["terminals"].items():
        if "schemes" not in spec:
            problems.append(f"terminal {tid!r} does not declare which schemes produce it")
            continue
        listed, actual = set(spec["schemes"]), produced.get(tid, set())
        if listed != actual:
            problems.append(
                f"terminal {tid!r}: declared schemes {sorted(listed)} != actual {sorted(actual)}"
            )

    if problems:
        raise SchemeError(f"{filename}:\n  - " + "\n  - ".join(problems))


INVALID = "invalid"
"""Id of an output that is not a label at all.  Wrong in every space, never dropped."""


@dataclass(frozen=True)
class Label:
    """A gold label from the annotation sheets, resolved against the vocabulary."""

    id: str
    kind: Literal["terminal", "coarse", "family", "out_of_scope", "idk", "invalid"]
    accepts: frozenset[str] = frozenset()  # terminals that satisfy this label

    def __str__(self) -> str:  # pragma: no cover
        return self.id


def normalize_label(raw_label: str, vocabulary: dict[str, Any] | None = None) -> Label:
    """Map a raw label from the annotation sheets onto the vocabulary.

    Six kinds come out of here, and the difference between the last three is the
    whole point of the function:

    ``terminal``      one of the 25 verdicts a diagram can reach.
    ``coarse``        names a set of terminals (``false_cause``).  Kept coarse
                      rather than refined into a guess: it resolves to the set
                      of terminals that satisfy it, and gives partial credit.
    ``family``        names a family (``ad_hominem``).  It accepts *no*
                      terminal, so in the fine space it can never be right; in a
                      collapsed space the family is itself a class.  Not a
                      coarse label: it does not refine and gives no credit.
    ``out_of_scope``  a real label no diagram produces.
    ``idk``           the annotator said they could not tell.
    ``invalid``       the model did not emit a label at all.  Wrong in every
                      space, and never a reason to drop the row.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    key = " ".join(str(raw_label).strip().lower().split())

    if key in {m.lower() for m in vocab.get("invalid_outputs", {})}:
        return Label(INVALID, "invalid")
    if key in {m.lower() for m in vocab.get("idk_markers", [])}:
        return Label("idk", "idk")

    key = vocab.get("aliases", {}).get(key, key)
    snake = key.replace(" ", "_").replace("(", "").replace(")", "")

    if snake in vocab["terminals"]:
        return Label(snake, "terminal", frozenset({snake}))
    if snake in vocab.get("coarse_labels", {}):
        return Label(snake, "coarse",
                     frozenset(vocab["coarse_labels"][snake]["refines_to"]))
    if snake in vocab.get("families", {}):
        return Label(snake, "family")
    if snake in vocab.get("out_of_scope", {}):
        return Label(snake, "out_of_scope")
    raise SchemeError(f"label {raw_label!r} is not in the vocabulary")


def family_members(family_id: str, vocabulary: dict[str, Any] | None = None) -> frozenset[str]:
    """The terminals a family groups, read off the ``family`` key of each terminal."""
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    if family_id not in vocab.get("families", {}):
        raise SchemeError(f"{family_id!r} is not a family")
    return frozenset(tid for tid, spec in vocab["terminals"].items()
                     if spec.get("family") == family_id)


@dataclass(frozen=True)
class Resolution:
    """A gold label checked against the scheme the item was annotated under."""

    label: Label
    status: Literal["exact", "coarse", "scheme_mismatch", "out_of_scope", "idk"]
    accepts: frozenset[str] = frozenset()
    reassign_to: str | None = None


def resolve_for_scheme(
    raw_label: str,
    scheme: Scheme,
    all_schemes: dict[str, Scheme] | None = None,
    vocabulary: dict[str, Any] | None = None,
) -> Resolution:
    """Resolve a gold label against the scheme it was filed under.

    Order of preference, least invasive first:

    1. the label is a terminal of this scheme -> ``exact``;
    2. it is a coarse label, or the name of a family, with at least one member
       this scheme produces -> ``coarse``, satisfied by any of them;
    3. it is a terminal of exactly one *other* scheme -> ``scheme_mismatch``,
       with the reassignment proposed but not applied;
    4. it is out of scope, or an IDK marker.

    **The rule, and it is asymmetric on purpose: a coarse gold label gives
    partial credit, a vague prediction gives none.**

    A gold label that names a family is an annotator being less precise than the
    diagram — the fallacy is an ad hominem, they did not say which — so any
    member the scheme produces satisfies it.  A *prediction* that names a family
    is a model being vague, and it earns nothing: the family id is not a
    terminal, so ``label_matches`` cannot find it in any gold's ``accepts``, and
    in the fine label space it is not a class at all.

    That asymmetry is why ``normalize_label`` leaves ``Label.accepts`` empty for
    a family and the credit is granted only here: it is a property of being the
    gold under a particular scheme, not a property of the string.  As with
    ``false_cause``, the share of gold that is coarse has to be reported
    separately — a metric over mixed granularity is not comparable with one over
    exact labels.
    """
    vocab = vocabulary or load_vocabulary(check_schemes=False)
    label = normalize_label(raw_label, vocab)

    if label.kind == "idk":
        return Resolution(label, "idk")
    if label.kind == "out_of_scope":
        return Resolution(label, "out_of_scope")

    # a family carries no accepts of its own; its members are read from the vocabulary
    accepts = family_members(label.id, vocab) if label.kind == "family" else label.accepts

    hits = accepts & scheme.terminals
    if label.kind == "terminal" and hits:
        return Resolution(label, "exact", hits)
    if label.kind in ("coarse", "family") and hits:
        return Resolution(label, "coarse", hits)

    producers = []
    if all_schemes:
        producers = sorted(sid for sid, sc in all_schemes.items()
                           if accepts & sc.terminals)
    return Resolution(label, "scheme_mismatch", frozenset(),
                      producers[0] if len(producers) == 1 else None)


def label_matches(predicted_terminal: str, gold: Resolution) -> bool:
    """Whether a predicted terminal satisfies the gold label.

    Coarse gold labels give partial credit: any refinement counts.  Report the
    coarse share separately, since a metric computed over mixed granularity is
    not comparable with one computed over exact labels only.
    """
    return predicted_terminal in gold.accepts
