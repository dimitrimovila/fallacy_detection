#!/usr/bin/env python3
"""Structural validation of the scheme diagrams.

Two layers:

1. **Integrity** — enforced by the loader on every import: arcs point somewhere
   real, every node is reachable, every listed CQ is actually asked, arcs cover
   the answer space exactly.

2. **Snapshot** — the structural facts below are frozen deliberately.  Changing a
   diagram is allowed; changing one *by accident* is not.  When a scheme is
   revised on purpose, update ``SNAPSHOT`` in the same commit.

Also prints the inverse index (verdict -> schemes) and the label/path ambiguity
report, both of which matter for how annotation records verdicts.
"""
import sys
from collections import defaultdict

from argfallacy.schemes import enumerate_paths, load_all, load_vocabulary

# scheme_id -> (n_cqs, depth, n_paths, has_good_argumentation)
SNAPSHOT = {
    "ad_hominem":           (6, 6,  7, False),
    "analogy":              (5, 4,  7, True),
    "cause_to_effect":      (6, 6,  9, True),
    "correlation_to_cause": (7, 7, 11, True),
    "example":              (6, 6,  8, True),
    "expert_opinion":       (6, 5,  7, True),
    "popular_opinion":      (6, 6,  7, True),
    "slippery_slope":       (6, 6,  8, True),
}


def check(schemes=None, vocab=None) -> list[str]:
    """The snapshot comparison alone, without the report.  Empty list means fine."""
    schemes = load_all() if schemes is None else schemes
    vocab = load_vocabulary() if vocab is None else vocab
    known = set(vocab["terminals"])
    failures: list[str] = []

    if set(schemes) != set(SNAPSHOT):
        failures.append(f"scheme set changed: {sorted(set(schemes) ^ set(SNAPSHOT))}")

    for sid in sorted(schemes):
        s = schemes[sid]
        actual = (len(s.cqs), s.max_depth(), len(enumerate_paths(s)),
                  s.has_good_argumentation)
        if sid in SNAPSHOT and actual != SNAPSHOT[sid]:
            failures.append(f"{sid}: structure {actual} != snapshot {SNAPSHOT[sid]}")
        unknown = s.terminals - known
        if unknown:
            failures.append(f"{sid}: terminals outside the vocabulary: {sorted(unknown)}")

    return failures


def main() -> int:
    schemes = load_all()
    vocab = load_vocabulary()
    failures = check(schemes, vocab)

    print(f"{len(schemes)} schemes loaded\n")
    print(f"{'scheme':<22}{'CQs':>5}{'depth':>7}{'paths':>7}"
          f"  {'reconv.node':<14}{'reconv.terminal':<20}{'good':>5}")
    print("-" * 82)

    for sid in sorted(schemes):
        s = schemes[sid]
        indeg = s.in_degree()
        rn = sorted(n for n, c in indeg.items() if c > 1 and n in s.nodes)
        rt = sorted(n for n, c in indeg.items() if c > 1 and n not in s.nodes)
        actual = (len(s.cqs), s.max_depth(), len(enumerate_paths(s)),
                  s.has_good_argumentation)
        print(f"{sid:<22}{actual[0]:>5}{actual[1]:>7}{actual[2]:>7}"
              f"  {','.join(rn) or '-':<14}{','.join(rt) or '-':<20}"
              f"{'yes' if actual[3] else 'NO':>5}")

    # ---- inverse index -------------------------------------------------------
    inverse = defaultdict(set)
    arcs_to = defaultdict(list)
    for sid, s in schemes.items():
        for t in s.terminals:
            inverse[t].add(sid)
        for node, edges in s.nodes.items():
            for a, e in edges.items():
                if e.is_terminal:
                    arcs_to[e.target].append(f"{sid}:{node}={a}")

    print("\nVerdict -> schemes")
    print("-" * 82)
    for t in sorted(inverse):
        if t != "good_argumentation":
            print(f"  {vocab['terminals'][t]['label']:<34} {', '.join(sorted(inverse[t]))}")
    unreached = set(vocab["terminals"]) - set(inverse)
    if unreached:
        print(f"\n  In the vocabulary but reached by no scheme: {sorted(unreached)}")

    # ---- label -> path ambiguity --------------------------------------------
    print("\nVerdicts reachable by more than one arc")
    print("-" * 82)
    for t, arcs in sorted(arcs_to.items(), key=lambda kv: -len(kv[1])):
        if len(arcs) > 1:
            print(f"  {vocab['terminals'][t]['label']:<26} {len(arcs)}  {', '.join(arcs)}")
    print("\n  => the terminal label alone does not identify the path;")
    print("     annotation must record (scheme, exit CQ) and the full answer vector.")

    # ---- open decisions ------------------------------------------------------
    print("\nDecisions recorded in the encoding")
    print("-" * 82)
    any_open = False
    for sid in sorted(schemes):
        s = schemes[sid]
        for f in s.flags:
            print(f"  [{sid}] {f}")
            any_open = True
        for cq_id in s.cq_order:
            note = s.cqs[cq_id].note
            if note:
                print(f"  [{sid}] {cq_id}: {' '.join(note.split())[:86]}…")
                any_open = True
    if not any_open:
        print("  none")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
