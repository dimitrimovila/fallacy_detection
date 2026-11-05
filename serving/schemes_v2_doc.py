#!/usr/bin/env python3
"""Write docs/schemes_v2.md, the readable description of the diagrams 2.0, from schemes/v2.

    python serving/schemes_v2_doc.py

The YAML files are the authoritative source: after any change to them, the document is
written again. It reads the files with PyYAML alone, without the package of the
repository, and writes the same bytes on every system (UTF-8, LF line endings).
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMES_DIR = REPO_ROOT / "schemes" / "v2"
OUTPUT = REPO_ROOT / "docs" / "schemes_v2.md"
ORDER = [
    "example", "analogy", "cause_to_effect", "correlation_to_cause", "ad_hominem",
    "expert_opinion", "popular_opinion", "slippery_slope",
]
INTRO = (
    "**Generated file. Do not edit by hand.** It is produced from `schemes/v2/*.yaml`, which "
    "are the authoritative source; after any change to those files, regenerate it. Version 1.1 "
    "(the published thesis of E. Bergamasco, chapter 3) is described in "
    "`schemes_and_diagrams.md`. The two versions differ in the wording of many questions and, "
    "for some schemes, in the structure of the diagram: always check which version an "
    "experiment uses.\n"
)


def scheme_lines(spec: dict, inverse: dict[str, set[str]], scheme: str) -> list[str]:
    """The section of one scheme; the terminals it reaches are added to ``inverse``."""
    lines = [f"\n## {spec['name']} (`{spec['scheme_id']}`, version {spec['version']})\n"]
    lines.append("**Schema**\n```\n" + spec["schema"].rstrip() + "\n```\n")
    lines.append(f"**Identification question**\n> {spec['identification_question'].strip()}\n")
    if spec.get("variables"):
        variables = "; ".join(f"`{k}` {v}" for k, v in spec["variables"].items())
        lines.append(f"**Variables.** {variables}\n")
    lines.append("**Critical questions**")
    default_space = spec.get("default_answer_space", ["yes", "no"])
    for cq in spec["critical_questions"]:
        space = cq.get("answer_space", default_space)
        extra = "" if space == default_space else f" *(answers: {', '.join(space)})*"
        lines.append(f"- **{cq['id']}**: {cq['text'].strip()}{extra}")
        if cq.get("note"):
            note = cq["note"].strip().replace(" — ", ": ")
            lines.append(f"  - Note in the YAML: {note}")
    graph = spec["graph"]
    lines.append(f"\n**Diagram** (start: {graph['start']})\n```")
    width = max(len(node) for node in graph["nodes"])
    for node, arcs in graph["nodes"].items():
        for k, (answer, target) in enumerate(arcs.items()):
            name = target.get("node") or target.get("terminal")
            if "terminal" in target:
                inverse[target["terminal"]].add(scheme)
            label = node if k == 0 else ""
            lines.append(f"{label:<{width}}  {answer:<8} → {name}")
    lines.append("```")
    terminals = sorted(
        {t["terminal"] for arcs in graph["nodes"].values() for t in arcs.values()
         if "terminal" in t}
    )
    lines.append("**Terminals**: " + ", ".join(f"`{t}`" for t in terminals))
    if spec.get("scheme_note"):
        lines.append("\n**Note in the YAML.** " + spec["scheme_note"].strip())
    return lines


def render(schemes_dir: Path) -> str:
    """The text of the document, from the eight files of ``schemes_dir``."""
    lines = ["# Argumentation schemes, critical questions and diagrams: version 2.0\n", INTRO]
    inverse: dict[str, set[str]] = defaultdict(set)
    hashes = []
    for scheme in ORDER:
        raw = (schemes_dir / f"{scheme}.yaml").read_bytes()
        hashes.append((scheme, hashlib.sha256(raw).hexdigest()))
        lines += scheme_lines(yaml.safe_load(raw), inverse, scheme)
    lines.append("\n## Inverse index: fallacy → schemes (version 2.0)\n")
    lines.append("| Terminal | Schemes |\n|---|---|")
    lines += [f"| `{t}` | {', '.join(sorted(inverse[t]))} |" for t in sorted(inverse)]
    lines.append("\n## Source files\n")
    lines.append("| File | SHA-256 |\n|---|---|")
    lines += [f"| `schemes/v2/{s}.yaml` | `{h}` |" for s, h in hashes]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schemes", type=Path, default=SCHEMES_DIR, help="the YAML folder")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="the Markdown file")
    args = parser.parse_args(argv)
    args.output.write_text(render(args.schemes), encoding="utf-8", newline="\n")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
