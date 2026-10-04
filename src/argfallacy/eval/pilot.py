"""The pilot report: ``argfallacy pilot report RUN_ID``.

Everything here is computed from the files a run leaves behind, never by hand:
``answers.csv`` and ``summary.csv`` of the parser, ``manifest.json`` of the
client, ``data/items.csv``, ``data/annotations.csv`` and the diagrams in
``schemes/``.  ``raw.jsonl`` is read, one line at a time, only for the tokens and
the layout of the JSON, which ``answers.csv`` does not carry; where it is absent (a
local copy of the run) those sections say so and the rest of the report is
unchanged.

The report measures; it decides nothing.  The stopping rule lists the questions
above threshold at the top, and the choices it informs (what to do with
``cannot_be_determined``, whether the reasoning-on condition enters phase 2) are
taken by reading it.

Three ways of turning the hard answers into a verdict are reported side by side.
The hard answer is the one of ``summary.csv``: sample 0 for an entry that answers
once, the majority of the valid samples for an entry that answers several times.
Only the first is the diagram; the other two are exploratory readings of an answer
the diagram does not draw, not the aggregators of phase 3:

``diagram``      the hard answers, as they are: ``cannot_be_determined``, ``na`` or
                 ``invalid`` leave the traversal incomplete, with no verdict;
``logprob``      at every node, the arc whose answer has the highest ``p_logprob``
                 (the mass of the answers outside the arcs is ignored);
``toward_good``  an answer outside the arcs takes the arc with the shortest path to
                 ``good_argumentation``; where neither arc reaches it (every node of
                 ad hominem, CQ3.1 to CQ3.3 of popular opinion, CQ1.1 of analogy,
                 CQ3.1 of expert opinion), the traversal stays incomplete.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..annotations import ANNOTATIONS_FILE, ANNOTATOR, ITEMS_FILE, load_annotations
from ..client.run import RAW_NAME, read_raw
from ..labels import label_space, load_vocabulary
from ..parse import ALL_OPTIONS, INVALID, valid_samples
from ..prompts import CANNOT_BE_DETERMINED as CBD
from ..prompts import NOT_APPLICABLE as NA
from ..schemes import Scheme, load_all, traverse
from .baselines import majority_by
from .canonical import score_fallacies, score_schemes
from .metrics import accuracy

REPORT_NAME = "pilot_report.md"

OUTSIDE_ARCS = (CBD, NA, INVALID)
ANSWERS = (*ALL_OPTIONS, INVALID)

CBD_THRESHOLD = 0.50
NA_THRESHOLD = 0.50
INVALID_THRESHOLD = 0.10
EXTREME = 0.01
"""A ``p_logprob`` above ``1 - EXTREME`` or below ``EXTREME`` counts as extreme."""

RULES = ("diagram", "logprob", "toward_good")

LAYOUT = {"stage1": '{"scheme": "', "stage2": '{"answer": "'}
"""How the content of every call begins when the server compacts the JSON."""


@cache
def good_terminal() -> str:
    """The one terminal of type ``good`` in the label dictionary."""
    terminals = load_vocabulary(check_schemes=False)["terminals"]
    (good,) = [tid for tid, entry in terminals.items() if entry.get("type") == "good"]
    return good


# --------------------------------------------------------------------------- data


@dataclass
class PilotData:
    """The tables of one run, plus what they are scored against."""

    run_id: str
    answers: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, Any]
    items: pd.DataFrame
    annotations: pd.DataFrame
    schemes: dict[str, Scheme]
    raw_path: Path | None

    @property
    def entries(self) -> list[str]:
        """The entries of the configuration that have answers, in its order.

        A run made in two times has, after the first, the answers of some entries
        only: the report covers those.
        """
        present = set(self.answers["model"])
        return [e for e in self.manifest["config"]["models"] if e in present]

    @property
    def stage2(self) -> pd.DataFrame:
        return self.answers[self.answers["stage"] == "stage2"]

    @property
    def sample0(self) -> pd.DataFrame:
        frame = self.stage2
        return frame[frame["sample_index"] == 0]

    @property
    def hard(self) -> pd.DataFrame:
        """Stage-two rows of sample 0, with the hard answer of ``summary.csv`` as ``answer``.

        The logprob columns stay those of sample 0, where the parser reads them.
        """
        keys = ["model", "item_id", "scheme", "cq_id"]
        summary = self.summary[self.summary["stage"] == "stage2"][[*keys, "answer"]]
        return self.sample0.drop(columns="answer").merge(summary, on=keys, how="left")

    def samples(self, entry: str) -> int:
        return int(self.manifest["models"][entry]["samples"])


def load_pilot(
    run_dir: str | Path,
    items: str | Path = ITEMS_FILE,
    annotations: str | Path = ANNOTATIONS_FILE,
    read_raw: bool = True,
) -> PilotData:
    """Read the files of a parsed run.  ``argfallacy parse`` must have run first."""
    run_dir = Path(run_dir)
    for name in ("answers.csv", "summary.csv", "manifest.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"no {name} in {run_dir}: run `argfallacy parse` first")
    raw = run_dir / RAW_NAME
    return PilotData(
        run_id=run_dir.name,
        answers=pd.read_csv(run_dir / "answers.csv", low_memory=False),
        summary=pd.read_csv(run_dir / "summary.csv"),
        manifest=json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        items=pd.read_csv(items),
        annotations=load_annotations(annotations),
        schemes=load_all(),
        raw_path=raw if read_raw and raw.is_file() else None,
    )


def reasoning_pairs(entries: Sequence[str]) -> list[tuple[str, str]]:
    """(reasoning off, reasoning on) for every model run in both conditions."""
    return [(e, e + "_think") for e in entries if not e.endswith("_think")
            and e + "_think" in entries]


# ------------------------------------------------------------------ integrity


def integrity(data: PilotData) -> pd.DataFrame:
    """Per entry: calls, errors, invalid answers, truncated answers, partial logprobs."""
    frame = data.answers
    grouped = frame.groupby("model")
    table = pd.DataFrame({
        "calls": grouped.size(),
        "errors": grouped["error"].apply(lambda s: int(s.notna().sum())),
        "invalid": grouped["answer"].apply(lambda s: int((s == INVALID).sum())),
        "truncated": grouped["finish_reason"].apply(lambda s: int((s == "length").sum())),
        "logprob_partial": grouped["logprob_partial"].mean(),
    })
    return table.reindex(data.entries)


# ------------------------------------------------------------------ stage one


def _unanimous(frame: pd.DataFrame, keys: list[str]) -> float:
    """Share of the questions whose valid samples all gave the same answer."""
    valid = valid_samples(frame)
    return (valid.groupby(keys)["answer"].nunique() == 1).mean()


def stage_one(data: PilotData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy of the hard answer on the gold scheme per entry, and its errors.

    The hard answer is scored with the canonical scorer.  Beside it, for an entry
    that answers several times, the share of items where the valid samples agree.
    """
    frame = data.answers[data.answers["stage"] == "stage1"]
    summary = data.summary[data.summary["stage"] == "stage1"]
    gold = data.items[["item_id", "gold_scheme"]]
    rows, errors = [], []
    for entry in data.entries:
        hard = summary[summary["model"] == entry][["item_id", "answer"]]
        score = score_schemes(hard.rename(columns={"answer": "predicted"}), gold)
        own = frame[frame["model"] == entry]
        several = data.samples(entry) > 1
        rows.append({"entry": entry, "samples": data.samples(entry), "items": score.n_items,
                     "accuracy": score.accuracy,
                     "unanimous": _unanimous(own, ["item_id"]) if several else math.nan})
        merged = hard.merge(gold, on="item_id")
        wrong = merged[merged["answer"] != merged["gold_scheme"]]
        for (g, p), n in wrong.groupby(["gold_scheme", "answer"]).size().items():
            errors.append({"entry": entry, "gold": g, "predicted": p, "n": int(n)})
    return pd.DataFrame(rows).set_index("entry"), pd.DataFrame(errors)


def stage_one_by_scheme(data: PilotData) -> pd.DataFrame:
    """Accuracy of the hard answer per gold scheme (rows) and entry (columns)."""
    frame = data.summary[data.summary["stage"] == "stage1"]
    merged = frame.merge(data.items[["item_id", "gold_scheme"]], on="item_id")
    merged["correct"] = merged["answer"] == merged["gold_scheme"]
    table = merged.pivot_table(index="gold_scheme", columns="model", values="correct",
                               aggfunc="mean")
    return table.reindex(columns=data.entries)


# ------------------------------------------------------------ answers per CQ


def cq_shares(data: PilotData) -> pd.DataFrame:
    """Share of every answer per (scheme, CQ, entry), over all the samples."""
    frame = data.stage2
    counts = (frame.groupby(["scheme", "cq_id", "model"])["answer"]
              .value_counts(normalize=True).unstack(fill_value=0.0))
    for answer in ANSWERS:
        if answer not in counts:
            counts[answer] = 0.0
    counts["n"] = frame.groupby(["scheme", "cq_id", "model"]).size()
    return counts[[*ANSWERS, "n"]]


def stopping_rule(shares: pd.DataFrame) -> pd.DataFrame:
    """The (scheme, CQ, entry) above any threshold of the stopping rule, with the reason."""
    flagged = shares[(shares[CBD] > CBD_THRESHOLD) | (shares[NA] > NA_THRESHOLD)
                     | (shares[INVALID] > INVALID_THRESHOLD)].copy()
    reasons = []
    for _, row in flagged.iterrows():
        why = []
        if row[CBD] > CBD_THRESHOLD:
            why.append(f"{CBD} {row[CBD]:.0%}")
        if row[NA] > NA_THRESHOLD:
            why.append(f"na {row[NA]:.0%}")
        if row[INVALID] > INVALID_THRESHOLD:
            why.append(f"invalid {row[INVALID]:.0%}")
        reasons.append(", ".join(why))
    flagged["reason"] = reasons
    return flagged


def overall_shares(data: PilotData) -> pd.DataFrame:
    """Share of ``cannot_be_determined``, ``na`` and ``invalid`` per entry, stage two."""
    frame = data.stage2
    table = pd.DataFrame({
        CBD: frame.groupby("model")["answer"].apply(lambda s: (s == CBD).mean()),
        NA: frame.groupby("model")["answer"].apply(lambda s: (s == NA).mean()),
        INVALID: frame.groupby("model")["answer"].apply(lambda s: (s == INVALID).mean()),
    })
    return table.reindex(data.entries)


def answers_where_another_cannot_determine(data: PilotData) -> pd.DataFrame:
    """What each entry answered, as hard answer, where another entry's hard answer was
    ``cannot_be_determined`` on the same item and question.

    One row per (entry answering, entry that could not determine): the number of
    such questions and the share of each answer given there.
    """
    zero = data.hard.pivot_table(index=["item_id", "cq_id"], columns="model",
                                 values="answer", aggfunc="first")
    rows = []
    for other in data.entries:
        where = zero[zero[other] == CBD]
        for entry in data.entries:
            if entry == other:
                continue
            given = where[entry].value_counts(normalize=True)
            rows.append({"entry": entry, "where_cbd": other, "questions": len(where),
                         **{a: given.get(a, 0.0) for a in ANSWERS}})
    table = pd.DataFrame(rows).set_index(["entry", "where_cbd"])
    return table.loc[:, [c for c in table.columns if c == "questions" or table[c].any()]]


# ------------------------------------------------------------------ samples


def stability(data: PilotData) -> pd.DataFrame:
    """Per entry: samples per question and, with several, the share of unanimous questions.

    Unanimous over the valid samples; the hard answer of such a question is the
    answer of every sample.
    """
    rows = []
    for entry in data.entries:
        own = data.stage2[data.stage2["model"] == entry]
        several = data.samples(entry) > 1
        rows.append({"entry": entry, "samples": data.samples(entry),
                     "unanimous": _unanimous(own, ["item_id", "cq_id"]) if several
                     else math.nan})
    return pd.DataFrame(rows).set_index("entry")


# ------------------------------------------------------------ probabilities


def _stage2_summary(data: PilotData) -> pd.DataFrame:
    return data.summary[data.summary["stage"] == "stage2"]


def probability_correlations(data: PilotData, by_cq: bool = False) -> pd.DataFrame:
    """Spearman correlation between ``p_logprob``, ``p_verbal`` and ``p_sample``.

    Each pair on the rows that have both: an answer of ``cannot_be_determined`` or
    ``na`` at sample 0 has no ``p_verbal``, and an entry that answers once has no
    ``p_sample``.  ``n`` counts the rows with ``p_logprob`` and ``p_verbal``.
    """
    frame = _stage2_summary(data)
    keys = ["model", "scheme", "cq_id"] if by_cq else ["model"]
    rows = []
    for key, group in frame.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        corr = group[["p_logprob", "p_verbal", "p_sample"]].corr(method="spearman")
        n = len(group.dropna(subset=["p_logprob", "p_verbal"]))
        rows.append({**dict(zip(keys, key, strict=True)), "n": n,
                     "logprob_verbal": corr.loc["p_logprob", "p_verbal"],
                     "logprob_sample": corr.loc["p_logprob", "p_sample"],
                     "verbal_sample": corr.loc["p_verbal", "p_sample"]})
    table = pd.DataFrame(rows).set_index(keys)
    return table if by_cq else table.reindex(data.entries)


def probability_shape(data: PilotData) -> pd.DataFrame:
    """How much the three probabilities spread, per entry.

    ``logprob_extreme``: share of ``p_logprob`` above 0.99 or below 0.01.
    ``logprob_median_max``: median, over the calls of sample 0, of the highest
    ``p_logprob`` among the admitted answers.  ``verbal_extreme``: share of
    ``p_verbal`` at or beyond 0.95 and 0.05.  ``sample_between``: share of
    ``p_sample`` strictly between 0 and 1, where the samples disagree; empty for an
    entry that answers once.
    """
    summary = _stage2_summary(data)
    zero = data.sample0
    columns = [f"p_logprob_{a}" for a in ANSWERS if a != INVALID]
    top = zero[columns].max(axis=1)
    rows = []
    for entry in data.entries:
        own = summary[summary["model"] == entry]
        lp = own["p_logprob"].dropna()
        verbal = own["p_verbal"].dropna()
        frequency = own["p_sample"].dropna()
        rows.append({
            "entry": entry,
            "logprob_extreme": ((lp > 1 - EXTREME) | (lp < EXTREME)).mean(),
            "logprob_median_max": top[zero["model"] == entry].median(),
            "verbal_extreme": ((verbal >= 0.95) | (verbal <= 0.05)).mean(),
            "sample_between": ((frequency > 0) & (frequency < 1)).mean(),
        })
    return pd.DataFrame(rows).set_index("entry")


def logprob_extreme_by_cq(data: PilotData) -> pd.DataFrame:
    """Share of extreme ``p_logprob`` per (scheme, CQ) and entry."""
    summary = _stage2_summary(data).dropna(subset=["p_logprob"])
    extreme = (summary["p_logprob"] > 1 - EXTREME) | (summary["p_logprob"] < EXTREME)
    table = (summary.assign(extreme=extreme)
             .pivot_table(index=["scheme", "cq_id"], columns="model", values="extreme",
                          aggfunc="mean"))
    return table.reindex(columns=data.entries)


# ------------------------------------------------------------------ verdicts


def _distance_to_good(scheme: Scheme) -> dict[str, int | None]:
    """Arcs from every node to ``good_argumentation`` along the shortest path, or None."""

    @cache
    def distance(node: str) -> int | None:
        best = None
        for edge in scheme.nodes[node].values():
            if edge.is_terminal:
                step = 0 if edge.target == good_terminal() else None
            else:
                below = distance(edge.target)
                step = None if below is None else below + 1
            if step is not None and (best is None or step < best):
                best = step
        return best

    return {node: distance(node) for node in scheme.nodes}


def toward_good_arc(scheme: Scheme, node: str) -> str | None:
    """The arc of ``node`` with the shortest path to ``good_argumentation``, if any."""
    distance = _distance_to_good(scheme)
    options = {}
    for answer, edge in scheme.nodes[node].items():
        if edge.is_terminal:
            options[answer] = 0 if edge.target == good_terminal() else None
        else:
            below = distance[edge.target]
            options[answer] = None if below is None else below + 1
    reachable = {a: d for a, d in options.items() if d is not None}
    return min(reachable, key=reachable.get) if reachable else None


def _answers_for_rule(scheme: Scheme, rows: pd.DataFrame, rule: str) -> dict[str, str]:
    hard = dict(zip(rows["cq_id"], rows["answer"], strict=True))
    if rule == "diagram":
        return hard
    if rule == "toward_good":
        out = {}
        for cq, answer in hard.items():
            if answer in OUTSIDE_ARCS and cq in scheme.nodes:
                arc = toward_good_arc(scheme, cq)
                if arc is not None:
                    out[cq] = arc
            else:
                out[cq] = answer
        return out
    if rule == "logprob":
        out = {}
        for row in rows.itertuples(index=False):
            if row.cq_id not in scheme.nodes:
                continue
            arcs = list(scheme.nodes[row.cq_id])
            masses = {a: getattr(row, f"p_logprob_{a}") for a in arcs}
            masses = {a: (0.0 if pd.isna(m) else float(m)) for a, m in masses.items()}
            out[row.cq_id] = (max(masses, key=masses.get) if sum(masses.values()) > 0
                              else hard[row.cq_id])
        return out
    raise ValueError(f"unknown rule {rule!r}; known: {RULES}")


def traversals(data: PilotData, rule: str = "diagram") -> pd.DataFrame:
    """One row per (entry, item): verdict or none, and where an incomplete one stopped."""
    rows = []
    for (entry, item_id), group in data.hard.groupby(["model", "item_id"]):
        scheme = data.schemes[group["scheme"].iloc[0]]
        hard = dict(zip(group["cq_id"], group["answer"], strict=True))
        result = traverse(scheme, _answers_for_rule(scheme, group, rule))
        rows.append({
            "entry": entry, "item_id": item_id, "scheme": scheme.scheme_id,
            "verdict": result.verdict,
            "stopped_at": result.stopped_at,
            "stop_answer": hard.get(result.stopped_at) if result.stopped_at else None,
        })
    return pd.DataFrame(rows)


def incomplete_traversals(data: PilotData) -> pd.DataFrame:
    """Incomplete traversals of the diagram rule, per scheme, node and answer."""
    table = traversals(data, "diagram")
    stopped = table[table["verdict"].isna()]
    return (stopped.groupby(["scheme", "stopped_at", "stop_answer", "entry"]).size()
            .unstack("entry", fill_value=0).reindex(columns=data.entries, fill_value=0))


def majority_baseline(data: PilotData, item_ids: Iterable[str]) -> pd.DataFrame:
    """The majority fallacy per gold scheme on the test set, as a prediction per item.

    An oracle on the prior (the gold of the test set, pilot items included, is
    used to build it), the comparison line of the scorer.
    """
    test = data.items[data.items["in_test_set"].astype(bool) & data.items["gold_scheme"].notna()]
    predicted = majority_by(list(test["gold_scheme"]), list(test["gold_fallacy"]))
    frame = pd.DataFrame({"item_id": test["item_id"], "predicted": predicted})
    return frame[frame["item_id"].isin(set(item_ids))]


def _binary(gold: Sequence[str], predicted: Sequence[str | None]) -> float:
    """Accuracy of fallacy against good argumentation; no verdict is wrong."""
    good = good_terminal()
    g = [label == good for label in gold]
    p = [None if label is None or (isinstance(label, float) and math.isnan(label))
         else label == good for label in predicted]
    return accuracy(g, p)


def verdict_scores(data: PilotData) -> pd.DataFrame:
    """Per entry and rule: coverage, fine accuracy and macro F1, binary accuracy."""
    gold = data.items[["item_id", "gold_fallacy"]]
    gold_map = dict(zip(gold["item_id"], gold["gold_fallacy"], strict=True))
    rows = []
    for rule in RULES:
        table = traversals(data, rule)
        for entry in data.entries:
            own = table[table["entry"] == entry]
            predictions = own[["item_id", "verdict"]].rename(columns={"verdict": "predicted"})
            score = score_fallacies(predictions, gold, "fine")
            rows.append({
                "rule": rule, "entry": entry, "items": score.n_items,
                "coverage": own["verdict"].notna().mean(),
                "accuracy_fine": score.accuracy, "macro_f1_fine": score.macro_f1,
                "accuracy_binary": _binary([gold_map[i] for i in own["item_id"]],
                                           list(own["verdict"])),
            })
    item_ids = data.hard["item_id"].unique()
    baseline = majority_baseline(data, item_ids)
    score = score_fallacies(baseline, gold, "fine")
    rows.append({
        "rule": "majority_per_scheme", "entry": "(no model)", "items": score.n_items,
        "coverage": 1.0, "accuracy_fine": score.accuracy, "macro_f1_fine": score.macro_f1,
        "accuracy_binary": _binary([gold_map[i] for i in baseline["item_id"]],
                                   list(baseline["predicted"])),
    })
    return pd.DataFrame(rows)


# ------------------------------------------------------- human annotation


def human_cq_agreement(data: PilotData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agreement of the hard answer with the per-CQ annotation of the ``Annotazione Dumitru``
    sheets.

    Only the items annotated under the same scheme the run used; ``idk`` in the
    annotation is compared with ``cannot_be_determined``.
    """
    notes = data.annotations[data.annotations["annotator"] == ANNOTATOR]
    keys = ["item_id", "sheet", "row"]
    scheme = notes[notes["field"] == "scheme"][[*keys, "value"]].rename(
        columns={"value": "annotated_scheme"})
    answers = notes[notes["field"].str.startswith("CQ")][[*keys, "field", "value"]]
    answers = answers.merge(scheme, on=keys, how="left").rename(
        columns={"field": "cq_id", "value": "human"})
    answers["human"] = (answers["human"].str.strip().str.lower()
                        .replace({"idk": CBD}))
    model = data.hard[["item_id", "scheme", "cq_id", "model", "answer"]]
    merged = answers.merge(model, on=["item_id", "cq_id"])
    merged = merged[merged["annotated_scheme"] == merged["scheme"]]
    merged["agree"] = merged["human"] == merged["answer"]
    merged["human_idk"] = merged["human"] == CBD
    overall = merged.groupby("model").agg(
        items=("item_id", "nunique"), answers=("agree", "size"),
        human_idk=("human_idk", "sum"), agreement=("agree", "mean"))
    by_cq = merged.pivot_table(index=["scheme", "cq_id"], columns="model", values="agree",
                               aggfunc="mean")
    return overall.reindex(data.entries), by_cq.reindex(columns=data.entries)


# ------------------------------------------------------------------- cost


def latency(data: PilotData) -> pd.DataFrame:
    """Mean and median latency in seconds per entry and stage."""
    table = (data.answers.groupby(["model", "stage"])["latency_s"]
             .agg(["mean", "median"]).unstack("stage"))
    table.columns = [f"{stat}_{stage}" for stat, stage in table.columns]
    return table.reindex(data.entries)


def full_run_questions(data: PilotData) -> dict[str, int]:
    """Questions of the full run: every item at stage one, every CQ of the items with a
    gold scheme at stage two (gold condition).  Each takes the samples of its entry."""
    per_scheme = {sid: len(s.nodes) for sid, s in data.schemes.items()}
    with_scheme = data.items["gold_scheme"].dropna()
    return {"items": len(data.items),
            "stage1": len(data.items),
            "stage2": int(with_scheme.map(per_scheme).sum())}


def projection(data: PilotData) -> pd.DataFrame:
    """Hours of the full run per entry, estimated as calls x mean latency / concurrency.

    The estimate leaves out the loading of the server and the smoke test, a few
    minutes per entry and per job.  For comparison, the same estimate is given for
    the pilot itself.
    """
    questions = full_run_questions(data)
    rows = []
    for entry in data.entries:
        concurrency = int(data.manifest["models"][entry]["max_concurrency"])
        samples = data.samples(entry)
        own = data.answers[data.answers["model"] == entry]
        mean = own.groupby("stage")["latency_s"].mean()
        rows.append({
            "entry": entry, "concurrency": concurrency, "samples": samples,
            "pilot_hours_estimated": own["latency_s"].sum() / concurrency / 3600,
            "full_run_hours": samples * (questions["stage1"] * mean.get("stage1", 0.0)
                                         + questions["stage2"] * mean.get("stage2", 0.0))
            / concurrency / 3600,
        })
    return pd.DataFrame(rows).set_index("entry")


def _reasoning_text(raw_response: Mapping[str, Any]) -> str:
    payload = raw_response.get("raw") or {}
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("reasoning_content") or message.get("reasoning") or ""


TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "reasoning_tokens", "answer_tokens",
                 "reasoning_chars"]


def raw_records(raw_path: Path | None) -> pd.DataFrame | None:
    """One row per answered call of ``raw.jsonl``, read one line at a time.

    ``reasoning_tokens`` comes from ``usage``; ``reasoning_chars`` is the length
    of the reasoning field of the message, the measure that holds where vLLM
    leaves ``usage.reasoning_tokens`` at zero.  ``answer_tokens`` is the completion
    less the reasoning tokens.  ``compact``: the content, after the reasoning
    parser, begins with the layout of :data:`LAYOUT` for its stage.  A line with
    ``error`` set is no answer and is left out.  None when there is no ``raw.jsonl``.
    """
    if raw_path is None:
        return None
    records = []
    for row in read_raw(raw_path):
        if row.get("error"):
            continue
        response = row.get("raw_response") or {}
        usage = response.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        layout = LAYOUT.get(str(row.get("stage")), "")
        records.append({
            "model": row.get("model"),
            "stage": row.get("stage"),
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "answer_tokens": completion - reasoning,
            "reasoning_chars": len(_reasoning_text(response)),
            "finish_reason": response.get("finish_reason"),
            "compact": bool(layout) and (response.get("content") or "").startswith(layout),
        })
    return pd.DataFrame(records)


def token_counts(raw_path: Path | None) -> pd.DataFrame | None:
    """Mean tokens per call and entry, from :func:`raw_records`."""
    records = raw_records(raw_path)
    return None if records is None else _token_means(records)


def _token_means(records: pd.DataFrame) -> pd.DataFrame:
    return records.groupby("model")[TOKEN_COLUMNS].mean()


def output_tokens(records: pd.DataFrame) -> pd.DataFrame:
    """Per entry: distribution of the completion tokens, and the calls cut off by the limit."""
    grouped = records.groupby("model")
    tokens = grouped["completion_tokens"]
    return pd.DataFrame({
        "calls": grouped.size(),
        "median": tokens.median(),
        "p95": tokens.quantile(0.95),
        "p99": tokens.quantile(0.99),
        "max": tokens.max(),
        "length": grouped["finish_reason"].apply(lambda f: int((f == "length").sum())),
    })


def layout_share(records: pd.DataFrame) -> pd.DataFrame:
    """Share of the calls whose content begins with the compact layout, per entry and stage."""
    return records.pivot_table(index="model", columns="stage", values="compact",
                               aggfunc="mean")


# -------------------------------------------------------- reasoning on/off


def reasoning_comparison(data: PilotData, tokens: pd.DataFrame | None = None) -> pd.DataFrame:
    """For every model run in both conditions, the numbers of the comparison side by side.

    With the token counts of :func:`token_counts`, also the ratio of the completion
    tokens, reasoning included, between the two conditions.
    """
    stage1, _ = stage_one(data)
    shape = probability_shape(data)
    shares = overall_shares(data)
    lat = data.answers.groupby("model")["latency_s"].mean()
    hours = projection(data)["full_run_hours"]
    zero = data.hard.pivot_table(index=["item_id", "cq_id"], columns="model",
                                 values="answer", aggfunc="first")
    decided = set(ALL_OPTIONS) - set(OUTSIDE_ARCS)
    rows = []
    for off, on in reasoning_pairs(data.entries):
        both = zero[off].isin(decided) & zero[on].isin(decided)
        ratio = {} if tokens is None else {"completion_tokens_ratio": (
            tokens.loc[on, "completion_tokens"] / tokens.loc[off, "completion_tokens"])}
        rows.append({
            "model": off,
            "agreement_hard": (zero[off] == zero[on]).mean(),
            "agreement_both_decided": (zero.loc[both, off] == zero.loc[both, on]).mean(),
            "stage1_off": stage1.loc[off, "accuracy"],
            "stage1_on": stage1.loc[on, "accuracy"],
            "cbd_off": shares.loc[off, CBD], "cbd_on": shares.loc[on, CBD],
            "logprob_extreme_off": shape.loc[off, "logprob_extreme"],
            "logprob_extreme_on": shape.loc[on, "logprob_extreme"],
            "latency_ratio": lat[on] / lat[off], **ratio,
            "full_run_hours_off": hours[off], "full_run_hours_on": hours[on],
        })
    return pd.DataFrame(rows).set_index("model")


def agreement_by_cq(data: PilotData) -> pd.DataFrame:
    """Agreement of the hard answers of the two conditions of a model, per CQ."""
    zero = data.hard.pivot_table(index=["scheme", "cq_id", "item_id"], columns="model",
                                 values="answer", aggfunc="first")
    table = pd.DataFrame({off: (zero[off] == zero[on]).groupby(level=["scheme", "cq_id"]).mean()
                          for off, on in reasoning_pairs(data.entries)})
    return table


# ------------------------------------------------------------ parser checks


def parser_checks(data: PilotData) -> pd.DataFrame:
    """Two checks on what the minimal parser wrote.

    ``hard_not_argmax``: rows of sample 0 at stage two whose answer is not the
    answer with the highest ``p_logprob`` (at temperature 0 they should coincide; at
    a higher temperature the sample can take another answer).
    ``stage1_p_sample_not_modal``: rows of stage one in ``summary.csv`` whose
    ``p_sample`` differs from the frequency of the hard answer over the valid
    samples; ``summarise`` orients it on the first admitted answer, which at stage
    one is the first scheme id, so the number has no meaning there.  An entry that
    answers once has no ``p_sample`` and no such row.
    """
    zero = data.sample0
    zero = zero[zero["answer"] != INVALID]
    columns = [f"p_logprob_{a}" for a in ANSWERS if a != INVALID]
    masses = zero[columns].fillna(0.0)
    has_mass = masses.sum(axis=1) > 0
    argmax = masses.idxmax(axis=1).str.removeprefix("p_logprob_")
    wrong = has_mass & (argmax != zero["answer"])

    stage1 = valid_samples(data.answers[data.answers["stage"] == "stage1"])
    keys = ["model", "item_id"]
    summary1 = data.summary[data.summary["stage"] == "stage1"]
    hard = summary1.set_index(keys)["answer"].rename("hard")
    samples = stage1.join(hard, on=keys)
    modal = ((samples["answer"] == samples["hard"])
             .groupby([samples["model"], samples["item_id"]]).mean()
             .rename("frequency").reset_index())
    summary1 = summary1.merge(modal, on=keys, how="left")
    off = summary1["p_sample"].notna() & (
        (summary1["p_sample"] - summary1["frequency"]).abs() > 1e-9)
    rows = []
    for entry in data.entries:
        rows.append({
            "entry": entry,
            "hard_not_argmax": int(wrong[zero["model"] == entry].sum()),
            "stage2_rows_sample0": int((zero["model"] == entry).sum()),
            "stage1_p_sample_not_modal": int(off[summary1["model"] == entry].sum()),
            "stage1_rows": int((summary1["model"] == entry).sum()),
        })
    return pd.DataFrame(rows).set_index("entry")


# ------------------------------------------------------------------ report


def _fmt(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if kind == "pct":
        return f"{100 * value:.0f}%"
    if kind == "pct1":
        return f"{100 * value:.1f}%"
    if kind == "f2":
        return f"{value:.2f}"
    if kind == "f1":
        return f"{value:.1f}"
    if kind == "int":
        return f"{int(value)}"
    return str(value)


def markdown_table(frame: pd.DataFrame, formats: Mapping[str, str] | None = None,
                   index: bool = True) -> str:
    """A pipe table; ``formats`` maps a column to pct, pct1, f2, f1 or int."""
    formats = formats or {}
    frame = frame.reset_index() if index else frame
    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in frame.itertuples(index=False):
        cells = [_fmt(v, formats.get(c, "")) for c, v in zip(frame.columns, row, strict=True)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct_all(frame: pd.DataFrame) -> dict[str, str]:
    return {c: "pct" for c in frame.columns}


def render(data: PilotData) -> str:
    """The whole report as Markdown."""
    config = data.manifest["config"]
    shares = cq_shares(data)
    flagged = stopping_rule(shares)
    s1, s1_errors = stage_one(data)
    records = raw_records(data.raw_path)
    tokens = None if records is None else _token_means(records)
    questions = full_run_questions(data)
    items = data.hard["item_id"].nunique()
    gold = data.items.set_index("item_id").loc[data.answers["item_id"].unique()]
    out: list[str] = []
    add = out.append

    add(f"# Pilot report: {data.run_id}\n")
    add("Generated by `argfallacy pilot report` from `answers.csv`, `summary.csv` and "
        "`manifest.json` of the run, `data/items.csv`, `data/annotations.csv` and "
        "`schemes/`. Spec 04, section 4.\n")
    add(f"* Items: {len(gold)} (seed {config['items']['seed']}, stratified by gold scheme, "
        f"at least {config['items']['min_per_scheme']} per scheme); scheme condition "
        f"`{config['scheme_condition']}`.")
    add("* Entries, with their samples per question and how each is drawn: " + ", ".join(
        f"`{e}` ({m['model_id']}, {m['samples']} at temperature {m['temperature']}, top_p "
        f"{m['top_p']}, top_k {m['top_k']}, max_tokens {m['max_tokens']})"
        for e, m in ((e, data.manifest["models"][e]) for e in data.entries)) + ".")
    add("* Hard answer: sample 0 for an entry with one sample; for an entry with several, the "
        "majority of the valid samples (parsed, not cut off), a tie going to the higher mean "
        "confidence, then to the lower sample index.")
    add(f"* Calls: {data.manifest['calls_executed']} executed of "
        f"{data.manifest['calls_planned']} planned, {data.manifest['calls_failed']} failed, "
        f"{data.manifest['calls_from_cache']} from the cache. Started "
        f"{data.manifest['started_at'][:16]}, finished {data.manifest['finished_at'][:16]} (UTC).")
    add(f"* Prompts {', '.join(data.manifest['prompt_versions'])}; schemes version "
        f"{data.manifest['schemes_version']['tag']}.")
    add("* Gold schemes of the items: " + ", ".join(
        f"{k} {v}" for k, v in gold["gold_scheme"].value_counts().items()) + ".\n")

    add("## 1. Stopping rule\n")
    add(f"Questions with more than {CBD_THRESHOLD:.0%} `{CBD}`, more than "
        f"{NA_THRESHOLD:.0%} `na` or more than {INVALID_THRESHOLD:.0%} `invalid` on an "
        f"entry, over all its samples: **{len(flagged)} (question, entry) pairs, "
        f"{flagged.reset_index()[['scheme', 'cq_id']].drop_duplicates().shape[0]} distinct "
        f"questions out of {len(shares.reset_index()[['scheme', 'cq_id']].drop_duplicates())}**.\n")
    if len(flagged):
        table = flagged.reset_index()[["scheme", "cq_id", "model", "reason"]]
        table = table.rename(columns={"model": "entry"})
        add(markdown_table(table, index=False) + "\n")
    add("Share of each answer outside the arcs over all stage-two calls:\n")
    add(markdown_table(overall_shares(data), {CBD: "pct1", NA: "pct1", INVALID: "pct1"}) + "\n")

    if len(data.entries) > 1:
        add(f"Hard answer of each entry where the hard answer of another entry was `{CBD}` "
            "on the same item and question:\n")
        across = answers_where_another_cannot_determine(data)
        add(markdown_table(across, {**{c: "pct" for c in across.columns}, "questions": "int"})
            + "\n")

    add("## 2. Integrity\n")
    add(markdown_table(integrity(data), {"logprob_partial": "pct1"}) + "\n")
    add("`truncated` counts the answers with `finish_reason` equal to `length`; each is also "
        "`invalid`. `logprob_partial`: an admitted answer is missing from the top_logprobs "
        "at the answer token.\n")
    if records is None:
        add(f"Layout of the JSON: no `{RAW_NAME}` in the run folder, not measured in this "
            "copy of the report.\n")
    else:
        add("Share of the calls whose content (after the reasoning parser) begins with "
            '`{"scheme": "` at stage one and `{"answer": "` at stage two; expected 100%:\n')
        add(markdown_table(layout_share(records).reindex(data.entries), {
            stage: "pct1" for stage in LAYOUT}) + "\n")

    add("## 3. Stage one\n")
    add(markdown_table(s1, {"samples": "int", "accuracy": "f2", "unanimous": "pct"}) + "\n")
    add("`accuracy`: of the hard answer. `unanimous`: share of items whose valid samples all "
        "agree, for an entry with several.\n")
    add("Accuracy of the hard answer per gold scheme:\n")
    add(markdown_table(stage_one_by_scheme(data), {e: "f2" for e in data.entries}) + "\n")
    if len(s1_errors):
        add("Errors of the hard answer (reduced confusion matrix):\n")
        pivot = s1_errors.pivot_table(index=["gold", "predicted"], columns="entry",
                                      values="n", aggfunc="sum", fill_value=0)
        add(markdown_table(pivot.reindex(columns=data.entries, fill_value=0),
                           {e: "int" for e in data.entries}) + "\n")

    add("## 4. Samples and probabilities\n")
    add("Stability of the stage-two answers over the samples (`unanimous`: share of the "
        "questions whose valid samples all agree):\n")
    add(markdown_table(stability(data), {"samples": "int", "unanimous": "pct1"}) + "\n")
    add("Spearman correlation between the three probabilities of a yes, each pair on the rows "
        "that have both (`n`: rows with `p_logprob` and `p_verbal`):\n")
    add(markdown_table(probability_correlations(data),
                       {"n": "int", "logprob_verbal": "f2", "logprob_sample": "f2",
                        "verbal_sample": "f2"}) + "\n")
    add("Spread of the probabilities. `logprob_extreme`: share of `p_logprob` above 0.99 or "
        "below 0.01; `logprob_median_max`: median of the highest `p_logprob` among the "
        "admitted answers at sample 0; `verbal_extreme`: share of `p_verbal` at or beyond "
        "0.95 and 0.05; `sample_between`: share of `p_sample` strictly between 0 and 1.\n")
    add(markdown_table(probability_shape(data),
                       {"logprob_extreme": "pct1", "logprob_median_max": "f2",
                        "verbal_extreme": "pct1", "sample_between": "pct1"}) + "\n")

    add("## 5. Verdicts through the diagrams\n")
    add(f"Hard answers, {items} items, scored with the canonical scorer (fine space, "
        f"{len(label_space('fine'))} terminals; no verdict counts as wrong). `binary` is "
        "fallacy against good "
        "argumentation. Rules: `diagram` uses the hard answers as they are, so "
        f"`{CBD}`, `na` and `invalid` stop the traversal; `logprob` takes at every node the "
        "arc with the highest `p_logprob`; `toward_good` sends an answer outside the arcs "
        "along the arc with the shortest path to `good_argumentation`, where one exists. "
        "The last two are exploratory, not the aggregators of phase 3. "
        "`majority_per_scheme` predicts the most frequent gold fallacy of the gold "
        f"scheme on the test set (an oracle on the prior). With {items} items the 95% "
        f"interval of an accuracy near 0.5 is about {1.96 * math.sqrt(0.25 / items):.2f} "
        "wide on each side.\n")
    add(markdown_table(verdict_scores(data),
                       {"items": "int", "coverage": "pct", "accuracy_fine": "f2",
                        "macro_f1_fine": "f2", "accuracy_binary": "f2"}, index=False) + "\n")
    add("Incomplete traversals of the `diagram` rule: node where they stop and the answer "
        "found there.\n")
    add(markdown_table(incomplete_traversals(data), {e: "int" for e in data.entries}) + "\n")

    overall, by_cq = human_cq_agreement(data)
    add("## 6. Agreement with the per-CQ human annotation\n")
    add("Hard answers against the `Annotazione Dumitru` sheets, on the pilot items annotated "
        "under the same scheme (`idk` compared with `cannot_be_determined`).\n")
    add(markdown_table(overall, {"items": "int", "answers": "int", "human_idk": "int",
                                 "agreement": "pct"}) + "\n")
    add("`human_idk`: annotated answers that are `idk`.\n")
    add(markdown_table(by_cq, {e: "pct" for e in data.entries}) + "\n")

    add("## 7. Cost\n")
    add("Latency in seconds per call:\n")
    lat = latency(data)
    add(markdown_table(lat, {c: "f1" for c in lat.columns}) + "\n")
    if tokens is None:
        add(f"Tokens: no `{RAW_NAME}` in the run folder, so the token counts are not in this "
            "copy of the report. Run the command where `raw.jsonl` lives (the cluster).\n")
    else:
        add("Mean tokens per call (`reasoning_chars`: length of the reasoning field of the "
            "message, which holds where `usage` reports no reasoning tokens):\n")
        add(markdown_table(tokens.reindex(data.entries), {c: "f1" for c in tokens.columns})
            + "\n")
        add("Output tokens per call (completion, reasoning included): median, 95th and 99th "
            "percentile, maximum, and the calls cut off by `max_tokens` (`finish_reason` "
            "`length`):\n")
        out_tokens = output_tokens(records).reindex(data.entries)
        add(markdown_table(out_tokens, {c: "int" for c in out_tokens.columns}) + "\n")
    add(f"Projection of the full run: {questions['items']} items at stage one, the CQs of "
        f"the items with a gold scheme at stage two, gold condition: {questions['stage1']} + "
        f"{questions['stage2']} questions, times the samples of the entry. Hours estimated as "
        "calls x mean latency / concurrency, without loading and smoke tests; "
        "`pilot_hours_estimated` applies the same estimate to the pilot.\n")
    add(markdown_table(projection(data), {"concurrency": "int", "samples": "int",
                                          "pilot_hours_estimated": "f1",
                                          "full_run_hours": "f1"}) + "\n")

    add("## 8. Reasoning off against reasoning on\n")
    if not reasoning_pairs(data.entries):
        add("No model of this run is in both reasoning conditions.\n")
    else:
        add(markdown_table(reasoning_comparison(data, tokens),
                           {"agreement_hard": "pct", "agreement_both_decided": "pct",
                            "stage1_off": "f2", "stage1_on": "f2", "cbd_off": "pct",
                            "cbd_on": "pct", "logprob_extreme_off": "pct",
                            "logprob_extreme_on": "pct", "latency_ratio": "f1",
                            "completion_tokens_ratio": "f1",
                            "full_run_hours_off": "f1", "full_run_hours_on": "f1"}) + "\n")
        add("`agreement_both_decided`: only the questions where both conditions answered on "
            "an arc of the diagram.\n")

    add("## 9. Parser checks\n")
    checks = parser_checks(data)
    add(markdown_table(checks, {c: "int" for c in checks.columns}) + "\n")
    add("`hard_not_argmax`: stage-two rows of sample 0 whose answer is not the answer "
        "with the highest `p_logprob` (meaningful at temperature 0). "
        "`stage1_p_sample_not_modal`: stage-one rows of `summary.csv` whose `p_sample` is "
        "not the frequency of the hard answer over the valid samples; the minimal "
        "parser orients `p_sample` and `p_verbal` on the first admitted answer, which at "
        "stage one is the first scheme id, so at stage one both columns have no meaning.\n")

    add("## Appendix A. Answers per question\n")
    add("Share of each answer over all the samples; `n` is the number of calls.\n")
    for entry in data.entries:
        own = shares.xs(entry, level="model")
        own = own.loc[:, [c for c in ANSWERS if own[c].any()] + ["n"]]
        add(f"### {entry}\n")
        add(markdown_table(own, {**{c: "pct" for c in ANSWERS}, "n": "int"}) + "\n")

    add("## Appendix B. Per question: extreme logprobs and agreement between conditions\n")
    extreme = logprob_extreme_by_cq(data).add_prefix("extreme_")
    if reasoning_pairs(data.entries):
        extreme = extreme.join(agreement_by_cq(data).add_prefix("on_off_agreement_"))
    add(markdown_table(extreme, _pct_all(extreme)) + "\n")

    add("## Appendix C. Correlation of the three probabilities per question\n")
    corr = probability_correlations(data, by_cq=True)
    add(markdown_table(corr, {"n": "int", "logprob_verbal": "f2", "logprob_sample": "f2",
                              "verbal_sample": "f2"}) + "\n")
    return "\n".join(out)


def write_report(run_dir: str | Path, **kwargs: Any) -> Path:
    """Compute the report of a run and write ``pilot_report.md`` beside its files."""
    data = load_pilot(run_dir, **kwargs)
    path = Path(run_dir) / REPORT_NAME
    path.write_text(render(data), encoding="utf-8", newline="\n")
    return path


__all__ = [
    "PilotData", "load_pilot", "render", "write_report", "REPORT_NAME", "RULES",
    "integrity", "stage_one", "stage_one_by_scheme", "cq_shares", "stopping_rule",
    "overall_shares", "answers_where_another_cannot_determine", "stability",
    "probability_correlations", "probability_shape",
    "traversals", "toward_good_arc", "incomplete_traversals", "majority_baseline",
    "verdict_scores", "human_cq_agreement", "latency", "projection", "token_counts",
    "raw_records", "output_tokens", "layout_share", "full_run_questions",
    "reasoning_comparison", "agreement_by_cq", "parser_checks", "reasoning_pairs",
    "markdown_table",
]
