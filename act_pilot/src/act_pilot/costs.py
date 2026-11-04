"""The number of calls a run will make, counted before any call is made.

Two figures are given for the answerer. *Without cache* counts every request. *With cache*
counts the requests that can reach the server when the cache starts empty: the original
question is answered once per item (its answers serve the step 0 of every ACT run and the
evaluation), and the other requests are counted once each. Both are upper bounds, because
a rewrite that repeats an earlier question, or a rejected step, costs nothing. For the
optimizer the minimum assumes every rewrite passes the constraints at the first attempt
and the maximum assumes every attempt is used.

Token counts are rough: four characters per token for the input, and the full
``max_tokens`` for every output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .answers import ANSWER_SYSTEM, ANSWER_USER_TEMPLATE
from .config import PilotConfig
from .data import LabeledItem
from .prompts import OPTIMIZER_SYSTEM
from .splits import fold_splits

CHARS_PER_TOKEN = 4
PROMPT_OVERHEAD_CHARS = 1500  # scheme text, descriptions and instructions of the optimizer


@dataclass
class BoundaryCost:
    boundary: str
    n_items: int
    n_positive: int
    n_negative: int
    n_fixed_questions: int
    act_answers_without_cache: int = 0
    act_answers_with_cache: int = 0
    eval_answers_without_cache: int = 0
    eval_answers_with_cache: int = 0
    optimizer_min: int = 0
    optimizer_max: int = 0
    answer_input_tokens: int = 0
    answer_output_tokens_max: int = 0
    optimizer_input_tokens: int = 0
    optimizer_output_tokens_max: int = 0
    per_fold: list[dict] = field(default_factory=list)

    @property
    def answers_without_cache(self) -> int:
        return self.act_answers_without_cache + self.eval_answers_without_cache

    @property
    def answers_with_cache(self) -> int:
        return self.act_answers_with_cache + self.eval_answers_with_cache

    @property
    def calls_max(self) -> int:
        return self.answers_without_cache + self.optimizer_max


def estimate_boundary(
    config: PilotConfig, boundary_id: str, items: list[LabeledItem], n_manual: int
) -> BoundaryCost:
    act = config.act
    seeds = len(act.seeds)
    n_fixed = 1 + n_manual + len(config.controls)
    cost = BoundaryCost(
        boundary=boundary_id,
        n_items=len(items),
        n_positive=sum(i.label for i in items),
        n_negative=sum(1 - i.label for i in items),
        n_fixed_questions=n_fixed,
    )
    for split in fold_splits(items, act):
        n_tv = len(split.inner_train) + len(split.inner_val)
        cost.per_fold.append(
            {
                "outer_fold": split.outer_fold,
                "inner_train": len(split.inner_train),
                "inner_val": len(split.inner_val),
                "test": len(split.test),
            }
        )
        cost.act_answers_without_cache += seeds * (act.steps + 1) * n_tv
        cost.act_answers_with_cache += seeds * act.steps * n_tv
        cost.optimizer_min += seeds * act.steps * 2
        cost.optimizer_max += seeds * act.steps * (1 + act.max_rewrite_attempts)
    n = len(items)
    cost.eval_answers_without_cache = n_fixed * n + seeds * n
    cost.eval_answers_with_cache = n_fixed * n + seeds * n

    mean_chars = sum(len(i.text) for i in items) / max(n, 1)
    answer_chars = len(ANSWER_SYSTEM) + len(ANSWER_USER_TEMPLATE) + 120 + mean_chars
    cost.answer_input_tokens = int(cost.answers_with_cache * answer_chars / CHARS_PER_TOKEN)
    cost.answer_output_tokens_max = cost.answers_with_cache * config.llm.answerer.max_tokens
    shown = 4 * act.examples_per_group * (min(mean_chars, act.max_example_chars) + 60)
    critique_chars = len(OPTIMIZER_SYSTEM) + PROMPT_OVERHEAD_CHARS + shown
    rewrite_chars = len(OPTIMIZER_SYSTEM) + PROMPT_OVERHEAD_CHARS + 1200
    per_step = critique_chars + act.max_rewrite_attempts * rewrite_chars
    steps_total = cost.optimizer_max // (1 + act.max_rewrite_attempts)
    cost.optimizer_input_tokens = int(steps_total * per_step / CHARS_PER_TOKEN)
    cost.optimizer_output_tokens_max = cost.optimizer_max * config.llm.optimizer.max_tokens
    return cost


def format_costs(costs: list[BoundaryCost]) -> str:
    """A Markdown table of the estimate, one row per boundary."""
    lines = [
        "| boundary | items (pos/neg) | answerer calls, no cache | answerer calls, with cache "
        "| optimizer calls (min to max) | answerer input tokens | optimizer input tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in costs:
        lines.append(
            f"| {c.boundary} | {c.n_items} ({c.n_positive}/{c.n_negative}) "
            f"| {c.answers_without_cache:,} | {c.answers_with_cache:,} "
            f"| {c.optimizer_min:,} to {c.optimizer_max:,} | ~{c.answer_input_tokens:,} "
            f"| ~{c.optimizer_input_tokens:,} |"
        )
    total_calls = sum(c.answers_with_cache + c.optimizer_max for c in costs)
    lines.append("")
    lines.append(
        f"Total, with cache and every rewrite attempt used: at most {total_calls:,} calls."
    )
    return "\n".join(lines)
