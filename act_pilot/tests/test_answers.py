import pytest

from act_pilot.answers import Answer, answer_prompt, parse_answer, reasoning_of, to_prediction


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Short reasoning.\nAnswer: yes", Answer.YES),
        ("Short reasoning.\nAnswer: No.", Answer.NO),
        ("**Answer:** Yes", Answer.YES),
        ("Answer: 'no'", Answer.NO),
        ("Final answer: yes, because the group is expert.", Answer.YES),
        ("It is unclear.\nAnswer: cannot determine", Answer.UNDETERMINED),
        ("Answer: Cannot be determined from the text", Answer.UNDETERMINED),
        ("Answer: not enough information", Answer.UNDETERMINED),
        ("Answer: unclear", Answer.UNDETERMINED),
        ("Answer: yes/no", Answer.INVALID),
        ("Answer: maybe", Answer.INVALID),
        ("I would say the text is popular.", Answer.INVALID),
        ("The reasoning goes here.\nYes", Answer.YES),
        ("The reasoning goes here.\n**No.**", Answer.NO),
        ("<think>Answer: yes is tempting</think>The group is biased.\nAnswer: no", Answer.NO),
        ("<think>never closed Answer: yes", Answer.INVALID),
        ("First draft. Answer: yes\nOn reflection, no.\nAnswer: no", Answer.NO),
        ("Answer: nothing suggests it", Answer.INVALID),
        ("", Answer.INVALID),
        (None, Answer.INVALID),
    ],
)
def test_parse_answer(raw, expected):
    assert parse_answer(raw) is expected


def test_not_is_not_no():
    assert parse_answer("Answer: not really") is Answer.INVALID


def test_prediction_policy():
    assert to_prediction(Answer.YES) == 1
    assert to_prediction(Answer.NO) == 0
    assert to_prediction(Answer.UNDETERMINED) == 0
    assert to_prediction(Answer.INVALID, "positive") == 1


def test_prompt_contains_question_and_text_slot():
    prompt = answer_prompt("Is   the group  expert?")
    assert "Question: Is the group expert?" in prompt.user_template
    assert "{text}" in prompt.user_template
    messages = prompt.render("Everyone agrees.")
    assert messages[0]["role"] == "system"
    assert "Everyone agrees." in messages[1]["content"]
    assert "{text}" not in messages[1]["content"]


def test_reasoning_of_drops_answer_line():
    assert reasoning_of("The group is biased.\nAnswer: no") == "The group is biased."
