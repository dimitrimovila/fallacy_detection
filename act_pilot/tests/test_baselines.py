from act_pilot.baselines import lexical_predictability, source_only, tfidf_cart, tfidf_rulefit
from act_pilot.config import LexicalConfig
from act_pilot.data import LabeledItem
from act_pilot.metrics import balanced_accuracy

CFG = LexicalConfig(max_features=200, rulefit_max_rules=10, rulefit_n_estimators=20,
                    rulefit_cv=False, inner_folds=3)


def _items():
    items = []
    for k in range(40):
        label = k % 2
        word = "experts" if label else "everyone"
        source = "A" if k < 20 else "B"
        text = f"{word} say the plan for item {k} works"
        items.append(LabeledItem(f"i{k}", text, source, "g", label, k % 5))
    return items


def test_cart_and_rulefit_find_a_keyword():
    items = _items()
    labels = [i.label for i in items]
    cart = tfidf_cart(items, CFG)
    assert balanced_accuracy(labels, cart.predictions) == 1.0
    assert len(cart.details) == 5
    rulefit = tfidf_rulefit(items, CFG)
    assert balanced_accuracy(labels, rulefit.predictions) >= 0.9
    assert "experts" in rulefit.description or "everyone" in rulefit.description


def test_source_only_learns_the_source_majority():
    items = []
    for k in range(30):
        source = "NLAS" if k < 15 else "LOGIC"
        items.append(LabeledItem(f"i{k}", "text", source, "g", int(k < 15), k % 5))
    result = source_only(items)
    assert result.predictions == [i.label for i in items]


def test_lexical_predictability_of_an_answer():
    items = _items()
    answers = [i.label for i in items]  # an answer that follows the keyword
    lp = lexical_predictability(items, answers, CFG)
    assert lp.auc > 0.95
    assert "experts" in lp.positive_terms and "everyone" in lp.negative_terms
    constant = lexical_predictability(items, [1] * len(items), CFG)
    assert constant.auc != constant.auc  # NaN when the answer never changes
    assert constant.yes_rate == 1.0
