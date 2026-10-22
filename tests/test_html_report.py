"""The presentation preserves saved values and paired-interval conclusions."""

import json
from html.parser import HTMLParser

import pandas as pd

from argfallacy.eval.html_report import METRICS, write_summary


def test_saved_numbers_intervals_and_safe_labels_are_presented_without_recomputing(tmp_path):
    reference = "qwen3_8_27b"
    manifest = {"name": "<experiment>", "reference_model": reference, "items": 10,
                "resamples": 10000, "seed": 123, "m": 5, "runs": {"answers": "full"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for system, score in [(f"A0:{reference}", .2), ("A2", .3), ("A5", .5),
                          (f"zeroshot:{reference}", .501)]:
        row = {"system": system, "n_items": 10}
        for metric in METRICS:
            row.update({metric: score, f"{metric}_low": score - .05,
                        f"{metric}_high": score + .05})
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "main_table.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / "main_table_without_nlas.csv", index=False)
    decide = []
    for system in ["A2", "A5"]:
        decide.append({"system": system, "macro_f1_difference": .123,
                       "macro_f1_low": -.01 if system == "A2" else .02,
                       "macro_f1_high": .2, "improvement": system == "A5",
                       "zeroshot_macro_f1_difference": -.001,
                       "zeroshot_macro_f1_low": -.03, "zeroshot_macro_f1_high": .04})
    pd.DataFrame(decide).to_csv(tmp_path / "deciding.csv", index=False)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = write_summary(tmp_path).read_text(encoding="utf-8")
    assert "+0.123" in result and "[-0.010, 0.200]" in result
    assert "[-0.030, 0.040]" in result
    assert "does not establish a difference" in result
    assert "do not show a measurable" in result
    assert "&lt;experiment&gt;" in result and "<experiment>" not in result
    assert all((tmp_path / name).read_bytes() == data for name, data in before.items())

    identifiers, targets = set(), set()

    class Resources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key == "id":
                    identifiers.add(value)
                if key == "href" and value.startswith("#"):
                    targets.add(value[1:])
                if key in {"src", "href"}:
                    assert not value.startswith(("http:", "https:", "//"))

    Resources().feed(result)
    assert targets <= identifiers
