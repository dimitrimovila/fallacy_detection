"""The reader of the ``Annotazione Dumitru`` sheets, on a small workbook written here."""

from __future__ import annotations

import pandas as pd
import pytest
from openpyxl import Workbook

from argfallacy.annotations import COLUMNS, load_annotations, update
from argfallacy.schemes import SchemeError

HEADER = ["item_id", "text", "scheme", "CQ1", "CQ1.1", "CQ2", "CQ3", "CQ4",
          "verdetto", "incertezza", "nota"]
ROWS = [
    ["a1", "First argument.", "YES", "yes", "no", "na", "idk", "yes", "Weak Analogy",
     "CQ1.1:AMB", ""],
    ["ESCLUSO", "A duplicate.", "YES", "yes", "yes", "yes", "yes", "yes", "Good Argumentation",
     "", ""],
    ["b2", "Second argument.", "NO", "na", "na", "na", "na", "na", "n.a. (schema assente)", "",
     ""],
]


@pytest.fixture
def files(tmp_path):
    items = tmp_path / "items.csv"
    pd.DataFrame({"item_id": ["a1", "b2"],
                  "text": ["First argument.", "Second argument."]}).to_csv(items, index=False)
    table = tmp_path / "annotations.csv"
    pd.DataFrame([["a1", "dumitru", "Annotazione Dumitru Analogy", 2, "verdict", "old", "old"],
                  ["a1", "enrico", "new_analogy", 5, "verdict", "weak_analogy", "Weak"]],
                 columns=COLUMNS).to_csv(table, index=False)
    return items, table


def _workbook(path, rows):
    book = Workbook()
    sheet = book.active
    sheet.title = "Annotazione Dumitru Analogy"
    for row in [HEADER, *rows]:
        sheet.append(row)
    book.save(path)
    return path


def test_the_dumitru_rows_are_replaced_and_the_rest_stays(tmp_path, files):
    items, table = files
    update(_workbook(tmp_path / "book.xlsx", ROWS), table, items)
    frame = load_annotations(table)
    rows = frame[frame["annotator"] == "dumitru"].set_index(["item_id", "field"])["value"]

    assert rows["a1", "scheme"] == "analogy"
    assert rows["a1", "verdict"] == "weak_analogy"
    assert rows["a1", "CQ1.1"] == "no" and rows["a1", "CQ2"] == "na" and rows["a1", "CQ3"] == "idk"
    assert rows["a1", "uncertainty"] == "CQ1.1:AMB"
    assert rows["b2", "scheme"] == "none" and ("b2", "verdict") not in rows.index
    assert "old" not in set(frame["value"])
    assert "ESCLUSO" not in set(frame["item_id"])
    assert (frame["annotator"] == "enrico").sum() == 1


def test_the_scheme_column_is_the_one_just_before_the_cqs(tmp_path, files):
    items, table = files
    header = ["item_id", "text", "scheme", "Scheme", *HEADER[3:]]
    rows = [[row[0], row[1], "NO", *row[2:]] for row in ROWS]
    book = Workbook()
    book.active.title = "Annotazione Dumitru Analogy"
    for row in [header, *rows]:
        book.active.append(row)
    book.save(tmp_path / "book.xlsx")
    update(tmp_path / "book.xlsx", table, items)
    frame = load_annotations(table)
    rows = frame[frame["annotator"] == "dumitru"].set_index(["item_id", "field"])
    assert rows.loc[("a1", "scheme"), "value"] == "analogy"
    assert rows.loc[("a1", "scheme"), "raw"] == "YES"


def test_an_id_column_pasted_one_row_off_stops_everything(tmp_path, files):
    items, table = files
    before = table.read_bytes()
    shifted = [[ROWS[i - 1][0] if i else "", *row[1:]] for i, row in enumerate(ROWS)]
    with pytest.raises(SchemeError, match="the text is not that of item"):
        update(_workbook(tmp_path / "book.xlsx", shifted), table, items)
    assert table.read_bytes() == before


@pytest.mark.parametrize("column", ["scheme", "CQ1", "CQ4"])
@pytest.mark.parametrize("value", ["-", "n.a.(C=A)"])
def test_the_markers_of_the_annotation_along_the_path_stop_the_reader(tmp_path, files, column,
                                                                       value):
    items, table = files
    before = table.read_bytes()
    first = list(ROWS[0])
    first[HEADER.index(column)] = value
    with pytest.raises(SchemeError, match="unknown"):
        update(_workbook(tmp_path / "book.xlsx", [first, *ROWS[1:]]), table, items)
    assert table.read_bytes() == before
