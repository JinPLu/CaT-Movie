from __future__ import annotations

from pathlib import Path

import pytest

from mspc.data.strict_json import load_strict_json_file, loads_strict_json


@pytest.mark.parametrize(
    "text",
    [
        ' {"value":1}',
        '{"value":1} ',
        '{"value":1}\n',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":1,"value":2}',
    ],
)
def test_shared_decoder_rejects_noncanonical_json(text: str):
    with pytest.raises(ValueError):
        loads_strict_json(text, label="fixture")


def test_file_decoder_allows_one_record_terminator_only(tmp_path: Path):
    valid = tmp_path / "valid.json"
    valid.write_text('{"value":1}\n', encoding="utf-8")
    assert load_strict_json_file(valid, label="fixture") == {"value": 1}

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"value":1}\n\n', encoding="utf-8")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        load_strict_json_file(invalid, label="fixture")
