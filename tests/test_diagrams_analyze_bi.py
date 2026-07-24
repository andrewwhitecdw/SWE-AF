"""Regression tests for the BI analysis diagram script."""

import importlib.util
import json
import sys
from pathlib import Path


def _load_analyze_bi():
    spec = importlib.util.spec_from_file_location(
        "analyze_bi",
        Path(__file__).parent.parent / "examples" / "diagrams" / "analyze_bi.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_bi"] = module
    spec.loader.exec_module(module)
    return module


def test_parse_all_skips_invalid_jsonl_lines(tmp_path, monkeypatch):
    """Malformed JSON lines should be ignored instead of crashing the parser."""
    module = _load_analyze_bi()

    log_file = tmp_path / "cat-issue-iter1.jsonl"
    log_file.write_text(
        json.dumps({"event": "start", "ts": 1.0, "model": "test"}) + "\n"
        + "this is not valid json\n"
        + json.dumps({"event": "end", "ts": 2.0}) + "\n"
    )
    monkeypatch.setattr(module, "LOG_DIR", tmp_path)

    records = module.parse_all()

    assert len(records) == 1
    assert records[0]["model"] == "test"
