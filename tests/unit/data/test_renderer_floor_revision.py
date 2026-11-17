"""真实旧标注的未知 observation 仍须被事实 schema 拒绝；不再钉退役提示词的原句。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_real_unknown_observation_is_rejected():
    from mspc.data.state_facts import AtomicFact
    import pytest
    failure = json.loads((ROOT / "tests/fixtures/renderer_floor_first/v6_failures.json").read_text())
    with pytest.raises(ValueError, match="unknown observation"):
        AtomicFact.model_validate(failure["unknown_fact"])
