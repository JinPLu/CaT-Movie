from __future__ import annotations

import pytest

from mspc.config import load_experiment_config
from mspc.runs.launch import require_train_launch_dataset
from tests.support.paths import REPO_ROOT


def test_the_cache_manifest_gate_covers_the_segment_cache() -> None:
    """段缓存通过起训前的 manifest 闸；测试的 MSPC_SHARED_ROOT 在本机不存在，必须报缺件。"""

    config = load_experiment_config(
        REPO_ROOT / "tests/fixtures/block_renderer_config.json"
    )
    with pytest.raises(ValueError, match="block training requires input file"):
        require_train_launch_dataset(config)
