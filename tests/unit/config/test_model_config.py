from __future__ import annotations

from pathlib import Path

import pytest

from mspc.config.experiment import ExperimentConfigError
from mspc.config.model import MODEL_PARSE_REGISTRY, parse_model


def test_model_registry_covers_all_supported_families() -> None:
    assert set(MODEL_PARSE_REGISTRY) == {"wan22_diffsynth"}


def test_parse_model_rejects_unknown_family(tmp_path: Path) -> None:
    with pytest.raises(ExperimentConfigError, match="model.type must be one of"):
        parse_model({"type": "unknown_backend"}, tmp_path)
