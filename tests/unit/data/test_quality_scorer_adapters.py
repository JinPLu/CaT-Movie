"""CPU-checkable behaviour of the real quality scorer adapters.

The model calls themselves need A100 weights and are exercised by the parity
smoke, not here. What is checkable locally is the part that decides *what* gets
scored and *how* the results are reduced: shot geometry, frame sampling,
invalid propagation, and the import boundary that keeps torch and
onnxruntime out of the orchestrator.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mspc.data.quality_scorer.adapters.dover import shot_seed
from mspc.data.quality_scorer.adapters.media import ShotLocator
from tests.support.paths import REPO_ROOT


ADAPTERS_DIR = REPO_ROOT / "src" / "mspc" / "data" / "quality_scorer" / "adapters"
HEAVY_IMPORTS = frozenset({"torch", "onnxruntime", "transformers", "decord", "cv2", "pyclipper"})


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def test_adapters_do_not_import_model_stacks_at_module_level() -> None:
    files = sorted(ADAPTERS_DIR.glob("*.py"))
    assert files
    for path in files:
        blocked = _module_level_imports(path) & HEAVY_IMPORTS
        assert not blocked, f"{path.name} imports {sorted(blocked)} at module level"


def test_shot_locator_frame_count_follows_semantics() -> None:
    half_open = ShotLocator(
        case_id="ci-1",
        source="ci_vid",
        mapping_kind="multisource",
        shot_index=0,
        source_path="/media/a.mp4",
        frame_start=4,
        frame_end=20,
        range_semantics="half_open",
    )
    inclusive = ShotLocator(
        case_id="cine-1",
        source="cine250k",
        mapping_kind="single_file",
        shot_index=0,
        source_path="/media/film.mp4",
        frame_start=10,
        frame_end=21,
        range_semantics="inclusive",
    )
    assert half_open.frame_count == 16
    assert inclusive.frame_count == 12
    assert inclusive.absolute_index(11) == 21
    with pytest.raises(ValueError, match="outside the shot"):
        inclusive.absolute_index(12)


def test_shot_seed_is_stable_and_case_specific() -> None:
    assert shot_seed(0, "case-1", 0) == shot_seed(0, "case-1", 0)
    assert shot_seed(0, "case-1", 0) != shot_seed(0, "case-1", 1)
    assert shot_seed(0, "case-1", 0) != shot_seed(1, "case-1", 0)
