"""Where the DOVER adapter casts decoded frames to float, per named view.

These tests pin the *wiring*, not upstream numerics: the real
``dover.datasets.get_single_view`` is not importable here (the pinned checkout
lives on the shared disk), so a stand-in is injected. For ``technical`` the
stand-in is a pure gather, which is the property that makes the upstream order
— decompose the raw tensor, cast afterwards — bit-for-bit equal to casting
first. What is verified is therefore: the cast happens on the correct side of
``get_single_view`` for each view name, and for a gather-only view the two
orders agree exactly. The aesthetic side's real inequivalence
(``torchvision.Resize`` on uint8 vs float) is measured against the upstream
checkout, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mspc.data.quality_scorer.adapters.dover import (
    PIXEL_MEAN,
    PIXEL_STD,
    DoverBackend,
    DoverSettings,
    PreparedDoverShot,
)
from mspc.data.quality_scorer.adapters.media import ShotLocator


torch = pytest.importorskip("torch")


def _gather(video):
    """A pure indexing stand-in for the technical view's fragment gather."""

    return video[:, :, ::2, ::2]


class _RecordingSingleView:
    def __init__(self) -> None:
        self.dtypes: dict[str, object] = {}

    def __call__(self, video, name, **_sopt):
        self.dtypes[name] = video.dtype
        if name == "technical":
            return _gather(video)
        # Stand-in for the aesthetic resize: keeps the float contract without
        # importing torchvision.
        return video.float()[:, :, ::2, ::2]


def _backend() -> DoverBackend:
    backend = DoverBackend(
        DoverSettings(
            source_root=Path("/unused"),
            weight_path=Path("/unused"),
            eval_yaml=Path("/unused"),
            seed=0,
            device="cpu",
            declared_sampling={},
        )
    )
    backend._torch = torch
    backend._sample_types = {"aesthetic": {"num_clips": 1}, "technical": {"num_clips": 1}}
    return backend


def _prepared(frames: np.ndarray) -> PreparedDoverShot:
    locator = ShotLocator(
        case_id="case-dtype",
        source="cine250k",
        mapping_kind="single_file",
        shot_index=0,
        source_path="/media/film.mp4",
        frame_start=0,
        frame_end=int(frames.shape[0]) - 1,
        range_semantics="inclusive",
    )
    return PreparedDoverShot(
        locator=locator,
        local_by_view={"aesthetic": list(range(frames.shape[0])), "technical": list(range(frames.shape[0]))},
        frames_by_view={"aesthetic": frames, "technical": frames.copy()},
        width=int(frames.shape[2]),
        height=int(frames.shape[1]),
    )


def _frames() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)


def _normalise(view, num_clips: int = 1):
    mean = torch.FloatTensor(list(PIXEL_MEAN))
    std = torch.FloatTensor(list(PIXEL_STD))
    return (
        ((view.permute(1, 2, 3, 0) - mean) / std)
        .permute(3, 0, 1, 2)
        .reshape(view.shape[0], num_clips, -1, *view.shape[2:])
        .transpose(0, 1)
    )


def test_technical_view_decomposes_before_the_float_cast() -> None:
    backend = _backend()
    single_view = _RecordingSingleView()
    backend._get_single_view = single_view

    tensor = backend._view_tensor(_prepared(_frames()), "technical")

    assert single_view.dtypes["technical"] == torch.uint8
    assert tensor.dtype == torch.float32


def test_aesthetic_view_casts_to_float_before_decomposition() -> None:
    backend = _backend()
    single_view = _RecordingSingleView()
    backend._get_single_view = single_view

    tensor = backend._view_tensor(_prepared(_frames()), "aesthetic")

    assert single_view.dtypes["aesthetic"] == torch.float32
    assert tensor.dtype == torch.float32


def test_technical_gather_is_order_independent() -> None:
    """Cast-after (what the adapter now does) equals cast-before, exactly."""

    frames = _frames()
    backend = _backend()
    backend._get_single_view = _RecordingSingleView()

    observed = backend._view_tensor(_prepared(frames), "technical")

    cast_first = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float()
    expected = _normalise(_gather(cast_first))

    assert torch.equal(observed, expected)
