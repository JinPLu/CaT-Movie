"""Identity pins for the quality-scorer speedup: dense-span frame fetch and DOVER batching."""

from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import pytest

from mspc.data.quality_scorer.adapters.media import (
    ShotLocator,
    decode_local_frames,
    fetch_absolute_indices,
)
from mspc.data.quality_scorer.adapters.dover import (
    DEFAULT_DOVER_BATCH,
    DOVER_BATCH_ENV,
    DoverBackend,
    DoverSettings,
    PreparedDoverShot,
    require_dover_batch_size,
    sampled_frame_record,
)


def _locator(path: str = "/media/film.mp4", *, shot_index: int = 0) -> ShotLocator:
    return ShotLocator(
        case_id="case-1",
        source="cine250k",
        mapping_kind="single_file",
        shot_index=shot_index,
        source_path=path,
        frame_start=0,
        frame_end=31,
        range_semantics="inclusive",
    )


class IndexReader:
    def __init__(self, path: str = "film", *, total: int = 200) -> None:
        self.path = path
        self.total = total
        self.fetched: list[int] | None = None

    def __len__(self) -> int:
        return self.total

    def get_batch(self, indices: list[int]) -> np.ndarray:
        self.fetched = [int(index) for index in indices]
        frames = np.zeros((len(indices), 2, 2, 3), dtype=np.uint8)
        marker = sum(ord(char) for char in self.path) % 256
        for position, index in enumerate(indices):
            frames[position] = (int(index) % 256, marker, 0)
        return frames


def test_fetch_absolute_indices_fills_a_dense_span_and_keeps_a_sparse_list() -> None:
    assert fetch_absolute_indices([10, 12, 14]) == list(range(10, 15))
    assert fetch_absolute_indices([0, 100, 200]) == [0, 100, 200]
    assert fetch_absolute_indices([3, 3, 3]) == [3]
    assert fetch_absolute_indices(np.array([10, 12, 14])) == list(range(10, 15))
    with pytest.raises(ValueError, match="no absolute frames"):
        fetch_absolute_indices(np.array([], dtype=np.int64))


def test_decode_preserves_caller_order_and_repeats() -> None:
    reader = IndexReader()
    frames = decode_local_frames(reader, _locator(), [4, 0, 4, 2])
    assert reader.fetched == list(range(0, 5))
    assert frames.shape == (4, 2, 2, 3)
    assert [int(frame[0, 0, 0]) for frame in frames] == [4, 0, 4, 2]
    again = decode_local_frames(IndexReader(), _locator(), np.array([4, 0, 4, 2]))
    np.testing.assert_array_equal(again, frames)


def test_decode_sparse_span_fetches_only_unique_sorted_indices() -> None:
    reader = IndexReader(total=400)
    locator = ShotLocator(
        case_id="case-1",
        source="cine250k",
        mapping_kind="single_file",
        shot_index=0,
        source_path="/media/film.mp4",
        frame_start=0,
        frame_end=399,
        range_semantics="inclusive",
    )
    frames = decode_local_frames(reader, locator, [0, 200, 0])
    assert reader.fetched == [0, 200]
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 200, 0]


class CountingBatchReader(IndexReader):
    get_batch_calls = 0
    get_batch_threads: list[int] = []

    def __init__(self, path: str = "film", *, total: int = 200) -> None:
        super().__init__(path, total=total)

    def get_batch(self, indices: list[int]) -> np.ndarray:
        CountingBatchReader.get_batch_calls += 1
        CountingBatchReader.get_batch_threads.append(threading.get_ident())
        return super().get_batch(indices)


class _FakeDover:
    sample_threads: list[int] = []

    def sample_local_indices(self, locator: ShotLocator) -> dict[str, list[int]]:
        _FakeDover.sample_threads.append(threading.get_ident())
        return {"aesthetic": [0, 2], "technical": [1, 3]}

    def prepare_shot(self, reader: object, locator: ShotLocator) -> PreparedDoverShot:
        local = self.sample_local_indices(locator)
        combined = [index for name in ("aesthetic", "technical") for index in local[name]]
        frames = decode_local_frames(reader, locator, combined)
        return self.prepared_from_frames(
            locator,
            local,
            {"aesthetic": frames[:2], "technical": frames[2:]},
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
        )

    def prepared_from_frames(
        self,
        locator: ShotLocator,
        local_by_view: dict[str, list[int]],
        frames_by_view: dict[str, object],
        *,
        width: int,
        height: int,
    ) -> PreparedDoverShot:
        return PreparedDoverShot(
            locator=locator,
            local_by_view=local_by_view,
            frames_by_view=frames_by_view,
            width=width,
            height=height,
        )

    def score_prepared(self, prepared: PreparedDoverShot) -> dict[str, object]:
        return self.score_prepared_batch([prepared], batch_size=1)[0]

    def score_prepared_batch(
        self,
        prepared: list[PreparedDoverShot],
        *,
        batch_size: int = 4,
    ) -> list[dict[str, object]]:
        rows = []
        for item in prepared:
            rows.append(
                {
                    **sampled_frame_record(item.locator, item.local_by_view),
                    "width": item.width,
                    "height": item.height,
                    "aesthetic_logit": 0.25,
                    "aesthetic_sigmoid": 0.56,
                    "technical_logit": -0.1,
                    "technical_sigmoid": 0.47,
                }
            )
        return rows


def test_dover_batch_size_defaults_to_four_and_refuses_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DOVER_BATCH_ENV, raising=False)
    assert DEFAULT_DOVER_BATCH == 4
    assert require_dover_batch_size({}) == 4
    assert require_dover_batch_size({DOVER_BATCH_ENV: "8"}) == 8
    assert require_dover_batch_size(batch_size=2) == 2
    with pytest.raises(ValueError, match="dover.batch_size"):
        require_dover_batch_size(batch_size=0)
    with pytest.raises(ValueError, match=DOVER_BATCH_ENV):
        require_dover_batch_size({DOVER_BATCH_ENV: "0"})
    with pytest.raises(ValueError, match=DOVER_BATCH_ENV):
        require_dover_batch_size({DOVER_BATCH_ENV: "cpu"})


class _MarkerDoverModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, views: dict[str, object]) -> list[object]:
        import torch

        assert len(views) == 1
        name, tensor = next(iter(views.items()))
        self.calls.append((name, int(tensor.shape[0])))
        flat = tensor.reshape(tensor.shape[0], -1)
        return [flat.sum(dim=-1)]


class _SpatialHeadDoverModel:
    """Real DOVER aesthetic head returns a 28x28 map, not one logit per clip."""

    spatial = 784

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, views: dict[str, object]) -> list[object]:
        import torch

        assert len(views) == 1
        name, tensor = next(iter(views.items()))
        lead = int(tensor.shape[0])
        self.calls.append((name, lead))
        marker = tensor.reshape(lead, -1)[:, 0]
        return [marker[:, None].expand(lead, self.spatial)]


def _prepared_marker_shot(shot_index: int, marker: int) -> PreparedDoverShot:
    locator = _locator(f"/media/shot-{shot_index}.mp4", shot_index=shot_index)
    frames = np.full((2, 4, 4, 3), marker, dtype=np.uint8)
    return PreparedDoverShot(
        locator=locator,
        local_by_view={"aesthetic": [0, 1], "technical": [0, 1]},
        frames_by_view={"aesthetic": frames, "technical": frames.copy()},
        width=4,
        height=4,
    )


def test_dover_named_view_batch_matches_one_at_a_time() -> None:
    import torch

    settings = DoverSettings(
        source_root=Path("/unused"),
        weight_path=Path("/unused"),
        eval_yaml=Path("/unused"),
        seed=0,
        device="cpu",
        declared_sampling={},
    )
    backend = DoverBackend(settings)
    model = _MarkerDoverModel()
    backend._model = model
    backend._torch = torch
    backend._get_single_view = lambda stacked, name, **_sopt: stacked
    backend._sample_types = {"aesthetic": {"num_clips": 1}, "technical": {"num_clips": 1}}
    prepared = [_prepared_marker_shot(index, marker=10 * (index + 1)) for index in range(4)]
    batched = backend.score_prepared_batch(prepared, batch_size=4)
    assert model.calls == [("aesthetic", 4), ("technical", 4)]
    model.calls.clear()
    serial = [backend.score_prepared(item) for item in prepared]
    assert model.calls == [("aesthetic", 1), ("technical", 1)] * 4
    assert batched == serial
    assert len({row["aesthetic_logit"] for row in batched}) == 4


def test_dover_named_view_batch_means_spatial_head_like_real_model() -> None:
    import torch

    settings = DoverSettings(
        source_root=Path("/unused"),
        weight_path=Path("/unused"),
        eval_yaml=Path("/unused"),
        seed=0,
        device="cpu",
        declared_sampling={},
    )
    backend = DoverBackend(settings)
    model = _SpatialHeadDoverModel()
    backend._model = model
    backend._torch = torch
    backend._get_single_view = lambda stacked, name, **_sopt: stacked
    backend._sample_types = {"aesthetic": {"num_clips": 1}, "technical": {"num_clips": 1}}
    prepared = [_prepared_marker_shot(index, marker=10 * (index + 1)) for index in range(4)]
    batched = backend.score_prepared_batch(prepared, batch_size=4)
    serial = [backend.score_prepared(item) for item in prepared]
    for batch_row, serial_row in zip(batched, serial):
        for key in ("aesthetic_logit", "technical_logit"):
            assert batch_row[key] == pytest.approx(serial_row[key])
    assert len({row["aesthetic_logit"] for row in batched}) == 4
