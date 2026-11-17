from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mspc.training.wan22_smoke_gate import _inspect_generated_video, resolve_new_run_root


def test_resolve_new_run_root_requires_one_completed_training_run(tmp_path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    new = tmp_path / "new"
    new.mkdir()
    (new / "run-snapshot.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    assert resolve_new_run_root(tmp_path, {old.resolve()}) == new.resolve()


def test_resolve_new_run_root_rejects_ambiguous_or_incomplete_runs(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_new_run_root(tmp_path, set())
    run = tmp_path / "run"
    run.mkdir()
    (run / "run-snapshot.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        resolve_new_run_root(tmp_path, set())


def test_generated_video_integrity_decodes_every_frame_and_writes_contact_sheet(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeReader:
        def __init__(self) -> None:
            self.frames = [
                np.zeros((2, 4, 3), dtype=np.uint8),
                np.full((2, 4, 3), 10, dtype=np.uint8),
                np.full((2, 4, 3), 20, dtype=np.uint8),
            ]
            self.closed = False

        def get_meta_data(self):
            return {"size": (4, 2), "fps": 15.0}

        def __iter__(self):
            return iter(self.frames)

        def close(self):
            self.closed = True

    reader = FakeReader()
    monkeypatch.setitem(
        sys.modules,
        "imageio",
        SimpleNamespace(get_reader=lambda *args, **kwargs: reader),
    )
    video_path = tmp_path / "generated.mp4"
    video_path.write_bytes(b"video")
    contact_sheet = tmp_path / "contact.png"

    report = _inspect_generated_video(
        video_path,
        expected_frames=3,
        expected_width=4,
        expected_height=2,
        expected_fps=15.0,
        shot_groups=[[0, 3]],
        contact_sheet_path=contact_sheet,
    )

    assert reader.closed is True
    assert report["status"] == "passed"
    assert report["decoded_frames"] == 3
    assert report["contact_sheet"]["sampled_frame_indices"] == [0, 1, 2]
    assert contact_sheet.is_file()
