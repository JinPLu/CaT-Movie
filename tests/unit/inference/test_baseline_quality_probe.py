from __future__ import annotations

from mspc.inference.contact_sheet import full_temporal_frame_indices


def test_baseline_contact_sheet_is_full_temporal_2fps() -> None:
    indices = full_temporal_frame_indices(98, native_fps=15.0, sample_fps=2.0)
    assert indices[0] == 0
    assert indices[-1] == 97
    assert len(indices) >= 14
