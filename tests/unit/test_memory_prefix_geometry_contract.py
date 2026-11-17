"""参考＋联合目标段统一布局；无参考时保持现有目标坐标和文字。"""

from __future__ import annotations

import pytest

from mspc.core.wan22_contract import MEMORY_ROW_ROLE_TEXT, memory_prefix_case_layout
from mspc.inference.wan22 import memory_prefix_inference_layout

ROWS = (MEMORY_ROW_ROLE_TEXT, MEMORY_ROW_ROLE_TEXT)
TARGETS = ("first  shot\n", "second shot")
RGB = ((0, 9), (9, 22))
LATENT = ((0, 3), (3, 7))


def test_reference_groups_precede_all_target_groups() -> None:
    assert memory_prefix_case_layout(
        target_prompts=TARGETS,
        target_shot_groups=RGB,
        target_latent_shot_groups=LATENT,
        memory_rows=ROWS, row_policy="fixed_role_v1",
    ) == (
        (*ROWS, *TARGETS),
        ((0, 1), (1, 2), (2, 11), (11, 24)),
        ((0, 1), (1, 2), (2, 5), (5, 9)),
    )


def test_zero_reference_preserves_target_rows_and_coordinates() -> None:
    assert memory_prefix_case_layout(
        target_prompts=TARGETS,
        target_shot_groups=RGB,
        target_latent_shot_groups=LATENT,
        memory_rows=(), row_policy="fixed_role_v1",
    ) == (TARGETS, RGB, LATENT)


@pytest.mark.parametrize("rows", [("", ""), (f"{MEMORY_ROW_ROLE_TEXT} A man in a blue coat.",)])
def test_reference_rows_require_fixed_role_instead_of_unverified_caption(rows) -> None:
    with pytest.raises(ValueError, match="fixed_role_v1"):
        memory_prefix_case_layout(
            target_prompts=TARGETS, target_shot_groups=RGB,
            target_latent_shot_groups=LATENT, memory_rows=rows, row_policy="fixed_role_v1")


def test_inference_consumes_complete_rows_through_the_same_layout() -> None:
    actual = memory_prefix_inference_layout(
        prompts=TARGETS,
        shot_groups=RGB,
        latent_shot_groups=LATENT,
        memory_shot_texts=ROWS,
        row_policy="fixed_role_v1",
    )
    expected = memory_prefix_case_layout(
        target_prompts=TARGETS,
        target_shot_groups=RGB,
        target_latent_shot_groups=LATENT,
        memory_rows=ROWS, row_policy="fixed_role_v1",
    )
    assert (tuple(actual[0]), actual[1], actual[2]) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_prompts": ()},
        {"target_shot_groups": ((0, 9),)},
        {"target_shot_groups": ((0, 9), (10, 23))},
        {"target_shot_groups": ((0, 9), (9, 23))},
        {"target_latent_shot_groups": ((0, 3), (3, 8))},
        {"target_latent_shot_groups": ((0, 3.0), (3, 7))},
    ],
)
def test_invalid_geometry_raises(overrides: dict) -> None:
    kwargs = dict(target_prompts=TARGETS, target_shot_groups=RGB, target_latent_shot_groups=LATENT, memory_rows=ROWS, row_policy="fixed_role_v1")
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        memory_prefix_case_layout(**kwargs)


def test_scalar_reference_text_is_not_a_row_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        memory_prefix_case_layout(target_prompts=TARGETS, target_shot_groups=RGB, target_latent_shot_groups=LATENT, memory_rows="reference", row_policy="fixed_role_v1")


def test_frame_visible_identity_rows_use_the_configured_policy_in_train_and_inference():
    rows = (f"{MEMORY_ROW_ROLE_TEXT} Visible identities: Character 1.", MEMORY_ROW_ROLE_TEXT)
    expected = (rows + TARGETS, ((0, 1), (1, 2), (2, 11), (11, 24)), ((0, 1), (1, 2), (2, 5), (5, 9)))
    actual = memory_prefix_case_layout(target_prompts=TARGETS, target_shot_groups=RGB,
        target_latent_shot_groups=LATENT, memory_rows=rows, row_policy="frame_entities_v1")
    inference = memory_prefix_inference_layout(prompts=TARGETS, shot_groups=RGB,
        latent_shot_groups=LATENT, memory_shot_texts=rows, row_policy="frame_entities_v1")
    assert actual == expected
    assert (tuple(inference[0]), inference[1], inference[2]) == expected
    with pytest.raises(ValueError, match="fixed_role_v1"):
        memory_prefix_case_layout(target_prompts=TARGETS, target_shot_groups=RGB,
            target_latent_shot_groups=LATENT, memory_rows=rows, row_policy="fixed_role_v1")
