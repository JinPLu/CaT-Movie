import pytest

from mspc.data.source_pipeline import content_size


@pytest.mark.parametrize("frame, expected", [
    ((1920, 1080), (256, 144)), ((3840, 2160), (256, 144)), ((1280, 720), (256, 144)),
    ((1920, 800), (256, 107)), ((720, 1280), (144, 256)), ((256, 144), (256, 144)), ((200, 100), (200, 100)),
])
def test_content_size_follows_pyscenedetect_auto_downscale(frame, expected):
    assert content_size(*frame) == expected
