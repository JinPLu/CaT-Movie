"""固件是 10-11 在真实 CineDance 源片上跑 ffmpeg 7.1.1 cropdetect 得到的原始输出行。"""
import pytest
from mspc.data.active_box_cropdetect import parse_crop

REAL = "[Parsed_cropdetect_0 @ 0x7f30c0003b00] x1:0 x2:703 y1:0 y2:347 w:704 h:348 x:0 y:0 pts:18170200 t:23.127690 limit:24.000000 crop=704:348:0:0"


def test_last_crop_line_becomes_source_pixel_box():
    earlier = REAL.replace("crop=704:348:0:0", "crop=700:340:2:4")
    assert parse_crop(earlier + "\n" + REAL + "\nframe= 564 fps=0.0\n", width=704, height=348) == [0, 0, 704, 348]
    assert parse_crop(REAL.replace("crop=704:348:0:0", "crop=1918:982:2:0"), width=1920, height=982) == [2, 0, 1920, 982]


def test_missing_or_empty_box_raises():
    with pytest.raises(ValueError):
        parse_crop("frame= 10 fps=0.0\n", width=704, height=348)
    with pytest.raises(ValueError):
        parse_crop(REAL.replace("crop=704:348:0:0", "crop=-688:-332:696:340"), width=704, height=348)
