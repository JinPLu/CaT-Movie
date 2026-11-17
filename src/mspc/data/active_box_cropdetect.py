"""片源黑边用 ffmpeg cropdetect 量：逐帧取非黑区域的外接框，reset=0 把全片的框并起来。

10-11 查出旧量法（scene_shot_manifest.active_box：全片逐行逐列平均亮度 ≥16 才算画面）在暗片上
把真实画面当黑边切掉；cropdetect 是 ffmpeg 自带的标准做法，只要某一帧在该行该列亮过就算画面。
limit 取滤镜默认 24；round 默认 16 是为编码对齐、会多切真实画面，这里取 2（偶数像素）。
"""
from __future__ import annotations

CROPDETECT = "cropdetect=limit=24:round=2:reset=0"


def parse_crop(stderr_text: str, *, width: int, height: int) -> list[int]:
    """取 cropdetect 最后一行的 ``crop=w:h:x:y``，转成源像素 ``[x0, y0, x1, y1)``。"""
    last = None
    for line in stderr_text.splitlines():
        if "crop=" in line:
            last = line
    if last is None:
        raise ValueError("cropdetect printed no crop line")
    w, h, x, y = (int(v) for v in last.rsplit("crop=", 1)[1].strip().split(":"))
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"cropdetect box outside frame or empty: crop={w}:{h}:{x}:{y} in {width}x{height}")
    return [x, y, x + w, y + h]
