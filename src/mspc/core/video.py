"""生成片段的无损拼接；不依赖benchmark题面或旧原生基线脚本。"""
from __future__ import annotations
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Mapping, Sequence


def concat_shot_videos(
    clips: Sequence[str | Path],
    *,
    output_path: str | Path,
    ffmpeg: str | None = None,
) -> Path:
    """Concatenate independently generated shot clips with stream copy.

    This is the native Wan floor path: each shot is a standalone clip, then
    the clips are joined. Plain Wan has no multi-shot timeline.
    """
    if not clips:
        raise ValueError("concat_shot_videos requires at least one clip")
    resolved_clips = [Path(clip).expanduser().resolve() for clip in clips]
    for clip in resolved_clips:
        if not clip.is_file() or clip.stat().st_size <= 0:
            raise FileNotFoundError(f"generated shot video missing or empty: {clip}")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(resolved_clips) == 1:
        destination.write_bytes(resolved_clips[0].read_bytes())
        return destination
    ffmpeg_bin = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError("ffmpeg is required to concatenate single-shot shot clips")
    concat_dir = destination.parent / "concat_lists"
    concat_dir.mkdir(parents=True, exist_ok=True)
    list_path = concat_dir / f"{destination.stem}.txt"
    list_path.write_text(
        "\n".join(f"file {shlex.quote(str(clip))}" for clip in resolved_clips) + "\n",
        encoding="utf-8",
    )
    command = [
        str(ffmpeg_bin),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(destination),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"concatenated video missing or empty: {destination}")
    return destination


def generate_story_by_single_shot_concat(
    *,
    shot_inputs: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    seed: int,
    shot_seed_stride: int,
    generate_shot: Any,
    concat_clips: Any = None,
    ffmpeg: str | None = None,
) -> list[Path]:
    """Generate each shot independently, then concatenate.

    ``generate_shot`` must write one native-Wan K=1 clip. The runtime path is
    this function, not a flag on the multi-shot sampler.

    Shot ``i`` draws its initial noise from ``seed + shot_seed_stride * i``. A stride of 0
    gives every shot of a story the same seed, which is how every per-shot render before
    2026-09-26 was made.
    """
    if not shot_inputs:
        raise ValueError("single-shot single-shot concat requires at least one shot")
    if type(seed) is not int:
        raise ValueError("generation seed must be an int")
    if type(shot_seed_stride) is not int or shot_seed_stride < 0:
        raise ValueError(f"shot_seed_stride must be a non-negative int; got {shot_seed_stride!r}")
    destination = Path(output_path)
    shots_dir = destination.parent / "shots" / destination.stem
    shot_paths: list[Path] = []
    for index, shot in enumerate(shot_inputs):
        if not isinstance(shot, Mapping):
            raise ValueError(f"shot_inputs[{index}] must be an object")
        groups = shot.get("shot_groups")
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError(
                f"shot_inputs[{index}] must be a native Wan K=1 case "
                f"(one shot_groups span); got {groups!r}"
            )
        shot_ids = shot.get("shot_ids")
        if not isinstance(shot_ids, list) or len(shot_ids) != 1:
            raise ValueError(f"shot_inputs[{index}].shot_ids must contain one shot id")
        shot_path = shots_dir / f"{index:02d}_{shot_ids[0]}.mp4"
        generate_shot(shot_inputs=shot, output_path=shot_path, seed=seed + shot_seed_stride * index)
        if not shot_path.is_file() or shot_path.stat().st_size <= 0:
            raise RuntimeError(f"single-shot shot video was not written: {shot_path}")
        shot_paths.append(shot_path)
    concat = concat_shot_videos if concat_clips is None else concat_clips
    concat(shot_paths, output_path=destination, ffmpeg=ffmpeg)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"single-shot concatenated video was not written: {destination}")
    return shot_paths
