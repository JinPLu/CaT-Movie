"""Shared helpers for feeding local video/image assets to Qwen VL APIs."""

from __future__ import annotations

import base64
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


def resolve_media_path(path: str | Path, repo_root: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if repo_root is not None:
        rooted = repo_root / candidate
        if rooted.is_file():
            return rooted
    return candidate


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def video_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _uniform_subsample(items: list[Path], limit: int) -> list[Path]:
    """Keep ``limit`` items spread evenly, always retaining the first and last."""

    if limit >= len(items):
        return items
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / (limit - 1)
    picked = sorted({int(round(index * step)) for index in range(limit)})
    return [items[index] for index in picked]


def extract_video_frames_by_index(
    video_path: Path,
    *,
    indices: list[int],
    out_dir: Path,
    width: int,
    jpeg_quality: int = 3,
) -> list[Path]:
    """Decode exactly the requested 0-based frame indices of ``video_path``.

    ``sample_video_frames_jpeg`` samples by wall-clock fps, which cannot address
    a specific frame and cannot honour a per-segment quota inside one file.  A
    caller that knows where the segment boundaries are (declared shot lengths,
    say) needs frames by index instead, so this selects them directly and
    verifies the count rather than trusting ffmpeg's rounding.
    """

    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if not indices:
        raise ValueError("indices must be non-empty")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("indices must be non-negative ints")
    if sorted(indices) != indices or len(set(indices)) != len(indices):
        raise ValueError("indices must be strictly increasing")
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive int")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 31:
        raise ValueError("jpeg_quality must be an int in 1..31")

    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg is required to sample video frames for vision judging")

    out_dir.mkdir(parents=True, exist_ok=True)
    selector = "+".join(f"eq(n\\,{index})" for index in indices)
    completed = subprocess.run(
        [
            binary,
            "-v", "error",
            "-y",
            "-i", str(video_path),
            "-vf", f"select='{selector}',scale={width}:-2",
            "-fps_mode", "passthrough",
            "-q:v", str(jpeg_quality),
            str(out_dir / "frame_%05d.jpg"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[:400]
        raise RuntimeError(f"ffmpeg frame selection failed for {video_path}: {detail}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) != len(indices):
        raise RuntimeError(
            f"{video_path}: selected {len(frames)} frames for {len(indices)} requested "
            f"indices (max index {indices[-1]})"
        )
    return frames


def sample_video_frames_jpeg(
    video_path: Path,
    *,
    out_dir: Path,
    fps: float,
    max_frames: int,
    width: int,
    jpeg_quality: int = 3,
) -> list[Path]:
    """Decode ``video_path`` into ordered JPEG frames for image-part vision calls.

    OpenAI-compatible proxies silently discard unknown ``video_url`` content
    parts, so a judge that must see pixels has to receive real image parts.
    Every failure here is raised: an empty or partial sample would otherwise
    become a text-only judgement.
    """

    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or float(fps) <= 0:
        raise ValueError("fps must be a positive number")
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames must be a positive int")
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive int")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 31:
        raise ValueError("jpeg_quality must be an int in 1..31")

    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg is required to sample video frames for vision judging")

    out_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            binary,
            "-v", "error",
            "-y",
            "-i", str(video_path),
            "-vf", f"fps={float(fps)},scale={width}:-2",
            "-q:v", str(jpeg_quality),
            str(out_dir / "frame_%05d.jpg"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[:400]
        raise RuntimeError(f"ffmpeg frame sampling failed for {video_path}: {detail}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
    return _uniform_subsample(frames, max_frames)


def video_content_entry(
    path: Path,
    *,
    fps: float | None = 2.0,
    max_frames: int | None = 32,
    use_file_uri: bool = True,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"video file not found: {path}")
    if use_file_uri:
        row: dict[str, Any] = {"type": "video_url", "video_url": {"url": path.resolve().as_uri()}}
    else:
        row = {"video": video_data_url(path)}
        if fps is not None:
            row["fps"] = fps
        if max_frames is not None:
            row["max_frames"] = max_frames
    return row


def openai_video_messages(
    *,
    video_path: Path,
    prompt: str,
    fps: float | None = None,
    max_frames: int | None = None,
    use_file_uri: bool = True,
) -> list[dict[str, Any]]:
    if use_file_uri:
        video_entry: dict[str, Any] = {
            "type": "video_url",
            "video_url": {"url": video_path.resolve().as_uri()},
        }
    else:
        video_entry = {"type": "video_url", "video_url": {"url": video_data_url(video_path)}}
    content: list[dict[str, Any]] = [
        video_entry,
        {"type": "text", "text": prompt},
    ]
    if fps is not None or max_frames is not None:
        # vLLM Qwen multimodal also accepts flat video dict in some paths; keep URI primary.
        pass
    return [{"role": "user", "content": content}]


def extract_openai_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception as exc:
        raise ValueError(
            "OpenAI-compatible response is missing choices[0].message.content"
        ) from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ValueError(
        "OpenAI-compatible message content has unsupported type "
        f"{type(content).__name__}"
    )


class VisionPayloadNotIngestedError(RuntimeError):
    """The API billed no vision tokens, so the judge never saw the frames.

    Deliberately not a ``ValueError``/``CloseAIError``: the per-attempt handler
    treats those as scoreable failures, while a dropped vision payload must abort
    the run instead of yielding text-only answers.
    """


def estimate_text_only_prompt_tokens(prompt: str) -> int:
    """Conservative floor for the text half of the request.

    Four characters per token under-counts CJK, which can only lower the
    ingestion floor; it never fails a request that did carry frames.
    """

    return max(1, len(prompt) // 4)


def require_vision_ingested(
    usage: Mapping[str, Any] | None,
    *,
    case_id: str,
    prompt: str,
    frame_count: int,
    min_tokens_per_frame: int,
) -> int:
    """Fail loud unless billed prompt tokens prove the frames reached the model."""

    prompt_tokens = None if usage is None else usage.get("prompt_tokens")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise VisionPayloadNotIngestedError(
            f"{case_id}: response usage has no positive prompt_tokens "
            f"({prompt_tokens!r}); cannot verify the frames reached the model"
        )
    text_tokens = estimate_text_only_prompt_tokens(prompt)
    floor = text_tokens + frame_count * min_tokens_per_frame
    if prompt_tokens < floor:
        raise VisionPayloadNotIngestedError(
            f"{case_id}: judge billed {prompt_tokens} prompt tokens for {frame_count} "
            f"frames, below the {floor} floor (text≈{text_tokens} + "
            f"{min_tokens_per_frame}/frame). The vision payload was dropped in "
            "transit, so any answer would score the shot plan text, not the video."
        )
    return prompt_tokens
