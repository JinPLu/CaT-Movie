"""源镜头区间的镜界修订：TransNetV2 逐镜检测切点，修首尾、剔镜中。

CI-VID 镜头文件开头常带原片前一镜的残帧，Cine250K 镜尾常混入下一镜 1–2 帧。
区间生产端在调用帧规划器之前，用本模块修订每个源镜头区间：

- 切点来源只有 TransNetV2（``scene_shot_manifest.transnet_cuts``，阈值 0.5）。10-09 在前
  2500 候选上与 ContentDetector（阈值 27）并排比过：看帧确认的 17 窗两者都命中 13 窗，
  ContentDetector 独有的镜中切点抽看约 3/4 是运动、手挡镜头之类误报，会多剔约 280 窗；
- 镜头开头 0.5 秒内的切点把起点挪到其中最后一个，结尾 0.5 秒内的切点把终点挪到其中第一个；
- 其余位置的切点说明镜中有切换（含叠化、淡入淡出），该镜所在的窗整窗剔除。

切点一律是源文件解码帧号，记为进入新镜的那一帧（半开区间 ``[start, end)`` 的约定）。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from mspc.data.scene_shot_manifest import TRANSNET_THRESHOLD, transnet_cuts


SHOT_BOUNDARY_REVISION_VERSION = "shot_boundary_revision_v1"
BOUNDARY_EDGE_SECONDS = 0.5
# 检测解码网格：decord 按 256x144 解码再 INTER_AREA 缩到 96x54，TransNetV2 引擎自己再缩到
# 48x27。10-09 前 2500 候选的实测读数就是这条网格上的结果。
DECODE_WIDTH = 256
DECODE_HEIGHT = 144
DETECT_WIDTH = 96
DETECT_HEIGHT = 54
# 多个镜头拼成一次 TransNetV2 调用时，每镜两侧各复制首/末帧这么多份；
# 等于 ``_TransNetEngine.predict`` 自己的首尾补帧，使每镜只看到自身内容。
TRANSNET_ISOLATION_PAD = 25

BOUNDARY_DETECTOR = "transnetv2"


class ShotBoundaryRejected(ValueError):
    """镜中有切点或修订后过短：该镜所在窗整窗剔除。``reason`` 是可计数的原因码。"""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class TransNetPredictor(Protocol):
    def predict(self, frames: Sequence[Any]) -> list[float]: ...


def _cut_list(values: Any, *, start: int, end: int, label: str) -> list[int]:
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"{label} must be a list of ints")
    if values != sorted(set(values)):
        raise ValueError(f"{label} must be strictly increasing")
    if any(not start < value < end for value in values):
        raise ValueError(f"{label} {values} must lie strictly inside segment [{start}, {end})")
    return list(values)


def partition_boundary_cuts(*, start: int, end: int, fps: float, cuts: list[int], edge_seconds: float):
    """唯一首尾并界规则；镜内切点交给调用者判定。"""
    cuts = _cut_list(cuts, start=start, end=end, label="transnet_cuts")
    edge = edge_seconds * fps
    head = [cut for cut in cuts if cut - start <= edge]
    tail = [cut for cut in cuts if cut not in head and end - cut <= edge]
    mid = [cut for cut in cuts if cut not in head and cut not in tail]
    return [head[-1] if head else start, tail[0] if tail else end], mid


def revise_shot_boundary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """按一镜的切点证据修订区间；镜中有切点则抛 :class:`ShotBoundaryRejected`。"""

    if evidence.get("version") != SHOT_BOUNDARY_REVISION_VERSION:
        raise ValueError(f"shot cut evidence version must be {SHOT_BOUNDARY_REVISION_VERSION}")
    segment = evidence["source_shot_segment"]
    if (
        not isinstance(segment, list)
        or len(segment) != 2
        or any(type(value) is not int for value in segment)
        or not 0 <= segment[0] < segment[1]
    ):
        raise ValueError(f"source_shot_segment must be [start, end) ints, got {segment!r}")
    start, end = segment
    fps = evidence["source_fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not fps > 0:
        raise ValueError("source_fps must be positive")
    cuts = _cut_list(evidence["transnet_cuts"], start=start, end=end, label="transnet_cuts")
    (revised_start, revised_end), mid = partition_boundary_cuts(
        start=start, end=end, fps=float(fps), cuts=cuts, edge_seconds=BOUNDARY_EDGE_SECONDS)
    if mid:
        raise ShotBoundaryRejected(
            "mid_shot_cut",
            f"segment [{start}, {end}) has {BOUNDARY_DETECTOR} cuts {mid} outside the "
            f"{BOUNDARY_EDGE_SECONDS:g}s head/tail bands",
        )
    return {
        "version": SHOT_BOUNDARY_REVISION_VERSION,
        "source_shot_segment": [start, end],
        "revised_shot_segment": [revised_start, revised_end],
        "trimmed_head_frames": revised_start - start,
        "trimmed_tail_frames": end - revised_end,
        "transnet_cuts": cuts,
    }


def decode_shot_frames(path: str, start: int, end: int) -> np.ndarray:
    """解码 ``[start, end)``，返回 RGB uint8 ``(n, 54, 96, 3)``；帧号与训练 reader 同为 decord 帧号。"""

    import cv2
    import decord

    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError(f"{path}: invalid shot segment [{start}, {end})")
    reader = decord.VideoReader(path, width=DECODE_WIDTH, height=DECODE_HEIGHT, num_threads=1)
    if end > len(reader):
        raise ValueError(f"{path}: segment end {end} exceeds decoded frames {len(reader)}")
    rgb = reader.get_batch(list(range(start, end))).asnumpy()
    return np.stack(
        [cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT), interpolation=cv2.INTER_AREA) for frame in rgb]
    )


def transnet_cuts_for_shots(frames_by_shot: Sequence[np.ndarray], *, starts: Sequence[int], engine: TransNetPredictor) -> list[list[int]]:
    """一次 TransNetV2 调用覆盖多镜；每镜首尾各补 ``TRANSNET_ISOLATION_PAD`` 份首/末帧隔开。"""

    pieces: list[np.ndarray] = []
    offsets: list[int] = []
    cursor = 0
    for frames in frames_by_shot:
        pad = TRANSNET_ISOLATION_PAD
        pieces.extend([np.repeat(frames[:1], pad, axis=0), frames, np.repeat(frames[-1:], pad, axis=0)])
        offsets.append(cursor + pad)
        cursor += len(frames) + 2 * pad
    probabilities = engine.predict(np.concatenate(pieces))
    if len(probabilities) != cursor:
        raise RuntimeError(f"TransNetV2 returned {len(probabilities)} probabilities for {cursor} frames")
    cuts: list[list[int]] = []
    for frames, offset, start in zip(frames_by_shot, offsets, starts, strict=True):
        boundaries, _peaks = transnet_cuts(probabilities[offset:offset + len(frames)], len(frames))
        cuts.append([start + boundary for boundary in boundaries])
    return cuts


def _evidence(path: str, segment: Sequence[int], fps: float, transnet: list[int]) -> dict[str, Any]:
    return {
        "version": SHOT_BOUNDARY_REVISION_VERSION,
        "source_video_path": path,
        "source_shot_segment": [int(segment[0]), int(segment[1])],
        "source_fps": float(fps),
        "transnet_cuts": transnet,
        "transnet_threshold": TRANSNET_THRESHOLD,
    }


class ShotCutDetector:
    """生产端检测：逐镜解码，多条记录的镜头共用一次 TransNetV2 调用。

    TransNetV2 跑在不装 torch 的独立解释器里，每次调用都要重新加载 TensorFlow，
    所以按记录批量调用；某条记录解码失败只让该条返回异常，不拖垮同批其余记录。
    """

    def __init__(self, engine: TransNetPredictor):
        self.engine = engine

    @classmethod
    def from_transnet_resource(cls, resource_root: Any, device: str) -> "ShotCutDetector":
        from pathlib import Path

        from mspc.evaluation.renderer_qualification_official_backend import _SubprocessTransNetEngine

        return cls(_SubprocessTransNetEngine(Path(resource_root), device))

    def detect_many(self, records: Sequence[Mapping[str, Sequence[Any]]]) -> list[list[dict[str, Any]] | Exception]:
        """``records`` 每条含 ``source_video_paths/source_shot_segments/source_fps_by_shot``。"""

        decoded: list[list[np.ndarray] | Exception] = []
        for record in records:
            try:
                decoded.append([
                    decode_shot_frames(path, int(segment[0]), int(segment[1]))
                    for path, segment in zip(record["source_video_paths"], record["source_shot_segments"], strict=True)
                ])
            except (RuntimeError, ValueError, OSError) as error:
                decoded.append(error)
        flat = [frames for item in decoded if not isinstance(item, Exception) for frames in item]
        starts = [
            int(segment[0])
            for record, item in zip(records, decoded, strict=True)
            if not isinstance(item, Exception)
            for segment in record["source_shot_segments"]
        ]
        transnet = iter(transnet_cuts_for_shots(flat, starts=starts, engine=self.engine) if flat else [])
        results: list[list[dict[str, Any]] | Exception] = []
        for record, item in zip(records, decoded, strict=True):
            if isinstance(item, Exception):
                results.append(item)
                continue
            results.append([
                _evidence(path, segment, fps, next(transnet))
                for path, segment, fps in zip(
                    record["source_video_paths"], record["source_shot_segments"], record["source_fps_by_shot"],
                    strict=True,
                )
            ])
        return results

    def __call__(
        self,
        *,
        source_video_paths: Sequence[str],
        source_shot_segments: Sequence[Sequence[int]],
        source_fps_by_shot: Sequence[float],
    ) -> list[dict[str, Any]]:
        (result,) = self.detect_many([{
            "source_video_paths": source_video_paths,
            "source_shot_segments": source_shot_segments,
            "source_fps_by_shot": source_fps_by_shot,
        }])
        if isinstance(result, Exception):
            raise result
        return result
