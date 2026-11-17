"""SAM 3.1 predictor seam. Tests inject a backend; this module is the only import site."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class SubjectPrompt:
    subject_id: str
    noun_phrase: str
    desc_en: str
    kind: str


@dataclass(frozen=True, slots=True)
class FrameDetection:
    subject_id: str
    box_xyxy: tuple[float, float, float, float] | None
    mask_area_ratio: float | None
    confidence: float | None
    presence: float
    wall_time_s: float


@dataclass(frozen=True, slots=True)
class BackendShotResult:
    detections: tuple[FrameDetection, ...]
    forward_count: int


class GroundingBackend(Protocol):
    def infer_shot(
        self,
        *,
        frames: Sequence[Any],
        prompts: Sequence[SubjectPrompt],
        fps: int,
        hotstart_delay: int,
        hotstart_unmatch_thresh: int,
        objects_per_forward: int,
    ) -> BackendShotResult: ...


def split_prompt_batches(
    prompts: Sequence[SubjectPrompt],
    objects_per_forward: int,
) -> list[tuple[SubjectPrompt, ...]]:
    if type(objects_per_forward) is not int or isinstance(objects_per_forward, bool):
        raise ValueError(
            f"objects_per_forward must be an int, got {type(objects_per_forward).__name__}"
        )
    if objects_per_forward < 1:
        raise ValueError(f"objects_per_forward must be >= 1, got {objects_per_forward}")
    if not prompts:
        return []
    return [
        tuple(prompts[start : start + objects_per_forward])
        for start in range(0, len(prompts), objects_per_forward)
    ]


def apply_hotstart_settings(
    model: Any,
    *,
    hotstart_delay: int,
    hotstart_unmatch_thresh: int,
) -> None:
    if not hasattr(model, "hotstart_delay"):
        raise ValueError("SAM model is missing attribute hotstart_delay")
    if not hasattr(model, "hotstart_unmatch_thresh"):
        raise ValueError("SAM model is missing attribute hotstart_unmatch_thresh")
    model.hotstart_delay = hotstart_delay
    model.hotstart_unmatch_thresh = hotstart_unmatch_thresh


def _require_box(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must be a 4-number xyxy box")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


class Sam3Backend:
    """Loads `build_sam3_predictor` only when constructed. Never imported by unit tests."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        bpe_path: str,
        version: str,
        objects_per_forward: int,
        enable_object_multiplexing: bool,
        use_fa3: bool,
        use_rope_real: bool,
        builder: Any | None = None,
    ) -> None:
        if enable_object_multiplexing is not True:
            raise ValueError("enable_object_multiplexing must be true")
        if type(objects_per_forward) is not int or isinstance(objects_per_forward, bool):
            raise ValueError(
                f"objects_per_forward must be an int, got {type(objects_per_forward).__name__}"
            )
        if objects_per_forward < 1:
            raise ValueError(f"objects_per_forward must be >= 1, got {objects_per_forward}")
        if builder is None:
            from sam3.model_builder import build_sam3_predictor

            builder = build_sam3_predictor
        self._predictor = builder(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            version=version,
            max_num_objects=objects_per_forward,
            multiplex_count=objects_per_forward,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            compile=False,
            warm_up=False,
        )
        self._objects_per_forward = objects_per_forward
        #: 上游构建器给的热启动默认值（SAM 3.1 multiplex：延迟 15 帧、热启动期内未匹配 8 帧即删）。
        self.official_hotstart = (int(self._predictor.model.hotstart_delay),
                                  int(self._predictor.model.hotstart_unmatch_thresh))

    def track_frame_directory(self, *, frame_dir: str, text: str, mask_path: str, hotstart_delay: int,
                              hotstart_unmatch_thresh: int) -> list[dict]:
        """一个文本概念在一个帧目录上的一次完整传播，保留实际实例 ID、mask 和帧号。

        上游的 text add_prompt 会 reset_state，故不同概念必须各自完整传播，
        不能连续 add 多个词后只 propagate 一次。返回框统一为 0..1000 xyxy。
        热启动参数每次显式设置：逐帧跟踪一个镜头用 :attr:`official_hotstart`；只有两三帧的稀疏帧序传 0。
        """
        import numpy as np

        apply_hotstart_settings(self._predictor.model, hotstart_delay=hotstart_delay,
                                hotstart_unmatch_thresh=hotstart_unmatch_thresh)
        session_id = self._predictor.handle_request({
            "type": "start_session", "resource_path": frame_dir,
            "offload_video_to_cpu": True,
        })["session_id"]
        frames, packed_masks = [], {}
        try:
            self._predictor.handle_request({"type": "add_prompt", "session_id": session_id,
                                            "frame_index": 0, "text": text})
            for item in self._predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session_id,
                "propagation_direction": "forward", "start_frame_index": 0,
            }):
                index, out = int(item["frame_index"]), item["outputs"]
                detections = []
                for obj_id, xywh, prob, mask in zip(out["out_obj_ids"], out["out_boxes_xywh"],
                                                   out["out_probs"], out["out_binary_masks"], strict=True):
                    x, y, w, h = map(float, xywh)
                    mask = np.asarray(mask, dtype=bool)
                    key = f"f{index}_o{int(obj_id)}"
                    packed_masks[key] = np.packbits(mask)
                    packed_masks[key + "_shape"] = np.asarray(mask.shape)
                    detections.append({"track_id": int(obj_id), "box": [1000*x, 1000*y, 1000*(x+w), 1000*(y+h)],
                                       "score": float(prob), "area": float(mask.mean()), "mask_key": key})
                frames.append({"frame_index": index, "detections": detections})
            np.savez_compressed(mask_path, **packed_masks)
        finally:
            self._predictor.handle_request({"type": "close_session", "session_id": session_id})
        return frames

    def infer_shot(
        self,
        *,
        frames: Sequence[Any],
        prompts: Sequence[SubjectPrompt],
        fps: int,
        hotstart_delay: int,
        hotstart_unmatch_thresh: int,
        objects_per_forward: int,
    ) -> BackendShotResult:
        del fps
        if objects_per_forward != self._objects_per_forward:
            raise ValueError(
                "objects_per_forward does not match the constructed multiplex bucket: "
                f"{objects_per_forward} != {self._objects_per_forward}"
            )
        apply_hotstart_settings(
            self._predictor.model,
            hotstart_delay=hotstart_delay,
            hotstart_unmatch_thresh=hotstart_unmatch_thresh,
        )
        detections: list[FrameDetection] = []
        forward_count = 0
        for batch in split_prompt_batches(prompts, objects_per_forward):
            detections.extend(
                self._infer_batch(
                    frames=frames,
                    prompts=batch,
                )
            )
            forward_count += 1
        return BackendShotResult(detections=tuple(detections), forward_count=forward_count)

    def _infer_batch(
        self,
        *,
        frames: Sequence[Any],
        prompts: Sequence[SubjectPrompt],
    ) -> list[FrameDetection]:
        if not frames:
            raise ValueError("frames must be a non-empty sequence")
        session_dir = _write_frame_dir(frames)
        started = self._predictor.handle_request(
            {"type": "start_session", "resource_path": str(session_dir)}
        )
        if not isinstance(started, Mapping) or "session_id" not in started:
            raise ValueError("start_session must return an object with session_id")
        session_id = started["session_id"]
        if type(session_id) is not str or not session_id:
            raise ValueError("session_id must be a non-empty string")
        for prompt in prompts:
            self._predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt.noun_phrase,
                    "obj_id": prompt.subject_id,
                }
            )
        stream = self._predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        )
        detections: list[FrameDetection] = []
        height, width = _frame_hw(frames[0])
        for frame_index, out in enumerate(stream):
            started_at = time.perf_counter()
            if not isinstance(out, Mapping):
                raise ValueError(f"propagate_in_video[{frame_index}] must be an object")
            per_subject = _detections_from_output(
                out,
                prompts=prompts,
                height=height,
                width=width,
                wall_time_s=0.0,
            )
            wall_time_s = time.perf_counter() - started_at
            share = wall_time_s / len(per_subject)
            for item in per_subject:
                detections.append(
                    FrameDetection(
                        subject_id=item.subject_id,
                        box_xyxy=item.box_xyxy,
                        mask_area_ratio=item.mask_area_ratio,
                        confidence=item.confidence,
                        presence=item.presence,
                        wall_time_s=share,
                    )
                )
        self._predictor.handle_request({"type": "close_session", "session_id": session_id})
        return detections


def _frame_hw(frame: Any) -> tuple[int, int]:
    array = _as_uint8_rgb(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 array")
    return int(array.shape[0]), int(array.shape[1])


def _as_uint8_rgb(frame: Any) -> Any:
    import numpy as np
    from PIL import Image

    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 array")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


def _write_frame_dir(frames: Sequence[Any]) -> Any:
    import tempfile
    from pathlib import Path
    from PIL import Image

    root = Path(tempfile.mkdtemp(prefix="sam3_spotcheck_"))
    for index, frame in enumerate(frames):
        image = Image.fromarray(_as_uint8_rgb(frame), mode="RGB")
        image.save(root / f"{index:05d}.jpg", format="JPEG", quality=95)
    return root


def _detections_from_output(
    out: Mapping[str, Any],
    *,
    prompts: Sequence[SubjectPrompt],
    height: int,
    width: int,
    wall_time_s: float,
) -> list[FrameDetection]:
    obj_ids = out.get("out_obj_ids", out.get("obj_ids"))
    boxes = out.get("out_boxes_xyxy", out.get("boxes"))
    masks = out.get("out_binary_masks", out.get("masks"))
    scores = out.get("out_probs", out.get("scores"))
    presence = out.get("out_presence", out.get("presence"))
    by_id: dict[str, dict[str, Any]] = {}
    if obj_ids is None:
        if len(prompts) != 1:
            raise ValueError("propagate output is missing out_obj_ids")
        by_id[prompts[0].subject_id] = {
            "box": None if boxes is None else boxes[0] if _is_seq(boxes) else boxes,
            "mask": None if masks is None else masks[0] if _is_seq(masks) else masks,
            "score": None if scores is None else scores[0] if _is_seq(scores) else scores,
            "presence": None if presence is None else presence[0] if _is_seq(presence) else presence,
        }
    else:
        if not _is_seq(obj_ids):
            raise ValueError("out_obj_ids must be a sequence")
        for index, obj_id in enumerate(obj_ids):
            key = str(obj_id)
            by_id[key] = {
                "box": _index_or_none(boxes, index),
                "mask": _index_or_none(masks, index),
                "score": _index_or_none(scores, index),
                "presence": _index_or_none(presence, index),
            }
    detections: list[FrameDetection] = []
    area = float(height * width)
    if area <= 0:
        raise ValueError("frame area must be positive")
    for prompt in prompts:
        payload = by_id.get(prompt.subject_id)
        if payload is None:
            detections.append(
                FrameDetection(
                    subject_id=prompt.subject_id,
                    box_xyxy=None,
                    mask_area_ratio=None,
                    confidence=None,
                    presence=0.0,
                    wall_time_s=wall_time_s,
                )
            )
            continue
        box = payload["box"]
        mask = payload["mask"]
        score = payload["score"]
        pres = payload["presence"]
        mask_area_ratio = None
        if mask is not None:
            import numpy as np

            mask_array = np.asarray(mask)
            mask_area_ratio = float(mask_array.astype(bool).sum()) / area
        detections.append(
            FrameDetection(
                subject_id=prompt.subject_id,
                box_xyxy=None if box is None else _require_box(box, f"{prompt.subject_id}.box"),
                mask_area_ratio=mask_area_ratio,
                confidence=None if score is None else float(score),
                presence=0.0 if pres is None else float(pres),
                wall_time_s=wall_time_s,
            )
        )
    return detections


def _is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple)) or (
        hasattr(value, "shape") and getattr(value, "ndim", 0) >= 1
    )


def _index_or_none(value: Any, index: int) -> Any:
    if value is None:
        return None
    if _is_seq(value):
        if hasattr(value, "__len__") and index >= len(value):
            return None
        return value[index]
    return value
