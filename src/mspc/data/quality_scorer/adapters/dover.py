"""DOVER named-head adapter.

The upstream ``DOVER.forward`` returns a *list* aligned with the iteration order
of the ``vclips`` dict it was handed, and the two official scripts disagree
about what that order means. This adapter never indexes that list: it feeds one
named view per call, so ``aesthetic`` and ``technical`` are read by name.

Sampling and spatial view decomposition are the upstream ones, imported from the
pinned checkout. Only the *source* of frames differs: a shot segment, not a
whole file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from mspc.data.quality_scorer.adapters.media import (
    ShotDecodeError,
    ShotLocator,
    clamp_shot_to_file,
    decode_local_frames,
    require_positive_int,
    require_positive_int_env,
)


NAMED_HEADS: tuple[str, ...] = ("aesthetic", "technical")
# The demo's normalisation constants, in the demo's channel order.
PIXEL_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
PIXEL_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)
EVAL_DATASET_KEY = "val-l1080p"
DOVER_BATCH_ENV = "QUALITY_SCORER_DOVER_BATCH"
DEFAULT_DOVER_BATCH = 4


def require_dover_batch_size(env: Any = None, *, batch_size: int | None = None) -> int:
    if batch_size is not None:
        return require_positive_int(batch_size, "dover.batch_size")
    mapping = os.environ if env is None else env
    raw = mapping.get(DOVER_BATCH_ENV)
    if raw is None or raw == "":
        return DEFAULT_DOVER_BATCH
    return require_positive_int_env(DOVER_BATCH_ENV, mapping)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def sampled_frame_record(
    locator: ShotLocator, local_by_view: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    """Frame provenance for one shot: per view, plus the order they were fed in.

    ``loop_padding`` is decided per view because that is where it happens — the
    upstream sampler wraps with ``np.mod`` when the shot is shorter than a view
    needs. Two views landing on the same frames of a normal-length shot is not
    padding, which is why the views are kept apart instead of only concatenated.
    """

    missing = [name for name in NAMED_HEADS if name not in local_by_view]
    if missing:
        raise ValueError(f"sampled frames are missing named views: {missing}")
    unknown = sorted(set(local_by_view) - set(NAMED_HEADS))
    if unknown:
        raise ValueError(f"sampled frames carry unknown named views: {unknown}")
    by_view = {
        name: [locator.absolute_index(int(index)) for index in local_by_view[name]]
        for name in NAMED_HEADS
    }
    return {
        "sampled_frame_indices": [index for name in NAMED_HEADS for index in by_view[name]],
        "sampled_frame_indices_by_view": by_view,
        "loop_padding": any(len(set(frames)) < len(frames) for frames in by_view.values()),
    }


def shot_seed(base_seed: int, case_id: str, shot_index: int) -> int:
    """Deterministic per-shot seed: upstream sampling draws random offsets."""

    digest = hashlib.sha256(f"{base_seed}:{case_id}:{shot_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class DoverSettings:
    source_root: Path
    weight_path: Path
    eval_yaml: Path
    seed: int
    device: str
    declared_sampling: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedDoverShot:
    locator: ShotLocator
    local_by_view: Mapping[str, Sequence[int]]
    frames_by_view: Mapping[str, Any]
    width: int
    height: int


class DoverBackend:
    """Loads the pinned DOVER checkout and scores one shot at a time."""

    def __init__(self, settings: DoverSettings) -> None:
        self._settings = settings
        self._model: Any = None
        self._samplers: dict[str, Any] = {}
        self._sample_types: dict[str, Any] = {}
        self._get_single_view: Any = None
        self._torch: Any = None

    def _import_upstream(self) -> tuple[Any, Any, Any]:
        root = str(self._settings.source_root)
        if not (self._settings.source_root / "dover").is_dir():
            raise ShotDecodeError(f"DOVER checkout is missing a dover package: {root}")
        if root not in sys.path:
            sys.path.insert(0, root)
        from dover.datasets import UnifiedFrameSampler, get_single_view  # type: ignore
        from dover.models import DOVER  # type: ignore

        return UnifiedFrameSampler, get_single_view, DOVER

    def _verify_declared_sampling(self, sample_types: Mapping[str, Any]) -> None:
        """The committed config and the pinned eval YAML must describe one recipe."""

        declared = self._settings.declared_sampling
        aesthetic = sample_types["aesthetic"]
        technical = sample_types["technical"]
        expected = {
            "aesthetic.size_h": (aesthetic["size_h"], declared["aesthetic"]["resize"]),
            "aesthetic.size_w": (aesthetic["size_w"], declared["aesthetic"]["resize"]),
            "aesthetic.clip_len": (aesthetic["clip_len"], declared["aesthetic"]["frames"]),
            "aesthetic.frame_interval": (
                aesthetic["frame_interval"],
                declared["aesthetic"]["interval"],
            ),
            "aesthetic.num_clips": (aesthetic["num_clips"], declared["aesthetic"]["clips"]),
            "technical.fragments_h": (technical["fragments_h"], 7),
            "technical.fragments_w": (technical["fragments_w"], 7),
            "technical.clip_len": (technical["clip_len"], declared["technical"]["frames"]),
            "technical.frame_interval": (
                technical["frame_interval"],
                declared["technical"]["interval"],
            ),
            "technical.num_clips": (technical["num_clips"], declared["technical"]["clips"]),
        }
        for label, (found, want) in expected.items():
            if int(found) != int(want):
                raise ValueError(
                    f"dover eval yaml {label}={found} disagrees with the committed "
                    f"config value {want}"
                )

    def load_preprocessor(self) -> None:
        """CPU workers load only the official sampler and view decomposition."""
        if self._get_single_view is not None:
            return
        import torch  # noqa: PLC0415 - deliberately lazy: CPU orchestration must not import torch
        import yaml  # noqa: PLC0415

        sampler_cls, get_single_view, _ = self._import_upstream()
        with self._settings.eval_yaml.open("r", encoding="utf-8") as handle:
            opt = yaml.safe_load(handle)
        dopt = opt["data"][EVAL_DATASET_KEY]["args"]
        sample_types = dopt["sample_types"]
        self._verify_declared_sampling(sample_types)
        missing = [name for name in NAMED_HEADS if name not in sample_types]
        if missing:
            raise ValueError(f"dover eval yaml is missing named views: {missing}")
        samplers: dict[str, Any] = {}
        for name, sopt in sample_types.items():
            if "t_frag" not in sopt:
                samplers[name] = sampler_cls(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                samplers[name] = sampler_cls(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
        self._torch = torch
        self._samplers = samplers
        self._sample_types = sample_types
        self._get_single_view = get_single_view

    def load(self) -> None:
        if self._model is not None:
            return
        self.load_preprocessor()
        import yaml

        torch = self._torch
        _, _, dover_cls = self._import_upstream()
        with self._settings.eval_yaml.open("r", encoding="utf-8") as handle:
            opt = yaml.safe_load(handle)
        model = dover_cls(**opt["model"]["args"])
        state = torch.load(str(self._settings.weight_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(self._settings.device)
        model.eval()
        self._model = model

    def sample_local_indices(self, locator: ShotLocator) -> dict[str, list[int]]:
        """Run the pinned samplers. Seeds are reset per view, matching upstream."""

        self.load_preprocessor()
        torch = self._torch
        import numpy as np  # noqa: PLC0415

        seed = shot_seed(self._settings.seed, locator.case_id, locator.shot_index)
        frame_count = locator.frame_count
        local_by_view: dict[str, list[int]] = {}
        for name in NAMED_HEADS:
            np.random.seed(seed)
            torch.manual_seed(seed)
            local = [int(index) for index in self._samplers[name](frame_count)]
            if not local:
                raise ShotDecodeError(
                    f"{locator.case_id}#{locator.shot_index}: upstream sampler produced no frames"
                )
            local_by_view[name] = local
        return local_by_view

    def prepare_shot(self, reader: Any, locator: ShotLocator) -> PreparedDoverShot:
        """Decode both named views in one pass. The official DOVER path does this."""

        self.load_preprocessor()
        clamp_shot_to_file(locator, reader)
        local_by_view = self.sample_local_indices(locator)
        combined = [index for name in NAMED_HEADS for index in local_by_view[name]]
        frames = decode_local_frames(reader, locator, combined)
        height, width = int(frames.shape[1]), int(frames.shape[2])
        offset = 0
        frames_by_view: dict[str, Any] = {}
        for name in NAMED_HEADS:
            count = len(local_by_view[name])
            frames_by_view[name] = frames[offset : offset + count]
            offset += count
        return PreparedDoverShot(
            locator=locator,
            local_by_view=local_by_view,
            frames_by_view=frames_by_view,
            width=width,
            height=height,
        )

    def prepared_from_frames(
        self,
        locator: ShotLocator,
        local_by_view: Mapping[str, Sequence[int]],
        frames_by_view: Mapping[str, Any],
        *,
        width: int,
        height: int,
    ) -> PreparedDoverShot:
        return PreparedDoverShot(
            locator=locator,
            local_by_view=local_by_view,
            frames_by_view=frames_by_view,
            width=width,
            height=height,
        )

    def score_from_frames(
        self,
        locator: ShotLocator,
        local_by_view: Mapping[str, Sequence[int]],
        frames_by_view: Mapping[str, Any],
        *,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        return self.score_prepared(
            self.prepared_from_frames(
                locator,
                local_by_view,
                frames_by_view,
                width=width,
                height=height,
            )
        )

    def _view_tensor(self, prepared: PreparedDoverShot, name: str, device: str | None = None) -> Any:
        """``device`` 给出时在该设备上做视图分解（同一套上游算子）；帧可以是 numpy 或已在内存里的 uint8 张量。"""
        torch = self._torch
        import numpy as np  # noqa: PLC0415

        locator = prepared.locator
        seed = shot_seed(self._settings.seed, locator.case_id, locator.shot_index)
        frames = prepared.frames_by_view[name]
        # Upstream `spatial_temporal_view_decomposition` hands the *raw* decoded
        # tensor to `get_single_view` and only reaches float later, when the
        # `- mean` below (a FloatTensor) promotes it. Where the two orders are
        # equivalent we follow upstream; where they are not, we keep the cast
        # that matches the interpolation upstream's own tensor receives.
        #   technical: `get_single_view` is a fragment gather — pure indexing
        #     and assembly, so casting after decomposition was measured
        #     bit-for-bit equal to casting before. Follow upstream: cast after.
        #   aesthetic: `get_single_view` runs torchvision `Resize`, whose uint8
        #     and float paths are not equivalent — reordering was measured at
        #     max abs diff 0.5 with 98.45% of elements differing. Cast before,
        #     so `Resize` sees the same float tensor it does upstream.
        cast_before_decomposition = name == "aesthetic"
        stacked = frames if torch.is_tensor(frames) else torch.from_numpy(np.ascontiguousarray(frames))
        if device is not None:
            stacked = stacked.to(device)
        stacked = stacked.permute(3, 0, 1, 2)
        if cast_before_decomposition:
            stacked = stacked.float()
        torch.manual_seed(seed)
        view = self._get_single_view(stacked, name, **self._sample_types[name])
        if not cast_before_decomposition:
            view = view.float()
        num_clips = int(self._sample_types[name].get("num_clips", 1))
        mean = torch.tensor(list(PIXEL_MEAN), dtype=torch.float32, device=view.device)
        std = torch.tensor(list(PIXEL_STD), dtype=torch.float32, device=view.device)
        return (
            ((view.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(view.shape[0], num_clips, -1, *view.shape[2:])
            .transpose(0, 1)
        )

    def prepare_views(self, prepared: PreparedDoverShot, names: Sequence[str] = NAMED_HEADS, device: str | None = None) -> dict[str, Any]:
        """Return the named views. Default: all heads as CPU tensors; ``device`` runs the decomposition there instead."""
        self.load_preprocessor()
        return {name: self._view_tensor(prepared, name, device) for name in names}

    def _payload_from_view_scores(
        self, prepared: PreparedDoverShot, results: Mapping[str, float]
    ) -> dict[str, Any]:
        locator = prepared.locator
        for name, value in results.items():
            if not math.isfinite(value):
                raise ShotDecodeError(
                    f"{locator.case_id}#{locator.shot_index}: dover {name} score is not finite"
                )
        if set(results) != set(NAMED_HEADS):  # 只打了部分头：只记这些头的取帧与分数
            payload: dict[str, Any] = {
                "sampled_frame_indices_by_view": {
                    name: [locator.absolute_index(i) for i in prepared.local_by_view[name]] for name in results
                },
                "width": prepared.width,
                "height": prepared.height,
            }
            for name, value in results.items():
                payload[f"{name}_logit"] = value
                payload[f"{name}_sigmoid"] = _sigmoid(value)
            return payload
        return {
            **sampled_frame_record(locator, prepared.local_by_view),
            "width": prepared.width,
            "height": prepared.height,
            "aesthetic_logit": results["aesthetic"],
            "aesthetic_sigmoid": _sigmoid(results["aesthetic"]),
            "technical_logit": results["technical"],
            "technical_sigmoid": _sigmoid(results["technical"]),
        }

    def score_prepared_batch(
        self,
        prepared: Sequence[PreparedDoverShot],
        *,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score shots by stacking the same named view along the model's batch axis."""

        self.load()
        chunk = require_dover_batch_size(batch_size=batch_size)
        if not prepared:
            return []
        rows: list[dict[str, Any]] = []
        for start in range(0, len(prepared), chunk):
            group = list(prepared[start : start + chunk])
            rows.extend(self.score_view_batch(group, [self.prepare_views(item) for item in group]))
        return rows

    def score_view_batch(
        self, prepared: Sequence[PreparedDoverShot], views: Sequence[Mapping[str, Any]],
        names: Sequence[str] = NAMED_HEADS,
    ) -> list[dict[str, Any]]:
        """Same named-head reduction for serial, batched, and CPU-prefetched inputs; ``names`` limits which heads run."""
        self.load()
        if len(prepared) != len(views) or not prepared:
            raise ValueError("nonempty aligned prepared shots and views required")
        group = list(prepared)
        with self._torch.inference_mode():
            per_shot: list[dict[str, float]] = [{} for _ in group]
            for name in names:
                tensors = [item[name] for item in views]
                stacked = self._torch.cat(tensors, dim=0).to(self._settings.device)
                scores = self._model({name: stacked})
                if len(scores) != 1:
                    raise ValueError(
                        f"dover returned {len(scores)} scores for the single named view {name!r}"
                    )
                # Official reduction is mean over the whole head tensor
                # (aesthetic is 28x28, not one logit per clip). Stacking
                # shots on dim=0 scales that tensor; split in proportion
                # to each shot's leading dim, then mean — same as serial
                # scores[0].mean().
                flat = scores[0].reshape(-1)
                lead = int(stacked.shape[0])
                if lead < 1 or int(flat.numel()) % lead != 0:
                    raise ValueError(
                        f"dover returned {int(flat.numel())} values for "
                        f"stacked {name} views with leading dim {lead}"
                    )
                unit = int(flat.numel()) // lead
                offset = 0
                for index, tensor in enumerate(tensors):
                    width = int(tensor.shape[0]) * unit
                    part = flat[offset : offset + width]
                    if int(part.numel()) != width:
                        raise ValueError(
                            f"dover returned {int(flat.numel())} values for "
                            f"stacked {name} views totalling "
                            f"{sum(int(item.shape[0]) for item in tensors)} "
                            f"leading units"
                        )
                    per_shot[index][name] = float(part.mean().item())
                    offset += width
                if offset != int(flat.numel()):
                    raise ValueError(
                        f"dover returned {int(flat.numel())} values for "
                        f"stacked {name} views totalling {offset} units"
                    )
            return [
                self._payload_from_view_scores(item, scores)
                for item, scores in zip(group, per_shot)
            ]

    def score_prepared(self, prepared: PreparedDoverShot) -> dict[str, Any]:
        return self.score_prepared_batch([prepared], batch_size=1)[0]

    def score_shot(self, reader: Any, locator: ShotLocator) -> dict[str, Any]:
        return self.score_prepared(self.prepare_shot(reader, locator))
