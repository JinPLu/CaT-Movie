"""Unified (section, type) component registry for config, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Mapping

from mspc.config.experiment import ExperimentConfigError


@dataclass(frozen=True)
class ComponentEntry:
    section: str
    type_name: str
    module: str | None = None
    compatible_datasets: frozenset[str] | None = None
    compatible_models: frozenset[str] | None = None
    requires_pipeline_inputs: bool | None = None
    task_paradigm: str | None = None
    prompt_format: str | None = None


class ComponentRegistry:
    """Lazy registry keyed by ``(section, type_name)``."""

    def __init__(self, *, section: str, entries: Mapping[str, ComponentEntry]) -> None:
        self.section = section
        self._entries = dict(entries)

    @property
    def modules(self) -> dict[str, str]:
        return {
            name: entry.module
            for name, entry in self._entries.items()
            if entry.module is not None
        }

    def known_types(self) -> frozenset[str]:
        return frozenset(self._entries)

    def get(self, type_name: str) -> ComponentEntry:
        if not isinstance(type_name, str) or not type_name.strip():
            raise ValueError(f"{self.section} type must be a non-empty string")
        try:
            return self._entries[type_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries))
            raise ValueError(
                f"unknown {self.section} type {type_name!r}; available: {available}"
            ) from exc

    def resolve(self, type_name: str) -> ModuleType:
        entry = self.get(type_name)
        if entry.module is None:
            raise ValueError(f"{self.section} type {type_name!r} has no runtime module")
        return import_module(entry.module)

    def validate_model_dataset_pair(self, model_type: str, dataset_type: str) -> None:
        model = _MODEL.get(model_type)
        dataset = _DATASET.get(dataset_type)
        if model.compatible_datasets is not None and dataset_type not in model.compatible_datasets:
            raise ExperimentConfigError(
                f"model.type={model_type!r} requires dataset.type in "
                f"{sorted(model.compatible_datasets)}; found {dataset_type!r}"
            )
        if dataset.compatible_models is not None and model_type not in dataset.compatible_models:
            raise ExperimentConfigError(
                f"dataset.type={dataset_type!r} requires model.type in "
                f"{sorted(dataset.compatible_models)}; found {model_type!r}"
            )


def validate_model_dataset_compatibility(model_type: str, dataset_type: str) -> None:
    _MODEL.validate_model_dataset_pair(model_type, dataset_type)


_MODEL = ComponentRegistry(
    section="model",
    entries={
        "wan22_diffsynth": ComponentEntry(
            section="model",
            type_name="wan22_diffsynth",
            module="mspc.training.backends.wan22",
            compatible_datasets=frozenset(
                {
                    "multishot_csv",
                    "block_latent_cache",
                    "block_video",
                }
            ),
            requires_pipeline_inputs=True,
            task_paradigm="text_to_multishot_video",
        ),
    },
)

_DATASET = ComponentRegistry(
    section="dataset",
    entries={
        "multishot_csv": ComponentEntry(
            section="dataset",
            type_name="multishot_csv",
            module="mspc.data.multishot",
            compatible_models=frozenset({"wan22_diffsynth"}),
        ),
        # 一行是「一个目标段 ＋ 它前接的 M 张干净参考潜帧」，读逐镜 latent 缓存的切片。
        "block_latent_cache": ComponentEntry(
            section="dataset",
            type_name="block_latent_cache",
            module="mspc.training.block_dataset",
            compatible_models=frozenset({"wan22_diffsynth"}),
        ),
        "block_video": ComponentEntry(
            section="dataset",
            type_name="block_video",
            module="mspc.training.block_dataset",
            compatible_models=frozenset({"wan22_diffsynth"}),
        ),
    },
)

_TRAINER = ComponentRegistry(
    section="trainer",
    entries={
        "sft": ComponentEntry(
            section="trainer",
            type_name="sft",
            module="mspc.training.loop",
        ),
    },
)

_INFERENCE = ComponentRegistry(
    section="inference",
    entries={
        "experiment_launch": ComponentEntry(
            section="inference",
            type_name="experiment_launch",
            module="mspc.inference.wan22",
        ),
    },
)

COMPONENT_REGISTRY: dict[str, ComponentRegistry] = {
    "model": _MODEL,
    "dataset": _DATASET,
    "trainer": _TRAINER,
    "inference": _INFERENCE,
}

MODEL_REGISTRY = _MODEL
DATASET_REGISTRY = _DATASET
TRAINER_REGISTRY = _TRAINER
INFERENCE_REGISTRY = _INFERENCE

@dataclass(frozen=True)
class TrainingComponents:
    model: ModuleType
    dataset: ModuleType
    trainer: ModuleType


def resolve_training_components(config: object) -> TrainingComponents:
    resolved = getattr(config, "resolved", None)
    if resolved is None:
        raise TypeError("config must expose a resolved mapping")
    return TrainingComponents(
        model=MODEL_REGISTRY.resolve(resolved["model"]["type"]),
        dataset=DATASET_REGISTRY.resolve(resolved["dataset"]["type"]),
        trainer=TRAINER_REGISTRY.resolve(resolved["trainer"]),
    )


def validate_trainer_type(trainer: str) -> None:
    if trainer not in _TRAINER.known_types():
        raise ExperimentConfigError(
            f"trainer must be one of {sorted(_TRAINER.known_types())}"
        )
