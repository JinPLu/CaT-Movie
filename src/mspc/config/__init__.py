"""Strict, versioned MSPC configuration contracts."""

from .experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentConfig,
    ExperimentConfigError,
    canonical_json,
    load_experiment_config,
    parse_experiment_mapping,
)

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentConfig",
    "ExperimentConfigError",
    "canonical_json",
    "load_experiment_config",
    "parse_experiment_mapping",
]
