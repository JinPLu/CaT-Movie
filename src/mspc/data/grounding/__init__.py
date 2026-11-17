"""SAM 3.1 grounding spot-check: stratified sample, visibility, review sheets."""

from mspc.data.grounding.sam3_backend import (
    BackendShotResult,
    FrameDetection,
    GroundingBackend,
    SubjectPrompt,
    apply_hotstart_settings,
    split_prompt_batches,
)

__all__ = [
    "BackendShotResult",
    "FrameDetection",
    "GroundingBackend",
    "SubjectPrompt",
    "apply_hotstart_settings",
    "split_prompt_batches",
]
