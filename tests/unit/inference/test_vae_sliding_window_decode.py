"""Sliding-window shot decode in the vendored Wan VAE (MultiShotMaster's ``sliding_window_vae_decode``)."""

from __future__ import annotations

import sys

import pytest
import torch

from tests.support.paths import REPO_ROOT

RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from diffsynth.models.wan_video_vae import WanVideoVAE  # noqa: E402


def _tagging_vae() -> tuple[WanVideoVAE, list[tuple[int, int]]]:
    """A VAE whose window decode tags every RGB frame with (absolute latent index, first-of-window).

    The causal Wan VAE turns a window's first latent frame into one RGB frame and every later
    latent frame into four; the tags record which latent frame (by absolute index, carried in
    channel 0) each output frame came from and whether it was decoded as a window's first frame.
    """

    vae = object.__new__(WanVideoVAE)
    windows: list[tuple[int, int]] = []

    def single_decode(hidden_state: torch.Tensor, device: str) -> torch.Tensor:
        latent_ids = hidden_state[0, 0, :, 0, 0].tolist()
        windows.append((int(latent_ids[0]), len(latent_ids)))
        frames = [(latent_ids[0], 1.0)]
        for latent_id in latent_ids[1:]:
            frames.extend([(latent_id, 0.0)] * 4)
        video = torch.zeros(1, 3, len(frames), 1, 1)
        for index, (latent_id, first) in enumerate(frames):
            video[0, 0, index] = latent_id
            video[0, 1, index] = first
        return video

    vae.single_decode = single_decode  # type: ignore[method-assign]
    return vae, windows


def _latents(total: int) -> torch.Tensor:
    latents = torch.zeros(1, 2, total, 1, 1)
    latents[0, 0, :, 0, 0] = torch.arange(total, dtype=torch.float32)
    return latents


def test_short_shots_decode_in_one_pass() -> None:
    vae, windows = _tagging_vae()
    video = vae.decode(_latents(31), device="cpu")
    assert windows == [(0, 31)]
    assert video.shape[2] == 1 + 30 * 4


@pytest.mark.parametrize("total", [32, 55, 77])
def test_long_shots_decode_in_31_frame_windows_every_24_latents(total: int) -> None:
    vae, windows = _tagging_vae()
    video = vae.decode(_latents(total), device="cpu")[0]
    assert video.shape[1] == 1 + (total - 1) * 4
    starts = [start for start, _ in windows]
    assert starts == list(range(0, total, 24))
    assert all(length == min(31, total - start) for start, length in windows)
    latent_of_frame = video[0, :, 0, 0].tolist()
    first_of_window = video[1, :, 0, 0].tolist()
    # Frame 0 is latent 0; frame f >= 1 is latent ceil(f / 4): the stitched clip reads as one
    # continuous decode, and no later window's one-frame "first latent" leaks into it.
    assert latent_of_frame == [0.0] + [float(-(-f // 4)) for f in range(1, len(latent_of_frame))]
    assert first_of_window == [1.0] + [0.0] * (len(first_of_window) - 1)
