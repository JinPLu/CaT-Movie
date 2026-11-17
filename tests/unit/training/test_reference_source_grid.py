"""真实块元数据的参考时间网格回归；不伪造SAM身份，也不执行视频／VAE。"""
from types import SimpleNamespace

import pytest
import torch

from mspc.core.draws import unit_draw
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA, training_geometry
from mspc.data.shot_targets import planned_row
from mspc.training.block_dataset import BlockDataset
from mspc.training.cache_readiness import cache_stage_plan
from mspc.training.loop import data_order_generator, training_sampler
from tests.unit.training.test_block_dataset import inputs  # Shared real block fixture.


MEDIA720 = {**ANNOTATION_MEDIA, "height": 704, "width": 1280,
            "target_fps": 24, "max_rgb_frames": 512}


class LegacyTargetGridReferences(BlockDataset):
    """固定本次修复前行为，只作为测试对照，不进入生产或回退路径。"""

    def reference_plan(self, row, span, key):
        if self.reference_policy != "random_frame":
            return super().reference_plan(row, span, key)
        if not self.references_enabled or unit_draw("reference_group", *key) < self.memory_options.dropout_p:
            return {}, 0
        frames = {position: list(range(0, row.frame_counts[position], 4))
                  for position in row.usable if not span[0] <= position < span[1]}
        count = (1 + int(unit_draw("reference_count", *key) * min(self.memory_options.frames_max, len(frames)))
                 if frames else 0)
        return frames, count

    def reference_source_frames(self, draw):
        member = self.members[draw.row.case_id]
        scene = {**member["scene"], "shots": member["shots"]}
        grids = {shot: planned_row(scene, shot, media=self.media)["source_frame_indices_by_shot"][0]
                 for shot, _ in draw.references}
        return [grids[shot][frame] for shot, frame in draw.references]


def video_dataset(inputs, *, media, cls=BlockDataset):
    return cls(**{**inputs, "media": media, "target_storage": "video",
        "target_index_path": None, "target_roots": None, "target_mask_weights": None,
        "target_latent_geometry": {"channels": 48, "spatial_stride": 16},
        "segment_capacity": {"max_frames": media["max_rgb_frames"]},
        "memory_prefix_options": {**inputs["memory_prefix_options"],
                                  "attention_policy": "routed_self_only_v1"}})


def sampler_args():
    return SimpleNamespace(max_train_steps=30, num_epochs=1, training_seed=42,
        gradient_accumulation_steps=1, per_device_batch_size=1,
        dataset_num_workers=2, dataloader_prefetch_factor=2,
        dataloader_split_batches=False, dataloader_drop_last=False,
        validation=None, experiment_config={
            "model": {"self_attention_policy": "holocine_first_frame"},
            "training": {"workload_cost": {
                "linear_seconds_per_latent": .01,
                "attention_seconds_per_frame_pair": .003}}})


def test_random_reference_source_frames_are_identical_across_target_fps(inputs):
    old = video_dataset(inputs, media=ANNOTATION_MEDIA)
    new = video_dataset(inputs, media=MEDIA720)
    assert old.reference_grid_media == new.reference_grid_media == ANNOTATION_MEDIA
    assert new.media == MEDIA720
    assert new.metadata_summary["reference_temporal_grid"] == {"target_fps": 16, "vae_temporal_stride": 4}
    assert new.metadata_summary["target_spatial_media"] == {"height": 704, "width": 1280}
    selected = 0
    for index in range(len(old)):
        before, after = old.draw(index), new.draw(index)
        assert before.references == after.references
        assert before.key == after.key
        assert old.reference_source_frames(before) == new.reference_source_frames(after)
        selected += len(after.references)
    assert selected > 0
    member = next(iter(new.members.values()))
    assert training_geometry(member["scene"], media=new.media)["output_size"] == [1280, 704]


def test_original_480_reference_draw_and_source_frames_remain_exact(inputs):
    current = video_dataset(inputs, media=ANNOTATION_MEDIA)
    previous = video_dataset(inputs, media=ANNOTATION_MEDIA, cls=LegacyTargetGridReferences)
    for index in range(len(current)):
        assert current.draw(index) == previous.draw(index)
        assert current.reference_source_frames(current.draw(index)) == previous.reference_source_frames(previous.draw(index))
    assert current.training_workloads(self_attention_policy="holocine_first_frame") == previous.training_workloads(
        self_attention_policy="holocine_first_frame")


def test_720_time_grid_fix_preserves_reference_counts_cost_sampler_and_stage_targets(inputs):
    pytest.importorskip("accelerate")
    corrected = video_dataset(inputs, media=MEDIA720)
    previous = video_dataset(inputs, media=MEDIA720, cls=LegacyTargetGridReferences)
    changed_source_times = 0
    for index in range(len(corrected)):
        after, before = corrected.draw(index), previous.draw(index)
        assert [shot for shot, _ in after.references] == [shot for shot, _ in before.references]
        assert len(after.references) == len(before.references)
        assert after.key == before.key
        changed_source_times += corrected.reference_source_frames(after) != previous.reference_source_frames(before)
    assert changed_source_times > 0
    assert corrected.training_workloads(self_attention_policy="holocine_first_frame") == previous.training_workloads(
        self_attention_policy="holocine_first_frame")
    assert corrected.latent_frame_upper_bounds() == previous.latent_frame_upper_bounds()
    args = sampler_args()
    def order(dataset):
        return list(training_sampler(dataset, generator=data_order_generator(args), bucket_size=2,
                                     experiment_config=args.experiment_config))
    assert order(corrected) == order(previous)
    rng = torch.get_rng_state().clone()
    after = cache_stage_plan(corrected, args, world_size=2, target_step=3)
    before = cache_stage_plan(previous, args, world_size=2, target_step=3)
    assert torch.equal(rng, torch.get_rng_state())
    assert after["rank_sample_indices"] == before["rank_sample_indices"]
    assert after["sample_indices"] == before["sample_indices"]
    assert after["target_keys"] == before["target_keys"]
