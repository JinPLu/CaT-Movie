"""真实块fixture的原索引文字审计；受控source标签只验接线，不当逐帧视觉证据。"""
from dataclasses import replace
import json

import pytest

from mspc.data.renderer_block_preflight import inspect_dataset, main
from mspc.training.block_dataset import BlockDataset
from tests.unit.training.test_block_dataset import inputs  # noqa: F401


def metadata_config(inputs):
    return {**inputs, 'target_storage': 'video', 'target_index_path': None, 'target_roots': None,
            'target_mask_weights': None, 'target_latent_geometry': {'channels': 48, 'spatial_stride': 16}}


def test_selected_original_indices_keep_full_dataset_and_do_not_read_pixels_or_latents(inputs, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('metadata preflight touched tensor or decode path')
    monkeypatch.setattr(BlockDataset, '__getitem__', unexpected)
    monkeypatch.setattr(BlockDataset, 'latent_frame_upper_bounds', unexpected)
    monkeypatch.setattr('mspc.training.block_dataset.read_reference_frames', unexpected)
    summary = inspect_dataset(dataset_cfg=metadata_config(inputs), output=tmp_path/'report', tensor_samples=0,
                              sample_indices=[5, 17, 60])
    assert summary['dataset']['sample_budget'] == 61
    assert summary['scope'] == {'dataset_samples': 61, 'sample_indices': [5, 17, 60]}
    assert summary['text_pass']['samples'] == summary['exposure']['samples'] == 3
    assert summary['tensor_pass']['samples'] == summary['reference_io']['output_frames'] == 0
    assert summary['text_spans']['target_rows'] >= 3 and summary['text_spans']['target_spans'] > 0
    assert summary['text_spans']['errors'] == 0
    assert summary['text_pass']['actual_target_token_max'] > 0


def test_source_spans_use_actual_label_reader_and_media_source_frames(inputs, tmp_path, monkeypatch):
    calls = []
    class Labels:
        def __init__(self, config, expected_sample_indices):
            assert config == {'labels_path': 'controlled-frame-labels'}
            assert list(expected_sample_indices) == list(range(61))
        def for_draw(self, index, draw, *, source_frames):
            calls.append((index, list(source_frames)))
            return [{'shot': shot + 1, 'frame': frame, 'source_frame': source, 'status': 'observed',
                     'entities': [{'name': 'Character 1', 'identity_status': 'registered', 'point_cell': None}],
                     'alignment_cells': []}
                    for (shot, frame), source in zip(draw.references, source_frames, strict=True)]
    monkeypatch.setattr('mspc.training.block_dataset.ReferenceEntities', Labels)
    config = metadata_config(inputs)
    probe = BlockDataset(**config)
    indices = [i for i in range(len(probe)) if probe.draw(i).references][:3]
    config.update(reference_entities={'labels_path': 'controlled-frame-labels'},
                  memory_prefix_options={**config['memory_prefix_options'], 'row_policy': 'frame_entities_v1'})
    summary = inspect_dataset(dataset_cfg=config, output=tmp_path/'source', tensor_samples=0, sample_indices=indices)
    assert summary['text_spans']['labelled_reference_rows'] == summary['text_spans']['reference_rows'] > 0
    assert summary['text_spans']['reference_spans'] == 2 * summary['text_spans']['reference_rows']
    assert summary['text_spans']['source_label_statuses'] == {'observed': summary['text_spans']['reference_rows']}
    assert summary['text_spans']['errors'] == 0
    assert all(frames == probe.reference_source_frames(probe.draw(index)) for index, frames in calls)


def test_invalid_actual_target_span_is_reported_with_original_sample_id(inputs, tmp_path, monkeypatch):
    original = BlockDataset.text
    def bad_span(self, index):
        text = original(self, index)
        first, *rest = text.compiled_targets
        span = {**first['entity_spans'][0], 'end': len(first['text']) + 1}
        return replace(text, compiled_targets=({**first, 'entity_spans': [span]}, *rest))
    monkeypatch.setattr(BlockDataset, 'text', bad_span)
    summary = inspect_dataset(dataset_cfg=metadata_config(inputs), output=tmp_path/'bad', tensor_samples=0,
                              sample_indices=[17])
    assert summary['text_spans']['errors'] == 1
    records = [json.loads(line) for line in (tmp_path/'bad/text_span_errors.jsonl').read_text().splitlines()]
    assert records[0]['data_id'] == 17 and records[0]['kind'] == 'target'


@pytest.mark.parametrize('indices', [[], [1, 1], [-1], [61], [True]])
def test_explicit_indices_are_valid_original_dataset_indices(inputs, tmp_path, indices):
    with pytest.raises(ValueError, match='original dataset indices'):
        inspect_dataset(dataset_cfg=metadata_config(inputs), output=tmp_path/'invalid', tensor_samples=0,
                        sample_indices=indices)


def test_cli_accepts_stage_projected_indices_and_zero_tensor_samples(inputs, tmp_path):
    dataset = tmp_path/'dataset.json'
    dataset.write_text(json.dumps(metadata_config(inputs)))
    indices = tmp_path/'sample-indices.json'
    indices.write_text(json.dumps([4, 9]))
    output = tmp_path/'cli'
    assert main(['--dataset-json', str(dataset), '--sample-indices-json', str(indices),
                 '--tensor-samples', '0', '--output', str(output)]) == 0
    summary = json.loads((output/'construction.json').read_text())
    assert summary['scope']['sample_indices'] == [4, 9] and summary['text_pass']['samples'] == 2
