"""真实块/v18文字的 CPU reader 测试；序号张量与由真实取帧计划组装的缓存记录只核布局，不代表真实编码验收。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from types import SimpleNamespace
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.core.draws import unit_draw
from mspc.data.shot_targets import planned_row, latent_frames, shot_key
from mspc.data.shot_text import reference_text
from mspc.data.renderer_block_preflight import inspect_dataset
from mspc.training.block_dataset import BlockDataset, BlockRow, BlockWindowSampler, pack_span
from tests.support.memory_prefix import FULL_WINDOW_CONFIG, NOISE_CONFIG, OFFSET_CONFIG
from tests.support.paths import REPO_ROOT

FIXTURE = REPO_ROOT / 'tests/fixtures/sampling_ledger'


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class ControlledTargetMasks:
    """目标 SAM 权重的受控 reader；参考取帧不读取 SAM。"""

    def __init__(self, *, index_path, root, weighting):
        self.members = {member['case_id']: member for member in lines(Path(index_path))}

    def record_for(self, member, position, *, latent_spatial_shape):
        return {'shot': position + 1}

    def read(self, member, position, *, latent_spatial_shape):
        n = latent_frames(member['shots'][position]['frames'])
        return (torch.ones((1, 1, n, *latent_spatial_shape)),
                {'shot': position + 1, 'valid_mask_frames': [True] * n, 'unknown_mask_frames': []})


def write(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def materialize(tmp_path, *, vae_by_origin=None, implementation_by_origin=None, latent_geometry=(48, 30, 52),
                target_origins=('first_batch_window', 'shot')):
    channels, height, width = latent_geometry
    block = lines(FIXTURE / 'blocks.jsonl')[0]
    annotation = lines(FIXTURE / 'block-annotations.jsonl')[0]['annotation']
    scene = block['scene']
    shots = [{**shot, 'shot': i + 1, 'frames': shot['render_frames'], 'usable': True, 'unusable_reasons': []}
             for i, shot in enumerate(scene['shots'])]
    member = {'schema_version': 1, 'key': block['key'], 'case_id': block['case_id'], 'annotation': annotation,
              'scene': {k:v for k,v in scene.items() if k != 'shots'}, 'shots': shots}
    write(tmp_path / 'members.jsonl', [member])
    roots = {origin: tmp_path / origin for origin in dict.fromkeys(target_origins)}
    for root in roots.values(): root.mkdir()
    records, index, refs = {origin: [] for origin in roots}, [], []
    for i, shot in enumerate(shots):
        origin = target_origins[i % len(target_origins)]
        vae = (vae_by_origin[origin] if vae_by_origin is not None else
               {'path': '/controlled-test/Wan2.2_VAE.pth', 'revision': 'controlled-test'})
        planned = planned_row(scene, i, media=ANNOTATION_MEDIA)
        n = latent_frames(shot['frames'])
        # 首批物理文件前缀放一镜的区间，确保 consumer 不把 absolute latent_start 忘掉。
        offset = 2 if origin == 'first_batch_window' else 0
        provenance = deepcopy(json.loads((REPO_ROOT / 'tests/unit/data/fixtures/shot_targets_real.json').read_text())['cases'][0]['first_batch']['record']['provenance'])
        provenance['models']['vae'] = vae
        if implementation_by_origin is not None:
            provenance['encoding']['implementation_revision'] = implementation_by_origin[origin]
        media = provenance['media']
        media['source_shot_segments'] = ([[0,5]] if offset else []) + [[shot['start_frame'],shot['end_frame']]]
        media['sampling']['frame_indices_by_shot'] = ([[0,1,2,3,4]] if offset else []) + planned['source_frame_indices_by_shot']
        media['preprocessing']['spatial'][0]['train_crop_box'] = scene['crop_box']
        media['files'][0]['path'] = scene['source_video_path']
        record = {'case_id': f'controlled-{i}', 'cache_file': f'{i}.safetensors', 'input_latents_shape': [1,channels,n+offset,height,width],
            'num_frames': shot['frames'] + (5 if offset else 0), 'shot_groups': ([[0,5],[5,5+shot['frames']]] if offset else None),
            'latent_shot_groups': ([[0,2],[2,2+n]] if offset else None), 'provenance': provenance}
        records[origin].append(record)
        entry = {k:shot[k] for k in ('sid','scene_id','source_shot_index')}
        index.append({**entry, 'origin': origin, 'file':record['cache_file'], 'latent_start':offset,
                      'latent_end':offset+n, 'train_frames':shot['frames']})
        tensor = (torch.arange(n+offset, dtype=torch.bfloat16) + 100*i).view(1,1,-1,1,1).expand(record['input_latents_shape']).contiguous()
        save_file({'input_latents':tensor}, str(roots[origin]/record['cache_file']), metadata={'provenance':json.dumps(provenance)})
    for origin, root in roots.items(): (root/'manifest.json').write_text(json.dumps({'records':records[origin]}))
    write(tmp_path/'targets.jsonl',index)
    row = BlockRow(member['case_id'], tuple(s['frames'] for s in shots), tuple(range(len(shots))))
    sampler = BlockWindowSampler([row], frame_buckets=[228], bucket_weights=[1], max_shots=2, seed=42, shot_count_weights=None)
    samples=[]
    for index in range(61):
        _, span, _ = sampler.draw(index)
        samples.append({'sample_index':index,'case_id':row.case_id,'span':list(span)})
    write(tmp_path/'sampling.jsonl',samples)
    return dict(media=ANNOTATION_MEDIA, sampling_manifest_path=str(tmp_path/'sampling.jsonl'), members_path=str(tmp_path/'members.jsonl'),
        target_storage='latents', target_index_path=str(tmp_path/'targets.jsonl'),
        target_roots={k:str(v) for k,v in roots.items()},
        sample_budget=61,tokenizer_path='controlled-tokenizer',segment_capacity={'max_frames':228},
        reference_row_dropout_p=0.0, reference_policy="random_frame", self_attention_policy='holocine_first_frame',
        target_mask_weights={'index_path':str(tmp_path/'members.jsonl'),'root':'controlled-root','weighting':'area_coverage'},
        box_dropout_p=None, box_jitter=None, training_seed=42, video_decode_num_threads=1,
        memory_prefix_options={'frames_max':4,'dropout_p':0.2,'noise':NOISE_CONFIG,'reference_offset':OFFSET_CONFIG,
                              'attention_policy':'bidirectional_v1','row_policy':'fixed_role_v1','placement':'ordered_slots',
                              'attention_window':FULL_WINDOW_CONFIG})


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    monkeypatch.setattr('mspc.training.block_dataset.load_umt5_token_counter',lambda _:lambda t:len(t.split()))
    return materialize(tmp_path)


def test_cache_stage_keeps_complete_metadata_and_requires_only_real_admitted_targets(inputs):
    complete = BlockDataset(**inputs)
    sample = next(row for row in complete.sampling_manifest if row['span'][1] - row['span'][0] == 1)
    index, position = sample['sample_index'], sample['span'][0]
    member = complete.members[sample['case_id']]
    key = shot_key(member['shots'][position])
    entry = complete.target_index[key]
    write(Path(inputs['target_index_path']), [entry])
    root = Path(inputs['target_roots'][entry['origin']])
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['records'] = [row for row in manifest['records'] if row['cache_file'] == entry['file']]
    (root/'manifest.json').write_text(json.dumps(manifest))
    selected = {**inputs, 'target_roots': {entry['origin']: str(root)}, 'required_sample_indices': [index]}
    stage = BlockDataset(**selected)
    assert len(stage) == len(complete) == inputs['sample_budget']
    assert stage.sampling_manifest == complete.sampling_manifest
    assert stage.latent_frame_upper_bounds() == complete.latent_frame_upper_bounds()
    assert stage.metadata_summary['sampled_target_shots'] == complete.metadata_summary['sampled_target_shots']
    assert stage.metadata_summary['cache_ready_target_shots'] == 1
    assert set(stage.targets) == {(sample['case_id'], position)}
    other = next(i for i in range(len(stage)) if i != index)
    with pytest.raises(ValueError, match='outside the explicit cache stage'):
        stage[other]
    with pytest.raises(ValueError, match='sampled target cache is missing'):
        BlockDataset(**{**selected, 'required_sample_indices': None})
    (root/entry['file']).unlink()
    with pytest.raises(FileNotFoundError, match='admitted target cache file is missing'):
        BlockDataset(**selected)


def test_cache_stage_validation_remains_full_and_checks_its_physical_files(inputs):
    from mspc.training.block_dataset import build_validation_dataset

    args = SimpleNamespace(dataset_cfg=inputs, validation={
        'members_path': inputs['members_path'], 'sample_budget': inputs['sample_budget'],
        'sampling_manifest_path': inputs['sampling_manifest_path'], 'text_admission_path': None},
        cache_stage_plan={'sample_indices': [0], 'validation_sample_indices': list(range(inputs['sample_budget']))})
    val = build_validation_dataset(args)
    assert val.required_sample_indices == frozenset(range(inputs['sample_budget']))
    assert len(val.targets) > 1
    entry = next(iter(val.target_index.values()))
    (Path(inputs['target_roots'][entry['origin']])/entry['file']).unlink()
    with pytest.raises(FileNotFoundError, match='admitted target cache file is missing'):
        build_validation_dataset(args)


def test_length_buckets_do_not_materialize_references(inputs, monkeypatch):
    dataset = BlockDataset(**{**inputs, 'video_decode_num_threads': 4})
    assert dataset.video_decode_num_threads == 4
    def unexpected(*args, **kwargs):
        raise AssertionError('length bucketing read reference pixels')
    monkeypatch.setattr('mspc.training.block_dataset.read_reference_frames', unexpected)
    lengths = dataset.latent_frame_upper_bounds()
    assert len(lengths) == len(dataset)
    assert not dataset._draws


def test_attention_workloads_match_the_actual_routed_model_inputs_without_loading_pixels(inputs, monkeypatch):
    dataset = BlockDataset(**{**inputs, 'memory_prefix_options': {**inputs['memory_prefix_options'],
                                                                'attention_policy': 'routed_self_only_v1'}})
    state = torch.get_rng_state().clone()
    with monkeypatch.context() as only_metadata:
        def unexpected(*args, **kwargs):
            raise AssertionError('workload grouping materialized pixels or text')
        only_metadata.setattr('mspc.training.block_dataset.read_reference_frames', unexpected)
        only_metadata.setattr(dataset, 'text', unexpected)
        workloads = dataset.training_workloads(self_attention_policy='holocine_first_frame')
    assert len(workloads) == len(dataset)
    assert not dataset._draws
    assert torch.equal(torch.get_rng_state(), state)
    chosen = {}
    for index in range(len(dataset)):
        row, span, key = dataset.target_draw(index)
        chosen.setdefault(bool(dataset.references(index, span, key)), index)
    assert set(chosen) == {False, True}
    for index in chosen.values():
        sample = dataset[index]
        refs = sample.get('memory_frame_count', 0)
        groups = sample.get('latent_shot_groups', [[0, sample['input_latents'].shape[2]]])[refs:]
        lengths = [end-start for start, end in groups]
        routing = sample.get('reference_routing', [[] for _ in lengths])
        assert workloads[index] == {
            'latent_frames': sample['_dataset_timing']['train_latent_frames'] + refs,
            'attention_frame_pairs': refs + sum(length*(length+len(lengths)+len(visible))
                for length, visible in zip(lengths, routing, strict=True)),
        }
    assert dataset.training_workloads(self_attention_policy='holocine_first_frame') == workloads


def test_calibrated_workloads_reject_other_attention_contracts(inputs):
    dataset = BlockDataset(**inputs)
    with pytest.raises(ValueError, match='holocine_first_frame'):
        dataset.training_workloads(self_attention_policy='full')
    with pytest.raises(ValueError, match='routed_self_only_v1'):
        dataset.training_workloads(self_attention_policy='holocine_first_frame')


@pytest.fixture(autouse=True)
def controlled_masks(monkeypatch):
    monkeypatch.setattr('mspc.training.block_dataset.TargetMaskWeights', ControlledTargetMasks)


@pytest.fixture(autouse=True)
def cpu_frames(monkeypatch):
    monkeypatch.setattr('mspc.training.block_dataset.read_reference_frames',
        lambda scene, indices, **kwargs: (torch.tensor([i % 256 for i in indices], dtype=torch.uint8)
            .view(-1,1,1,1).expand(-1,3,480,832), {"reference_output_frames": len(set(indices))}))


def test_reads_targets_and_only_randomly_selected_reference_frames(inputs):
    dataset=BlockDataset(**inputs)
    assert len(dataset)==61
    seen=set();lengths=dataset.latent_frame_upper_bounds()
    for i in range(len(dataset)):
        sample=dataset[i];draw=dataset.draw(i);seen.add(draw.span)
        assert sample['input_latents'].shape[2]<=lengths[i]
        assert not set(sample['target_shot_ids']) & set(sample['reference_shot_ids'])
        assert len(draw.references) <= 4
        assert len(set(draw.references)) == len(draw.references)
        assert sample['reference_images'].shape == (len(draw.references), 3, 480, 832)
        count=len(draw.references)
        assert sample['prompt'][:count] == [reference_text()] * count
        assert not sample['_sample_metadata']['reference_text_dropped']
        target=sample['input_latents'][0,0,:,0,0].tolist()
        expected=[]
        for shot in range(*draw.span):
            start=2 if shot==0 else 0
            expected += [100*shot+n for n in range(start,start+latent_frames(draw.row.frame_counts[shot]))]
        assert target==expected
    assert seen=={(0,1),(0,2),(1,2)}
    other=BlockDataset(**inputs)
    assert [dataset.draw(i) for i in range(61)]==[other.draw(i) for i in range(61)]


def test_target_weight_batch_concatenates_only_target_shots(inputs, monkeypatch):
    # 数学张量只核batch接线；真实SAM生产/读取由test_target_masks覆盖。
    class ControlledWeights(ControlledTargetMasks):
        def read(self, member, position, *, latent_spatial_shape):
            n = latent_frames(member['shots'][position]['frames'])
            return (torch.full((1, 1, n, *latent_spatial_shape), 1.25 + position * .25),
                    {'shot': position + 1, 'valid_mask_frames': [True] * n, 'unknown_mask_frames': []})

    monkeypatch.setattr('mspc.training.block_dataset.TargetMaskWeights', ControlledWeights)
    dataset = BlockDataset(**inputs)
    references_seen = False
    for index in range(len(dataset)):
        sample = dataset[index]
        weights = sample['target_loss_weights']
        assert weights.shape == (1, 1, sample['input_latents'].shape[2], 30, 52)
        assert [record['shot'] for record in sample['target_mask_metadata']] == sample['target_shot_ids']
        expected = torch.cat([torch.full((latent_frames(dataset.members[sample['case_id']]['shots'][shot-1]['frames']),),
                                         1.25 + (shot-1) * .25) for shot in sample['target_shot_ids']])
        assert torch.equal(weights[0, 0, :, 0, 0], expected)
        references_seen |= bool(sample['memory_frame_count'])
    assert references_seen


def test_explicit_layouts_are_target_metadata_even_with_reference_dropout(inputs):
    members = lines(Path(inputs['members_path']))
    members[0]['annotation']['layouts'] = []
    write(Path(inputs['members_path']), members)
    dataset = BlockDataset(**inputs)
    for index in range(len(dataset)):
        sample = dataset[index]
        assert len(sample['target_layouts']) == len(sample['target_shot_ids'])
        target_prompts = sample['prompt'][sample['memory_frame_count']:]
        assert [layout['text'] for layout in sample['target_layouts']] == target_prompts
        assert all(layout['entities'] == [] for layout in sample['target_layouts'])


def test_cpu_preflight_and_source_mismatch(inputs,tmp_path):
    summary=inspect_dataset(dataset_cfg=inputs,output=tmp_path/'report',tensor_samples=5)
    assert summary['tensor_pass']['samples']==5
    assert summary['text_overflow']['samples']==0
    rows=lines(Path(inputs['members_path']));rows[0]['scene']['crop_box'][0]+=1
    write(Path(inputs['members_path']),rows)
    with pytest.raises(ValueError,match='crop'):
        BlockDataset(**inputs)


def test_wan21_cache_geometry_keeps_frozen_draws_and_target_mask_grid(inputs, tmp_path):
    native_root = tmp_path / 'wan21'
    native_root.mkdir()
    vae = {'path': '/controlled-test/Wan2.1_VAE.pth', 'revision': 'controlled-wan21'}
    native_inputs = materialize(native_root, latent_geometry=(16, 60, 104),
                                vae_by_origin={'shot': vae}, target_origins=('shot',))
    previous, native = BlockDataset(**inputs), BlockDataset(**native_inputs)
    assert set(native.roots) == {'shot'}
    assert (native.latent_channels, native.latent_spatial_shape, native.vae_spatial_stride) == (16, (60, 104), 8)
    assert (previous.latent_channels, previous.latent_spatial_shape, previous.vae_spatial_stride) == (48, (30, 52), 16)
    assert [previous.draw(i) for i in range(len(previous))] == [native.draw(i) for i in range(len(native))]
    sample = native[0]
    assert sample['input_latents'].shape[1:] == (16, sample['input_latents'].shape[2], 60, 104)
    assert sample['target_loss_weights'].shape[3:] == (60, 104)


def test_online_video_uses_the_frozen_draw_and_existing_pixel_reader(inputs, monkeypatch):
    seen = []

    def read(row, indices, *, output_mode, video_decode_num_threads, media):
        seen.append((row, indices, output_mode, video_decode_num_threads))
        return torch.tensor([frame % 256 for frame in indices], dtype=torch.uint8).view(1, -1, 1, 1).expand(3, -1, 480, 832)

    def masks(*, member, positions, spatial_root, weighting, vae_spatial_stride, media):
        assert spatial_root == 'controlled-spatial-root' and weighting == 'area_coverage' and vae_spatial_stride == 8
        return [ControlledTargetMasks(index_path=inputs['members_path'], root='', weighting=weighting).read(
            member, position, latent_spatial_shape=(60, 104)) for position in positions]

    monkeypatch.setattr('mspc.data.scene_cache.read_scene_video', read)
    monkeypatch.setattr('mspc.training.block_dataset.compile_member_target_mask_weights', masks)
    cached = BlockDataset(**inputs)
    online = BlockDataset(**{**inputs, 'target_storage': 'video', 'target_index_path': None, 'target_roots': None,
        'target_latent_geometry': {'channels': 16, 'spatial_stride': 8},
        'target_mask_weights': {'spatial_root': 'controlled-spatial-root', 'weighting': 'area_coverage'}})
    with pytest.raises(RuntimeError, match='bind the loaded online VAE'):
        online[0]
    with pytest.raises(ValueError, match='differs from the explicit'):
        online.bind_online_vae_geometry(48, 16)
    online.bind_online_vae_geometry(16, 8)
    assert online.stateless_index_samples is True
    assert [online.draw(index) for index in range(len(online))] == [cached.draw(index) for index in range(len(cached))]
    assert not online.target_index and not online.targets and not online.roots
    index = next(index for index in range(len(online)) if len(online.draw(index).references))
    sample, comparison, draw = online[index], cached[index], online.draw(index)
    assert sample['target_storage'] == 'video' and 'input_latents' not in sample
    assert sample['prompt'] == comparison['prompt']
    assert sample['_sample_metadata'] == comparison['_sample_metadata']
    assert sample['shot_groups'] == comparison['shot_groups']
    assert sample['latent_shot_groups'] == comparison['latent_shot_groups']
    assert torch.equal(sample['reference_images'], comparison['reference_images'])
    assert sample['target_loss_weights'].shape == (1, 1, sample['_dataset_timing']['train_latent_frames'], 60, 104)
    scene = {**online.members[draw.row.case_id]['scene'], 'shots': online.members[draw.row.case_id]['shots']}
    expected = [frame for position in range(*draw.span) for frame in planned_row(scene, position, media=ANNOTATION_MEDIA)['source_frame_indices_by_shot'][0]]
    assert seen[0][1:] == (expected, 'uint8', inputs['video_decode_num_threads'])
    assert sample['video'][0, :, 0, 0].tolist() == [frame % 256 for frame in expected]


def test_video_storage_cannot_rebind_latent_cache_inputs(inputs):
    with pytest.raises(ValueError, match='do not accept latent cache'):
        BlockDataset(**{**inputs, 'target_storage': 'video',
                        'target_latent_geometry': {'channels': 16, 'spatial_stride': 8}})
    with pytest.raises(ValueError, match='only video targets'):
        BlockDataset(**inputs).bind_online_vae_geometry(48, 16)


def test_header_provenance_is_checked_when_target_bytes_are_read(inputs):
    dataset=BlockDataset(**inputs)
    entry, record=dataset.targets[(dataset.rows[0].case_id,0)]
    path=dataset.roots[entry['origin']]/entry['file']
    save_file({'input_latents':torch.zeros(record['input_latents_shape'],dtype=torch.bfloat16)},str(path),metadata={'provenance':'{}'})
    with pytest.raises(ValueError):dataset._target_latent(dataset.rows[0].case_id,0)


def test_uncached_usable_reference_keeps_the_same_draw_and_only_sampled_targets_need_cache(inputs):
    sampling_path = Path(inputs['sampling_manifest_path'])
    samples = lines(sampling_path)
    for row in samples:
        row['span'] = [0, 1]
    write(sampling_path, samples)
    complete = BlockDataset(**inputs)
    target_path = Path(inputs['target_index_path'])
    write(target_path, [row for row in lines(target_path) if row['origin'] == 'first_batch_window'])
    for file in Path(inputs['target_roots']['shot']).glob('*.safetensors'):
        file.unlink()
    selected = BlockDataset(**{**inputs, 'target_roots': {'first_batch_window': inputs['target_roots']['first_batch_window']}})
    assert selected.rows == complete.rows
    assert selected.rows[0].usable == (0, 1)
    assert selected.metadata_summary['sampled_target_shots'] == 1
    assert [selected.draw(i) for i in range(len(selected))] == [complete.draw(i) for i in range(len(complete))]
    index = next(i for i in range(len(selected)) if selected.draw(i).references)
    sample = selected[index]
    assert sample['reference_shot_ids'] == [2]
    assert sample['reference_images'].shape[0] == 1
    assert set(selected.targets) == {(selected.rows[0].case_id, 0)}


def test_missing_sampled_target_fails_while_unused_shot_cache_cannot_substitute(inputs):
    samples = lines(Path(inputs['sampling_manifest_path']))
    for row in samples:
        row['span'] = [0, 1]
    write(Path(inputs['sampling_manifest_path']), samples)
    write(Path(inputs['target_index_path']), [row for row in lines(Path(inputs['target_index_path'])) if row['origin'] == 'shot'])
    with pytest.raises(ValueError, match='sampled target cache is missing'):
        BlockDataset(**{**inputs, 'target_roots': {'shot': inputs['target_roots']['shot']}})


def test_cached_targets_can_use_explicit_original_sam_rle(inputs, monkeypatch):
    def compile_masks(*, member, positions, spatial_root, weighting, vae_spatial_stride, media):
        assert spatial_root == 'original-sam' and weighting == 'area_coverage' and vae_spatial_stride == 16
        return [ControlledTargetMasks(index_path=inputs['members_path'], root='', weighting=weighting).read(
            member, position, latent_spatial_shape=(30, 52)) for position in positions]

    monkeypatch.setattr('mspc.training.block_dataset.compile_member_target_mask_weights', compile_masks)
    dataset = BlockDataset(**{**inputs, 'target_mask_weights': {'spatial_root': 'original-sam', 'weighting': 'area_coverage'}})
    sample = dataset[0]
    assert sample['target_storage'] == 'latents' and 'video' not in sample
    assert sample['target_loss_weights'].shape[2:] == sample['input_latents'].shape[2:]
    assert dataset.target_mask_weights is None


def test_window_sampling_balances_duration_buckets_without_a_shot_cap():
    # 抽样是纯数学函数；12 镜短镜块总帧数 >228，坏镜保留原位置，窗口只受 228 帧约束、镜数不封顶。
    rows=(BlockRow('short',(21,)*12,(0,1,2,3,4,6,7,8,9,10,11)),BlockRow('long',(57,)*9,tuple(range(9))))
    sampler=BlockWindowSampler(rows,frame_buckets=[80,160,228],bucket_weights=[1,1,1],max_shots=None,seed=42, shot_count_weights=None)
    buckets,shots=Counter(),Counter()
    for index in range(24000):
        row,span,_=sampler.draw(index)
        frames=sum(row.frame_counts[slice(*span)])
        assert frames <= 228 and set(range(*span)) <= set(row.usable)
        buckets[sampler.bucket(frames)]+=1
        shots[span[1]-span[0]]+=1
    assert all(abs(buckets[b]/24000-1/3)<.01 for b in range(3))
    assert max(shots) == 6 and shots[5] > 0
    replay=BlockWindowSampler(rows,frame_buckets=[80,160,228],bucket_weights=[2,2,2],max_shots=None,seed=42, shot_count_weights=None)
    assert [sampler.draw(i) for i in range(200)] == [replay.draw(i) for i in range(200)]


def test_window_sampling_uses_each_legal_window_uniformly_within_a_bucket():
    rows=(BlockRow('four',(57,)*4,(0,1,2,3)),BlockRow('two',(57,)*2,(0,1)))
    sampler=BlockWindowSampler(rows,frame_buckets=[114,228],bucket_weights=[1,0],max_shots=None,seed=42, shot_count_weights=None)
    counts=Counter((row.case_id,span) for row,span,_ in (sampler.draw(i) for i in range(20000)))
    assert len(counts) == 10 and all(span[1]-span[0] <= 2 for _, span in counts)
    assert all(abs(n/20000-.1)<.01 for n in counts.values())


def test_single_shot_windows_for_the_per_shot_control():
    rows=(BlockRow('four',(57,)*4,(0,1,2,3)),)
    sampler=BlockWindowSampler(rows,frame_buckets=[228],bucket_weights=[1],max_shots=1,seed=42, shot_count_weights=None)
    assert {span for _,span,_ in (sampler.draw(i) for i in range(500))} == {(0,1),(1,2),(2,3),(3,4)}


def test_window_sampling_rejects_an_empty_requested_bucket_and_bad_bounds():
    with pytest.raises(ValueError,match='up to 228 frames'):
        BlockWindowSampler([BlockRow('single',(57,),(0,))],frame_buckets=[80,228],bucket_weights=[1,1],max_shots=None,seed=42, shot_count_weights=None)
    with pytest.raises(ValueError,match='increasing positive'):
        BlockWindowSampler([BlockRow('single',(57,),(0,))],frame_buckets=[160,80],bucket_weights=[1,1],max_shots=None,seed=42, shot_count_weights=None)


def test_pack_span_capacity():
    assert pack_span((57,57,57,57,57),0)==(0,4)
    with pytest.raises(ValueError):pack_span((229,),0)


def test_single_usable_shot_needs_no_reference_candidate_and_bad_neighbor_is_never_sampled(inputs):
    rows=lines(Path(inputs['members_path']))
    rows[0]['shots'][1].update(usable=False,unusable_reasons=['controlled-failure'])
    write(Path(inputs['members_path']),rows)
    samples=[{'sample_index':i,'case_id':rows[0]['case_id'],'span':[0,1]} for i in range(61)]
    write(Path(inputs['sampling_manifest_path']),samples)
    dataset=BlockDataset(**inputs)
    for i in range(len(dataset)):
        assert dataset.draw(i).span==(0,1)
        assert not dataset.draw(i).references


@pytest.fixture
def aliased_inputs(tmp_path, monkeypatch):
    """首批实存 VAE provenance + 补建日志明确记录的另一运行根；张量仍为受控序号。"""
    monkeypatch.setattr('mspc.training.block_dataset.load_umt5_token_counter', lambda _: lambda t: len(t.split()))
    real = json.loads((REPO_ROOT / 'tests/unit/data/fixtures/shot_targets_real.json').read_text())
    first = real['cases'][0]['first_batch']['record']['provenance']['models']['vae']
    shot = {**first, 'path': first['path'].replace(
        'outputs/experiments/source-pipeline-20261011/first-batch-members',
        'outputs/datasets/scene_blocks_v1/target_latents')}
    assert first['path'] != shot['path'] and first['revision'] == shot['revision']
    return materialize(tmp_path, vae_by_origin={'first_batch_window': first, 'shot': shot})


def test_same_producer_revision_accepts_different_cache_run_roots(aliased_inputs):
    dataset = BlockDataset(**aliased_inputs)
    for index in range(len(dataset)):
        sample = dataset[index]  # 连同 target header 与各镜 reference header 一起核
        assert sample['input_latents'].shape[2] <= dataset.latent_frame_upper_bounds()[index]


@pytest.fixture
def opaque_source_inputs(tmp_path, monkeypatch):
    """Only index origins change; packed-window layout and producer headers stay intact."""
    from mspc.data.cache_provenance import build_cache_source_collection

    monkeypatch.setattr('mspc.training.block_dataset.load_umt5_token_counter', lambda _: lambda t: len(t.split()))
    implementations = {'first_batch_window': '69f094f0a64f32b9a596774ddf96ecbf7826e60f',
                       'shot': '86f59ef97bee57dd29006f2381fafe31db61d165'}
    arguments = materialize(tmp_path, implementation_by_origin=implementations)
    names = {'first_batch_window': 'existing_shot', 'shot': 'backfilled_shot'}
    index = Path(arguments['target_index_path'])
    write(index, [{**row, 'origin': names[row['origin']]} for row in lines(index)])
    arguments['target_roots'] = {names[name]: root for name, root in arguments['target_roots'].items()}
    original_files = {str(path): path.read_bytes() for root in arguments['target_roots'].values()
                      for path in Path(root).iterdir()}
    collection = build_cache_source_collection(index, arguments['target_roots'])
    assert {name: source['encoding']['implementation_revision'] for name, source in collection['sources'].items()} == {
        names[name]: revision for name, revision in implementations.items()}
    return arguments, original_files


def test_opaque_sources_preserve_two_producer_versions_and_original_files(opaque_source_inputs):
    arguments, original_files = opaque_source_inputs
    dataset = BlockDataset(**arguments)
    assert set(dataset.roots) == {'existing_shot', 'backfilled_shot'}
    case = dataset.rows[0].case_id
    for position in dataset.rows[0].usable:
        tensor = dataset._target_latent(case, position)
        entry, record = dataset.targets[(case, position)]
        assert tensor.shape[2] == entry['latent_end'] - entry['latent_start']
        # Existing packed-window files retain a nonzero offset even though
        # their origin is no longer named first_batch_window.
        if entry['origin'] == 'existing_shot':
            assert entry['latent_start'] == 2
            assert float(tensor[0, 0, 0, 0, 0]) == 2
    assert {name: Path(name).read_bytes() for name in original_files} == original_files


@pytest.mark.parametrize('difference', ['source_frame', 'crop', 'vae', 'encoding'])
def test_opaque_sources_still_validate_actual_trace_and_encoding(opaque_source_inputs, difference):
    arguments, _ = opaque_source_inputs
    path = Path(arguments['target_roots']['backfilled_shot']) / 'manifest.json'
    manifest = json.loads(path.read_text())
    provenance = manifest['records'][0]['provenance']
    if difference == 'source_frame':
        provenance['media']['sampling']['frame_indices_by_shot'][0][1] += 1
    elif difference == 'crop':
        provenance['media']['preprocessing']['spatial'][0]['train_crop_box'][0] += 2
    elif difference == 'vae':
        provenance['models']['vae']['revision'] += '-different'
    else:
        provenance['encoding']['pixel_conversion'] += '-different'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='target source mismatch|target VAE/encoding differs'):
        BlockDataset(**arguments)


@pytest.mark.parametrize('difference', ['empty_name', 'unknown_origin', 'duplicate_key'])
def test_opaque_sources_refuse_invalid_names_origins_and_duplicate_keys(opaque_source_inputs, difference):
    arguments, _ = opaque_source_inputs
    if difference == 'empty_name':
        root = arguments['target_roots'].pop('existing_shot')
        arguments['target_roots'][' '] = root
    else:
        path = Path(arguments['target_index_path'])
        rows = lines(path)
        if difference == 'unknown_origin':
            rows[0]['origin'] = 'not_declared'
        else:
            rows.append(deepcopy(rows[0]))
        write(path, rows)
    with pytest.raises(ValueError, match='non-empty source names|duplicate target key or unknown origin'):
        BlockDataset(**arguments)


@pytest.mark.parametrize('difference', ['revision', 'encoding'])
def test_mixed_target_revision_or_encoding_still_fails(aliased_inputs, difference):
    path = Path(aliased_inputs['target_roots']['shot']) / 'manifest.json'
    manifest = json.loads(path.read_text())
    provenance = manifest['records'][0]['provenance']
    if difference == 'revision':
        provenance['models']['vae']['revision'] += '-different'
    else:
        provenance['encoding']['tiled'] = not provenance['encoding']['tiled']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='target VAE/encoding differs across the pool'):
        BlockDataset(**aliased_inputs)


def test_only_selected_source_indices_are_read(inputs, monkeypatch):
    dataset = BlockDataset(**inputs)
    index = next(i for i in range(len(dataset)) if dataset.draw(i).references)
    draw = dataset.draw(index)
    member = dataset.members[draw.row.case_id]
    scene = {**member['scene'], 'shots': member['shots']}
    expected = [planned_row(scene, shot, media=ANNOTATION_MEDIA)['source_frame_indices_by_shot'][0][frame] for shot, frame in draw.references]
    calls = []
    def read(scene, indices, **kwargs):
        calls.append(indices)
        return torch.zeros(len(indices), 3, 480, 832, dtype=torch.uint8), {}
    monkeypatch.setattr('mspc.training.block_dataset.read_reference_frames', read)
    dataset[index]
    assert calls == [expected]


def test_manifest_length_must_match_the_sample_budget(inputs):
    path=Path(inputs['sampling_manifest_path'])
    write(path,lines(path)[:-1])
    with pytest.raises(ValueError,match='length differs'):
        BlockDataset(**inputs)


def random_references(*, span, seed, dropout_p=0.0, frames_max=10, usable=None, empty_target_entities=False):
    """复用真实十二镜块的在场名单与帧数，只运行元数据抽样，不读 SAM 或目标缓存。"""
    real = json.loads((REPO_ROOT / 'tests/fixtures/block_layouts/film-003759-b0001.json').read_text())
    frames = tuple(shot['render_frames'] for shot in real['frames']['shots'])
    if empty_target_entities:
        for position in range(*span):
            real['annotation']['shots'][position]['present'] = []
    row = BlockRow(real['key'], frames, tuple(range(len(frames))) if usable is None else usable)
    helper = SimpleNamespace(references_enabled=True, planner_tokens=None, reference_policy='random_frame',
        memory_options=SimpleNamespace(dropout_p=dropout_p, frames_max=frames_max),
        annotations={row.case_id: real['annotation']},
        reference_frame_counts={row.case_id: frames},
        sampling_manifest=[{'case_id': row.case_id}], row_by_case={row.case_id: row})
    helper.reference_plan = lambda *args: BlockDataset.reference_plan(helper, *args)
    key = (seed, 0)
    return helper.reference_plan(row, span, key), BlockDataset.references(helper, 0, span, key)


def test_random_references_keep_the_frozen_draw_keys():
    # 与 06776a926 同一组抽签键：同块旧表的参考标签对未变的样本仍可用。
    real = json.loads((REPO_ROOT / 'tests/fixtures/block_layouts/film-003759-b0001.json').read_text())
    shots = real['annotation']['shots']
    for seed in range(50):
        (frames, count), references = random_references(span=(0, 1), seed=seed)
        key = (seed, 0)
        assert count == 1 + int(unit_draw('reference_count', *key) * min(10, len(frames)))
        wanted = {entry['name'] for entry in shots[0]['present']}
        order = sorted(frames, key=lambda position: (
            not wanted & {entry['name'] for entry in shots[position]['present']},
            unit_draw('reference_pick', *key, position)))
        assert references == tuple((position, frames[position][int(unit_draw('reference_frame', *key, position)
                                                                  * len(frames[position]))])
                                   for position in sorted(order[:count]))


def test_first_segment_can_reference_later_usable_shots_without_sam():
    for seed in range(200):
        (frames, count), references = random_references(span=(0, 2), seed=seed)
        assert set(frames) == set(range(2, 12))
        assert 1 <= count <= 10 and len(references) == count
        assert all(position >= 2 for position, _ in references)


def test_references_exclude_target_span_and_unusable_shots():
    usable = tuple(i for i in range(12) if i not in (1, 9))
    for seed in range(100):
        (frames, count), references = random_references(span=(4, 8), usable=usable, seed=seed)
        assert set(frames) == {0, 2, 3, 8, 10, 11}
        assert len(references) == count <= len(frames)
        assert all(position in frames for position, _ in references)


def test_shared_entities_are_prioritized_without_discarding_other_shots():
    related = {2, 4, 6}  # 真实首镜的 Character 1 / Object 1 也在这三镜出现。
    fallback_seen = False
    for seed in range(200):
        plan, references = random_references(span=(0, 1), seed=seed)
        assert len(plan[0]) == 11
        assert sum(position in related for position, _ in references) == min(len(related), len(references))
        fallback_seen |= any(position not in related for position, _ in references)
        assert (plan, references) == random_references(span=(0, 1), seed=seed)
    assert fallback_seen


@pytest.mark.parametrize('empty_target_entities', [False, True])
def test_no_shared_or_missing_entities_do_not_remove_reference_images(empty_target_entities):
    # 这些真实候选镜只有 Character 2/3 与 Object 2，与首镜没有共同实体。
    usable = (0, 1, 3, 5, 7, 8, 10)
    (frames, count), references = random_references(span=(0, 1), usable=usable, seed=1,
                                                   empty_target_entities=empty_target_entities)
    assert set(frames) == set(usable) - {0}
    assert 1 <= count <= 6 and len(references) == count


def test_ten_reference_cap_counts_distinct_shots_and_returns_chronological_order():
    counts = set()
    for seed in range(300):
        (_, count), references = random_references(span=(0, 1), seed=seed)
        counts.add(count)
        positions = [position for position, _ in references]
        assert len(positions) == count <= 10
        assert positions == sorted(set(positions))
    assert counts == set(range(1, 11))


def test_uniform_reference_frame_grid_includes_zero_and_the_last_grid_frame():
    seen = set()
    for seed in range(512):
        (frames, count), references = random_references(span=(0, 1), usable=(0, 8), seed=seed)
        assert frames == {8: list(range(0, 85, 4))}
        assert count == 1 and len(references) == 1
        seen.add(references[0][1])
    assert seen == set(range(0, 85, 4))


def test_a_segment_without_an_outside_usable_shot_has_no_references():
    plan, references = random_references(span=(0, 2), usable=(0, 1), seed=1)
    assert plan == ({}, 0) and references == ()


def test_a_dropped_reference_group_has_no_candidates():
    plan, references = random_references(span=(1, 2), seed=1, dropout_p=0.9999)
    assert plan == ({}, 0) and references == ()


def test_training_reference_policy_is_one_of_the_two_named_selections(inputs):
    with pytest.raises(ValueError, match='reference_policy must be one of'):
        BlockDataset(**{**inputs, 'reference_policy': 'unified_views'})


def test_distinct_camera_shared_only_policy_degrades_same_camera_windows_to_empty_and_keeps_distinct_shared(inputs):
    members = lines(Path(inputs['members_path']))
    sampling = lines(Path(inputs['sampling_manifest_path']))
    by_case = {m['case_id']: m for m in members}
    sample_idx, span, member, outside = next(
        (i, row['span'], by_case[row['case_id']],
         [p for p, s in enumerate(by_case[row['case_id']]['shots'])
          if s['usable'] and not (row['span'][0] <= p < row['span'][1])])
        for i, row in enumerate(sampling)
        if any(s['usable'] and not (row['span'][0] <= p < row['span'][1])
               for p, s in enumerate(by_case[row['case_id']]['shots'])))
    cand_pos = outside[0]
    shots = member['annotation']['shots']
    place = member['annotation']['places'][0]['name']
    memory = {**inputs['memory_prefix_options'], 'dropout_p': 0.0, 'attention_policy': 'routed_self_only_v1'}
    # 情形 1：全块同机位，窗外镜共有目标实体 → 不兜底，参考置空退化为纯窗内互读。
    target_names = [e['name'] for t in range(*span) for e in shots[t]['present']]
    shared_name = target_names[0]
    for pos, shot in enumerate(shots):
        shot['place'] = place
        shot['camera'] = {'shot_size': 'medium shot', 'angle': 'eye-level shot'}
        if pos in outside:
            shot['present'] = [{'name': shared_name}]
    Path(inputs['members_path']).write_text(''.join(json.dumps(row) + '\n' for row in members))
    ds_same = BlockDataset(**{**inputs, 'memory_prefix_options': memory,
                              'reference_policy': 'distinct_camera_shared_only'})
    draw_same = ds_same.draw(sample_idx)
    assert draw_same.references == () and draw_same.reference_offset is None
    text_same = ds_same.text(sample_idx)
    assert text_same.metadata['references'] == [] and text_same.metadata['reference_offset'] is None
    wl_same = ds_same.sample_workloads()[sample_idx]
    # 情形 2：窗外仅有无关镜（哪怕异机位）→ 不补无关镜，参考仍置空。
    for pos in outside:
        shots[pos]['camera'] = {'shot_size': 'close-up shot', 'angle': 'eye-level shot'}
        shots[pos]['present'] = []
    Path(inputs['members_path']).write_text(''.join(json.dumps(row) + '\n' for row in members))
    ds_unrel = BlockDataset(**{**inputs, 'memory_prefix_options': memory,
                               'reference_policy': 'distinct_camera_shared_only'})
    assert ds_unrel.draw(sample_idx).references == ()
    # 情形 3：窗外有一个异机位共有实体镜，其余为无关镜 → 只选这一镜。
    shots[cand_pos]['present'] = [{'name': shared_name}]
    Path(inputs['members_path']).write_text(''.join(json.dumps(row) + '\n' for row in members))
    ds_distinct = BlockDataset(**{**inputs, 'memory_prefix_options': memory,
                                  'reference_policy': 'distinct_camera_shared_only'})
    draw_distinct = ds_distinct.draw(sample_idx)
    assert [shot for shot, _ in draw_distinct.references] == [cand_pos]
    assert draw_distinct.reference_offset is not None
    assert ds_distinct.sample_workloads()[sample_idx]['latent_frames'] == wl_same['latent_frames'] + 1


def test_verified_reference_consumer_checks_actual_source_frames_and_uses_registry_appearance(inputs, monkeypatch):
    from mspc.data.shot_text import shot_text_with_entity_spans
    calls = []

    class Labels:
        def for_draw(self, index, draw, *, source_frames):
            calls.append(list(source_frames))
            return [{'shot': shot + 1, 'frame': frame, 'source_frame': source,
                     'entities': [{'name': 'Character 1', 'identity_status': 'registered', 'point_cell': None}],
                     'alignment_cells': []}
                    for (shot, frame), source in zip(draw.references, source_frames, strict=True)]

    monkeypatch.setattr('mspc.training.block_dataset.ReferenceEntities', lambda *args: Labels())
    monkeypatch.setattr('mspc.training.block_dataset.read_reference_frames', lambda scene, indices, **kwargs:
                        (torch.zeros((len(indices), 3, 480, 832), dtype=torch.uint8), {}))
    memory = {**inputs['memory_prefix_options'], 'row_policy': 'frame_entities_v1',
              'attention_policy': 'routed_self_only_v1'}
    dataset = BlockDataset(**{**inputs, 'memory_prefix_options': memory,
                             'reference_entities': {'labels_path': 'controlled-verified-rows'}})
    index = next(i for i in range(len(dataset)) if dataset.draw(i).references)
    text = dataset.text(index)
    sample = dataset[index]
    expected = dataset.reference_source_frames(text.draw)
    assert calls and all(call == expected for call in calls)
    target_prompts = sample['prompt'][len(text.draw.references):]
    assert [layout['text'] for layout in sample['target_layouts']] == target_prompts
    annotation = dataset.annotations[text.draw.row.case_id]
    assert sample['reference_characters'] is annotation['characters']
    assert len(sample['reference_shot_objects']) == len(text.draw.references)
    assert 'Character 1:' in text.prompts[0]
    assert annotation['characters'][0]['appearance'].removeprefix('a ').removeprefix('an ') in text.prompts[0]
    for shot, prompt, references in zip(range(*text.draw.span), target_prompts,
                                         text.metadata['target_reference_slots'], strict=True):
        assert prompt == shot_text_with_entity_spans(annotation, shot, references=references)['text']


def test_reference_images_do_not_require_target_sam_weights(inputs):
    dataset = BlockDataset(**{**inputs, 'target_mask_weights': None})
    assert dataset.target_mask_weights is None
    assert any(dataset.draw(index).references for index in range(len(dataset)))


def test_position_sentences_are_dropped_for_the_whole_sample_and_replay_the_same_draw(inputs):
    members = lines(Path(inputs['members_path']))
    anchor = {'time': 0.0, 'visibility': 'visible', 'box': [0.1, 0.1, 0.4, 0.9]}
    members[0]['annotation']['layouts'] = [
        {'shot': shot, 'entities': [{'name': 'Character 1', 'anchors': [anchor]}]} for shot in (1, 2)]
    write(Path(inputs['members_path']), members)
    dataset = BlockDataset(**{**inputs, 'box_dropout_p': 0.5, 'minimum_layout_shot_fraction': 0.5})
    dropped_seen = kept_seen = False
    for index in range(len(dataset)):
        sample = dataset[index]
        dropped = sample['_sample_metadata']['layout_text_dropped']
        assert dropped == (unit_draw('box_dropout', 42, index) < 0.5) == dataset[index]['_sample_metadata']['layout_text_dropped']
        prompts = sample['prompt'][sample['memory_frame_count']:]
        assert [layout['text'] for layout in sample['target_layouts']] == prompts
        assert all(('composition.' in prompt) != dropped for prompt in prompts)
        assert all(bool(layout['entities']) != dropped for layout in sample['target_layouts'])
        dropped_seen |= dropped
        kept_seen |= not dropped
    assert dropped_seen and kept_seen


def test_omitted_target_text_drops_layout_and_picture_type_sentences_for_every_sample(inputs):
    members = lines(Path(inputs['members_path']))
    anchor = {'time': 0.0, 'visibility': 'visible', 'box': [0.1, 0.1, 0.4, 0.9]}
    members[0]['annotation']['layouts'] = [
        {'shot': shot, 'entities': [{'name': 'Character 1', 'anchors': [anchor]}]} for shot in (1, 2)]
    for shot in members[0]['annotation']['shots']:
        shot['picture_type'] = 'live_action'
    write(Path(inputs['members_path']), members)
    kept = BlockDataset(**inputs)
    omitted = BlockDataset(**{**inputs, 'omit_target_text': ('layouts', 'picture_type')})
    for index in range(len(omitted)):
        full = kept[index]['prompt'][kept[index]['memory_frame_count']:]
        plain = omitted[index]['prompt'][omitted[index]['memory_frame_count']:]
        assert all('composition.' in prompt and ('Live-action footage.' in prompt) for prompt in full)
        assert not any('composition' in prompt or 'Live-action' in prompt for prompt in plain)
        assert [layout['text'] for layout in omitted[index]['target_layouts']] == plain


def test_picture_type_sentence_is_dropped_for_half_of_the_samples_by_its_own_draw(inputs):
    members = lines(Path(inputs['members_path']))
    for shot in members[0]['annotation']['shots']:
        shot['picture_type'] = 'live_action'
    write(Path(inputs['members_path']), members)
    dataset = BlockDataset(**{**inputs, 'picture_type_dropout_p': 0.5})
    seen = set()
    for index in range(len(dataset)):
        text = dataset.text(index)
        dropped = text.metadata['picture_type_text_dropped']
        assert dropped == (unit_draw('picture_type_dropout', 42, index) < 0.5)
        targets = text.prompts[len(text.draw.references):]
        assert all(('Live-action footage.' in prompt) != dropped for prompt in targets)
        seen.add(dropped)
    assert seen == {True, False}
    with pytest.raises(ValueError, match="picture_type_dropout_p"):
        BlockDataset(**{**inputs, 'picture_type_dropout_p': 0.5, 'omit_target_text': ('picture_type',)})


def test_omitting_layouts_excludes_layout_dropout(inputs):
    with pytest.raises(ValueError, match="omitting layouts excludes"):
        BlockDataset(**{**inputs, 'omit_target_text': ('layouts',), 'box_dropout_p': 0.2})


def test_dataset_transmits_frame_entity_policy_to_shared_geometry(inputs, monkeypatch):
    from dataclasses import replace
    dataset = BlockDataset(**inputs)
    dataset.memory_options = replace(dataset.memory_options, row_policy="frame_entities_v1")
    index = next(i for i in range(len(dataset)) if dataset.draw(i).references)
    original = dataset.text
    visible = reference_text() + " Visible identities: Character 1."

    def frame_entity_text(item):
        text = original(item)
        count = len(text.draw.references)
        return replace(text, prompts=(visible,) * count + text.prompts[count:])

    monkeypatch.setattr(dataset, "text", frame_entity_text)
    sample = dataset[index]
    assert sample["prompt"][:sample["memory_frame_count"]] == [visible] * sample["memory_frame_count"]
    assert sample["shot_groups"][0] == [0, 1]


@pytest.mark.parametrize("memory_policy", ["routed_self_only_v1", "routed_first_latent_self_only_v1"])
def test_routed_reference_policy_writes_the_shared_routing_only_for_samples_with_references(inputs, memory_policy):
    from mspc.data.reference_routing import route_references
    routed = BlockDataset(**{**inputs, 'memory_prefix_options': {**inputs['memory_prefix_options'],
                                                                 'attention_policy': memory_policy}})
    plain = BlockDataset(**inputs)
    counts = Counter()
    for index in range(len(routed)):
        sample, draw = routed[index], routed.draw(index)
        if not draw.references:
            # 整组丢弃或段外没有可用镜：没有参考就没有路由，走目标自己的注意力。
            assert 'reference_routing' not in sample and 'reference_routing' not in sample['_sample_metadata']
            counts['without'] += 1
            continue
        routing, reasons = route_references(routed.annotations[draw.row.case_id]['shots'],
                                            reference_shots=[shot for shot, _ in draw.references],
                                            target_shots=range(*draw.span))
        assert sample['reference_routing'] == [list(entry) for entry in routing]
        assert sample['_sample_metadata']['reference_routing'] == [
            [draw.references[j][0] + 1 for j in entry] for entry in routing]
        assert sample['_sample_metadata']['reference_routing_reasons'] == list(reasons)
        assert 'reference_routing' not in plain[index]
        counts['with'] += 1
    assert counts['with'] and counts['without']


@pytest.mark.parametrize('cache_capacity,window_capacity', [(308, 308), (337, 360)])
def test_720_targets_and_reference_pixels_use_target_media_with_original_reference_times(tmp_path, monkeypatch, cache_capacity, window_capacity):
    """真实源镜界保留；目标24fps、参考旧16fps时刻，参考像素按704画幅解码裁缩。"""
    from mspc.data.scene_training_manifest import training_geometry
    media720 = dict(ANNOTATION_MEDIA, height=704, width=1280, target_fps=24,
                    max_rgb_frames=window_capacity)
    cache_media720 = dict(media720, max_rgb_frames=cache_capacity)
    arguments = materialize(tmp_path, latent_geometry=(48, 44, 80), target_origins=('shot',))
    member, = lines(tmp_path / 'members.jsonl')
    scene = {**member['scene'], 'shots': member['shots']}
    before = deepcopy(member)
    root = Path(arguments['target_roots']['shot'])
    manifest = json.loads((root / 'manifest.json').read_text())
    index = lines(tmp_path / 'targets.jsonl')
    for i, (record, entry) in enumerate(zip(manifest['records'], index, strict=True)):
        plan = planned_row(scene, i, media=cache_media720)
        frames = plan['shot_groups'][0][1]
        count = latent_frames(frames)
        record.update(num_frames=frames, input_latents_shape=[1, 48, count, 44, 80])
        evidence = record['provenance']['media']
        evidence['sampling'].update(target_fps=24, max_rgb_frames=cache_capacity,
                                    frame_indices_by_shot=plan['source_frame_indices_by_shot'])
        evidence['preprocessing']['spatial'] = [training_geometry(scene, media=media720)]
        entry.update(train_frames=frames, latent_start=0, latent_end=count)
        save_file({'input_latents': torch.full(record['input_latents_shape'], i, dtype=torch.bfloat16)},
                  str(root / record['cache_file']), metadata={'provenance': json.dumps(record['provenance'])})
    (root / 'manifest.json').write_text(json.dumps(manifest))
    write(tmp_path / 'targets.jsonl', index)
    write(tmp_path / 'sampling.jsonl', [{'sample_index': 0, 'case_id': member['case_id'], 'span': [0, 1]}])
    arguments.update(media=media720, segment_capacity={'max_frames': window_capacity},
                     sample_budget=1, target_mask_weights=None)
    arguments['memory_prefix_options']['dropout_p'] = 0.0
    monkeypatch.setattr('mspc.training.block_dataset.load_umt5_token_counter', lambda _: lambda text: len(text.split()))
    captured = {}
    def references(source, indices, *, video_decode_num_threads, media):
        captured.update(scene=source, indices=indices, media=media)
        return torch.zeros((len(indices), 3, media['height'], media['width']), dtype=torch.uint8), {}
    monkeypatch.setattr('mspc.training.block_dataset.read_reference_frames', references)
    dataset = BlockDataset(**arguments)
    sample = dataset[0]
    assert (sample['height'], sample['width']) == (704, 1280)
    assert tuple(sample['input_latents'].shape[-2:]) == (44, 80)
    assert sample['reference_images'].shape[0] > 0
    assert tuple(sample['reference_images'].shape[-2:]) == (704, 1280)
    assert captured['media'] == media720
    draw = dataset.draw(0)
    assert captured['indices'] == [planned_row(scene, shot, media=ANNOTATION_MEDIA)['source_frame_indices_by_shot'][0][frame]
                                    for shot, frame in draw.references]
    assert member == before


def test_explicit_window_capacity_keeps_a_long_shot_intact_or_reports_it():
    row = BlockRow('complete-long-shot', (337,), (0,))
    sampler = BlockWindowSampler([row], frame_buckets=[360], bucket_weights=[1], max_shots=None, seed=42, shot_count_weights=None)
    assert sampler.draw(0)[1] == (0, 1)
    assert sampler.bucket(337) == 0
    with pytest.raises(ValueError, match='complete usable shot'):
        BlockWindowSampler([row], frame_buckets=[228], bucket_weights=[1], max_shots=None, seed=42, shot_count_weights=None)


def test_long_single_shot_share_applies_only_inside_the_single_shot_layer():
    # 纯窗口抽签数学：a 镜 4（90 帧）、b 镜 2（130 帧）够长且有动作；a 镜 9（100 帧）够长但没有动作或变化。
    rows = [BlockRow("a", (20, 25, 30, 90, 20, 20, 20, 25, 100), tuple(range(9))), BlockRow("b", (60, 130, 50), (0, 1, 2))]
    action = frozenset({("a", 0), ("a", 3), ("b", 1)})
    plain = BlockWindowSampler(rows, frame_buckets=[128], bucket_weights=None, max_shots=None, seed=42,
                               shot_count_weights=[15, 25, 30, 30])
    long_ = BlockWindowSampler(rows, frame_buckets=[128], bucket_weights=None, max_shots=None, seed=42,
                               shot_count_weights=[15, 25, 30, 30], long_single_shot=(0.5, 80, action))
    assert {(row.case_id, start) for row, start, _ in long_.long_single_pool} == {("a", 3), ("b", 1)}
    singles = active_longs = inactive_longs = 0
    for index in range(4000):
        row_a, span_a, _ = plain.draw(index)
        row_b, span_b, _ = long_.draw(index)
        assert (span_a[1] - span_a[0] == 1) == (span_b[1] - span_b[0] == 1)  # 层不变
        if span_a[1] - span_a[0] > 1:
            assert (row_a.case_id, span_a) == (row_b.case_id, span_b)
        else:
            singles += 1
            active_longs += (row_b.case_id, span_b[0]) in {("a", 3), ("b", 1)}
            inactive_longs += (row_b.case_id, span_b[0]) == ("a", 8)
    assert 0.12 < singles / 4000 < 0.18
    # 单镜层一半只取有动作的长单镜，另一半在全部 12 个单镜窗里均匀取（其中 2 个有动作的长单镜、1 个没有动作的）。
    assert 0.52 < active_longs / singles < 0.65
    assert inactive_longs / singles < 0.08
    with pytest.raises(ValueError, match="does not have"):
        BlockWindowSampler(rows, frame_buckets=[128], bucket_weights=None, max_shots=None, seed=42,
                           shot_count_weights=[15, 25, 30, 30], long_single_shot=(0.3, 80, frozenset({("a", 8)}) - {("a", 8)}))
    with pytest.raises(ValueError, match="frozenset of"):
        BlockWindowSampler(rows, frame_buckets=[128], bucket_weights=None, max_shots=None, seed=42,
                           shot_count_weights=[15, 25, 30, 30], long_single_shot=(0.3, 80, {("a", 3)}))


@pytest.mark.parametrize('activity,changes,expected', [
    ('action_or_state_change', [], True), ('talking_only', [{'name': 'Character 1', 'what': 'posture'}], True),
    ('talking_only', [], False), ('no_human_activity', [], False)])
def test_action_or_change_reads_activity_and_changes(activity, changes, expected):
    from mspc.training.block_dataset import shows_action_or_change
    assert shows_action_or_change({'activity': activity, 'changes': changes}) is expected
    with pytest.raises(KeyError):
        shows_action_or_change({'activity': activity})


def test_all_target_attention_workloads_read_every_target_latent_and_the_routed_references(inputs):
    from mspc.data.reference_routing import route_references
    routed = {**inputs, 'memory_prefix_options': {**inputs['memory_prefix_options'],
                                                  'attention_policy': 'routed_self_only_v1'}}
    first_frame, full = BlockDataset(**routed), BlockDataset(**{**routed, 'self_attention_policy': 'full'})
    sparse, dense = first_frame.sample_workloads(), full.sample_workloads()
    seen = Counter()
    for index in range(len(full)):
        row, span, key = full.target_draw(index)
        lengths = [latent_frames(n) for n in row.frame_counts[slice(*span)]]
        references = full.references(index, span, key)
        routing = (route_references(full.annotations[row.case_id]['shots'], reference_shots=[s for s, _ in references],
                                    target_shots=range(*span))[0] if references else [() for _ in lengths])
        total = sum(lengths)
        assert dense[index] == {'latent_frames': total + len(references), 'attention_frame_pairs': len(references) + sum(
            length * (total + len(entry)) for length, entry in zip(lengths, routing, strict=True))}
        assert sparse[index]['latent_frames'] == dense[index]['latent_frames']
        if not references:
            assert dense[index]['attention_frame_pairs'] == total * total
        seen['with_references' if references else 'without_references'] += 1
        seen['multi_shot'] += len(lengths) > 1
    assert seen['with_references'] and seen['without_references'] and seen['multi_shot']
    cap = sorted(load['attention_frame_pairs'] for load in dense)[len(dense) // 2]
    capped = BlockDataset(**{**routed, 'self_attention_policy': 'full',
                             'workload_cap': {'latent_frames': 1000, 'attention_frame_pairs': cap}})
    assert capped.excluded_sample_indices == {i for i, load in enumerate(dense) if load['attention_frame_pairs'] > cap}
    assert capped.metadata_summary['self_attention_policy'] == 'full'
    with pytest.raises(ValueError, match='differ'):
        full.training_workloads(self_attention_policy='holocine_first_frame')
    with pytest.raises(ValueError, match='self_attention_policy must be one of'):
        BlockDataset(**{**routed, 'self_attention_policy': 'all_targets'})
