"""真实成片逐帧 MUSIQ 与生成镜区间；不运行模型。

数据来源：主服务器 benchmark-v3-baselines-20261024/dev，
native_wan_5b/shots/seed-42/film-004064-b0008-shot01。
REAL_SCORES 是 metrics 原件的完整 163 帧；镜计数来自同成片 segments.json。
它是旧 Euler 运行，用于验证后处理，不是新采样器或题库 v5 的实验读数。
"""
from copy import deepcopy
import json
from pathlib import Path
from statistics import fmean

import pytest

from mspc.evaluation import shot_imaging_quality as quality


REAL_SCORES = [
    0.7423149108886719, 0.7418854522705078, 0.7370325469970703, 0.7304916381835938,
    0.7380097198486328, 0.7348881530761718, 0.7421602630615234, 0.7386190032958985,
    0.7458383178710938, 0.7398830413818359, 0.7336932373046875, 0.736570816040039,
    0.7467686462402344, 0.744611587524414, 0.744839859008789, 0.7446031951904297,
    0.7445973968505859, 0.7466431427001953, 0.7474924468994141, 0.7473641967773438,
    0.7479090881347656, 0.6541291809082032, 0.6593279266357421, 0.6422186279296875,
    0.6381404495239258, 0.6407932281494141, 0.6528656768798828, 0.6549259185791015,
    0.6404074096679687, 0.6422371673583984, 0.6433235168457031, 0.635391960144043,
    0.636456298828125, 0.6371390533447265, 0.6369158172607422, 0.6400328063964844,
    0.6494275665283203, 0.6576203155517578, 0.7680263519287109, 0.7701627349853516,
    0.7716119384765625, 0.770704116821289, 0.7706095886230468, 0.7713863372802734,
    0.7721450042724609, 0.7734004211425781, 0.7715337371826172, 0.7700811004638672,
    0.771462173461914, 0.7745321655273437, 0.7743332672119141, 0.7754888916015625,
    0.775739517211914, 0.7753717041015625, 0.7757583618164062, 0.7119907379150391,
    0.7008446502685547, 0.7025241088867188, 0.6967821502685547, 0.6975955963134766,
    0.6961585998535156, 0.6973113250732422, 0.7020536041259766, 0.7018505096435547,
    0.7148262023925781, 0.7131682586669922, 0.7150015258789062, 0.713630599975586,
    0.7147481536865234, 0.7148690795898438, 0.7172953033447266, 0.719707260131836,
    0.7109156036376953, 0.7140985870361328, 0.7127339935302734, 0.7133111572265625,
    0.6913291168212891, 0.701561279296875, 0.6900325775146484, 0.6902620697021484,
    0.5091125869750976, 0.5804729080200195, 0.5983042907714844, 0.6037553024291992,
    0.6234356307983399, 0.6248049163818359, 0.6153005218505859, 0.6146070480346679,
    0.6434156799316406, 0.6482729339599609, 0.6521619415283203, 0.656710433959961,
    0.6837889862060547, 0.6800441741943359, 0.6814019775390625, 0.6801392364501954,
    0.7095465087890624, 0.7131844329833984, 0.7131948089599609, 0.7151667022705078,
    0.7155517578125, 0.7119087982177734, 0.7162076568603516, 0.7165847778320312,
    0.7156515502929688, 0.716739501953125, 0.7184618377685547, 0.7170301055908204,
    0.7185402679443359, 0.7134506225585937, 0.7158769989013671, 0.714455337524414,
    0.715441665649414, 0.684602279663086, 0.687461166381836, 0.6948129272460938,
    0.6961481475830078, 0.6972873687744141, 0.6947837066650391, 0.696534652709961,
    0.6836959838867187, 0.6681855773925781, 0.6733130645751954, 0.6783763122558594,
    0.682899398803711, 0.6934407043457032, 0.684939956665039, 0.6843265533447266,
    0.6839337921142579, 0.6889859771728516, 0.7075447845458984, 0.6909273529052734,
    0.6877909088134766, 0.6887529754638672, 0.6884630584716797, 0.6890535736083985,
    0.6894777679443359, 0.6833744049072266, 0.6896530151367187, 0.7042676544189453,
    0.7220821380615234, 0.7145488739013672, 0.7068295288085937, 0.6960465240478516,
    0.6946854400634765, 0.6812380218505859, 0.6991104125976563, 0.6950765228271485,
    0.7021437835693359, 0.7150344848632812, 0.7315625762939453, 0.7274014282226563,
    0.7217705535888672, 0.7107535552978516, 0.6993719482421875, 0.7067767333984375,
    0.7058731842041016, 0.6953510284423828, 0.7006088256835937, 0.7037684631347656,
    0.7247980499267578, 0.727031478881836, 0.732383041381836,
]
REAL_FRAME_COUNTS = [21, 17, 17, 41, 17, 17, 33]


@pytest.fixture
def sample():
    imaging = {'status': 'measured', 'value': 0.7023590221288013,
               'frame_indices': list(range(163)), 'frame_scores': REAL_SCORES.copy(),
               'frame_index_basis': 'zero_based_input_video', 'score_scale': 'MUSIQ/100',
               'precision': 'float32', 'preprocessing': 'longer'}
    record = {'method': 'native_wan_5b', 'story_id': 'film-004064-b0008-shot01',
              'input_tier': 'shots', 'seed': 42,
              'non_vlm': {'status': 'completed', 'frame_count': 163, 'fps': 16.,
                          'metrics': {'imaging_quality': imaging}}}
    segments = [{'start_shot': i, 'end_shot': i + 1, 'frame_counts': [count],
                 'num_frames': count} for i, count in enumerate(REAL_FRAME_COUNTS)]
    return record, segments


def test_real_story_uses_generated_shots_and_covers_every_frame(sample):
    record, segments = sample
    result = quality.summarize_story(record, segments, short_shot_max_seconds=17/16)
    assert result['frame_count'] == 163
    assert result['fps'] == 16. and result['duration_seconds'] == 163/16
    assert result['short_shot_max_seconds'] == 17/16
    assert [shot['frames'] for shot in result['shots']] == REAL_FRAME_COUNTS
    assert [(shot['start_frame'], shot['end_frame']) for shot in result['shots']] == [
        (0, 21), (21, 38), (38, 55), (55, 96), (96, 113), (113, 130), (130, 163)]
    assert [shot['shot'] for shot in result['shots']] == list(range(1, 8))
    assert result['shots'][0]['imaging_quality']['min'] == 0.7304916381835938
    assert result['shots'][1]['imaging_quality']['min'] == 0.635391960144043
    assert result['shots'][1]['imaging_quality']['mean'] == pytest.approx(fmean(REAL_SCORES[21:38]))
    groups = result['duration_groups']
    assert groups['at_most_threshold_seconds']['shots'] == 4
    assert groups['at_most_threshold_seconds']['frames'] == 68
    assert groups['more_than_threshold_seconds']['shots'] == 3
    assert groups['more_than_threshold_seconds']['frames'] == 95
    assert quality.distribution([]) == {'shots': 0, 'frames': 0, 'duration_seconds': 0, 'mean_of_shot_means': None,
                                        'mean_of_shot_minima': None, 'minimum_frame_score': None}


@pytest.mark.parametrize('change,match', [
    (lambda r: r['non_vlm'].update(status='technical_failed'), 'must be completed'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality'].update(status='not_applicable'), 'must be measured'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality'].update(frame_indices=list(range(1, 164))), 'every frame'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality']['frame_indices'].__setitem__(1, 0), 'every frame'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality']['frame_indices'].__setitem__(0, False), 'every frame'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality']['frame_scores'].pop(), 'length'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality']['frame_scores'].__setitem__(0, float('nan')), 'finite'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality'].update(value=.2), 'equal the mean'),
    (lambda r: r['non_vlm']['metrics']['imaging_quality'].update(score_scale='MUSIQ'), 'MUSIQ/100'),
])
def test_incomplete_or_incompatible_frame_measurements_fail(sample, change, match):
    record, segments = sample
    change(record)
    with pytest.raises(ValueError, match=match):
        quality.summarize_story(record, segments, short_shot_max_seconds=17/16)


def test_missing_frame_scores_are_an_explicit_missing_input(sample):
    record, segments = sample
    del record['non_vlm']['metrics']['imaging_quality']['frame_scores']
    with pytest.raises(KeyError, match='frame_scores'):
        quality.summarize_story(record, segments, short_shot_max_seconds=17/16)


@pytest.mark.parametrize('change,match', [
    (lambda s: s[0].update(num_frames=22), 'frames 21 != num_frames 22'),
    (lambda s: s[1].update(start_shot=2), 'starts at shot'),
    (lambda s: s[0].update(end_shot=2), 'shot frames invalid'),
    (lambda s: s[0]['frame_counts'].__setitem__(0, 0), 'shot frames invalid'),
    (lambda s: s[0]['frame_counts'].__setitem__(0, True), 'shot frames invalid'),
    (lambda s: s.pop(), 'differs from scored'),
])
def test_segment_holes_overlaps_or_frame_drift_fail(sample, change, match):
    record, segments = sample
    change(segments)
    with pytest.raises(ValueError, match=match):
        quality.summarize_story(record, segments, short_shot_max_seconds=17/16)


def write_pair(directory: Path, sample):
    record, segments = sample
    metrics_path, segments_path = directory / 'metrics.json', directory / 'segments.json'
    metrics_path.write_text(json.dumps(record))
    segments_path.write_text(json.dumps(segments))
    return metrics_path, segments_path


def test_cli_preserves_sources_and_only_describes_supplied_noise_labels(tmp_path, sample):
    metrics, segments = write_pair(tmp_path, sample)
    record, _ = sample
    label = {name: record[name] for name in quality.SAMPLE_FIELDS}
    labels = tmp_path / 'confirmed.json'
    labels.write_text(json.dumps({'source': 'unit-test annotation, not a model prediction',
        'shots': [{**label, 'shot': 2}, {**label, 'story_id': 'outside-this-slice', 'shot': 1}]}))
    output = tmp_path / 'readout.json'
    assert quality.main(['--pair', str(metrics), str(segments), '--noise-labels', str(labels),
                         '--output', str(output), '--short-shot-max-seconds', '1.0625']) == 0
    result = json.loads(output.read_text())
    assert result['samples'][0]['sources'] == {'metrics': str(metrics), 'segments': str(segments)}
    assert result['groups'][0]['all_shots']['shots'] == 7
    assert result['short_shot_max_seconds'] == 17/16
    assert result['short_shot_definition'] == 'actual_output_duration_seconds_lte_threshold'
    marked = result['noise_labels']
    assert marked['labels_declared'] == 2 and marked['labels_in_samples'] == 1
    assert len(marked['labels_outside_samples']) == 1
    assert marked['duration_groups']['at_most_threshold_seconds']['shots'] == 1
    assert marked['shots'][0]['imaging_quality'] == result['samples'][0]['shots'][1]['imaging_quality']


def test_duplicate_samples_and_invalid_labels_fail(tmp_path, sample):
    pair = write_pair(tmp_path, sample)
    with pytest.raises(ValueError, match='duplicate sample'):
        quality.summarize_pairs([pair, pair], short_shot_max_seconds=17/16)
    record, segments = sample
    parsed = quality.summarize_story(record, segments, short_shot_max_seconds=17/16)
    label = {**{name: record[name] for name in quality.SAMPLE_FIELDS}, 'shot': 8}
    labels = tmp_path / 'confirmed.json'
    labels.write_text(json.dumps({'source': 'unit-test annotation', 'shots': [label]}))
    with pytest.raises(ValueError, match='outside sample'):
        quality.noise_labels(labels, [parsed])
    label['shot'] = 2
    labels.write_text(json.dumps({'source': 'unit-test annotation', 'shots': [label, deepcopy(label)]}))
    with pytest.raises(ValueError, match='duplicate noise label'):
        quality.noise_labels(labels, [parsed])


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '16'])
def test_invalid_output_fps_or_duration_threshold_is_rejected(sample, value):
    record, segments = sample
    with pytest.raises(ValueError, match='short_shot_max_seconds'):
        quality.summarize_story(record, segments, short_shot_max_seconds=value)
    record['non_vlm']['fps'] = value
    with pytest.raises(ValueError, match='non_vlm.fps'):
        quality.summarize_story(record, segments, short_shot_max_seconds=17/16)


def test_output_fps_and_cli_duration_threshold_are_required(sample):
    record, segments = sample
    del record['non_vlm']['fps']
    with pytest.raises(KeyError, match='fps'):
        quality.summarize_story(record, segments, short_shot_max_seconds=17/16)
    with pytest.raises(SystemExit) as caught:
        quality.main(['--pair', 'metrics.json', 'segments.json', '--output', 'readout.json'])
    assert caught.value.code == 2
