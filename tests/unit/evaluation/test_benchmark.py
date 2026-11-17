from argparse import Namespace
import json
from pathlib import Path

import pytest

from mspc.evaluation import benchmark as owner
from tests.support.closeai_fakes import fake_chat_completion

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/fixtures/evaluation/story_ratings_sunglasses.json'


def response():
    return json.loads(FIXTURE.read_text())


def row(**values):
    return {"method": "native", "story_id": "lsh-003-b", "input_tier": "shots", "seed": 42,
            "story_request": "Two different women confront each other. The first puts on sunglasses.",
            "video_path": "/videos/lsh-003-b.mp4", "generation_record_path": None, "expected_shot_count": None,
            "expected_cut_times_seconds": None, **values}


def manifest(rows):
    conditions = sorted({(r['input_tier'], r['seed']) for r in rows})
    return {"evaluation_scope": "renderer", "requests": rows, 'comparisons': [
        {'input_tier': tier, 'seed': seed,
         'methods': sorted({r['method'] for r in rows if (r['input_tier'], r['seed']) == (tier, seed)}),
         'story_ids': sorted({r['story_id'] for r in rows if (r['input_tier'], r['seed']) == (tier, seed)})}
        for tier, seed in conditions]}


def test_actual_medium_response_and_invalid_evidence():
    data = response()
    assert [r['score'] for r in owner.validate_ratings(data, 17.75)] == [2, 3, 3, 2, 2]
    data['ratings'][0]['timestamps_seconds'] = [18]
    with pytest.raises(ValueError, match='within'):
        owner.validate_ratings(data, 17.75)


def test_tiers_and_seeds_have_distinct_storage_and_summary(tmp_path):
    rows = [row(), row(input_tier='paragraph'), row(seed=43)]
    source = tmp_path / 'manifest.json'
    owner.write_json(source, manifest(rows))
    loaded = owner.load_manifest(source)
    paths = [owner.result_path(tmp_path, r) for r in rows]
    assert len(set(paths)) == 3
    for r, path in zip(rows, paths):
        owner.write_json(path, {**r, 'judge': {**response(), 'status': 'rated'}, 'media': {'duration_seconds': 17.75}})
    groups = owner.summarize(loaded, tmp_path)['groups']
    assert len(groups) == 3 and all(g['stories_expected'] == 1 for g in groups)


def test_methods_cannot_be_scored_against_different_requests(tmp_path):
    source = tmp_path / 'manifest.json'
    owner.write_json(source, manifest([row(), row(method='other', story_request='Different story')]))
    with pytest.raises(ValueError, match='different evaluation requests'):
        owner.load_manifest(source)


def test_partial_coverage_and_not_applicable_are_explicit(tmp_path):
    rows = [row(story_id=f'story-{i}') for i in range(4)]
    for i in (0, 1):
        data = response()
        if i:
            data['ratings'][0].update(status='not_applicable', score=None)
            data['ratings'][1]['score'] = 1
        owner.write_json(owner.result_path(tmp_path, rows[i]), {
            **rows[i], 'judge': {**data, 'status': 'rated'}, 'media': {'duration_seconds': 17.75}})
    owner.write_json(owner.result_path(tmp_path, rows[2]), {**rows[2], 'judge': {'status': 'format_failed'}})
    group, = owner.summarize(manifest(rows), tmp_path)['groups']
    assert group['dimensions']['object_consistency']['status_counts'] == {'rated': 2, 'format_failed': 1, 'pending': 1}
    assert group['dimensions']['character_consistency']['n'] == 1
    assert group['dimensions']['character_consistency']['status_counts']['not_applicable'] == 1
    assert group['dimensions']['object_consistency']['mean'] == 2


def args_for(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'local media placeholder; probe is mocked')
    source = tmp_path / 'manifest.json'
    owner.write_json(source, manifest([row(video_path=str(video))]))
    key = tmp_path / 'key.txt'
    key.write_text('offline-fixture')
    return Namespace(manifest=source, output=tmp_path / 'out', config=ROOT / 'configs/eval/benchmark.json', components='judge',
        prompt=ROOT / 'configs/eval/story_ratings_prompt.txt', response_format=ROOT / 'configs/eval/story_ratings_response.json',
        api_key_file=key, budget_usd='1', proxy='http://offline.invalid', ffprobe='ffprobe', limit=1)


def install_transport(monkeypatch, transport):
    client = owner.OpenRouterClient
    monkeypatch.setattr(owner, 'OpenRouterClient', lambda **kwargs: client(**kwargs, transport=transport))
    monkeypatch.setattr(owner, 'inspect_video', lambda *args: {'duration_seconds': 17.75})


def test_offline_dispatch_keeps_actual_response_and_resumes_without_resending(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    requests = []
    def transport(**request):
        requests.append(request)
        # The in-flight record already exists before the provider responds.
        sample = owner.load_manifest(args.manifest)['requests'][0]
        assert json.loads(owner.result_path(args.output, sample).read_text())['judge']['status'] == 'request_started'
        return json.loads(json.dumps(fake_chat_completion(json.dumps(response()), prompt_tokens=2894, completion_tokens=1291,
            prompt_tokens_details={'video_tokens': 1188}), default=vars))
    install_transport(monkeypatch, transport)
    owner.score(args)
    owner.score(args)
    assert len(requests) == 1
    assert 'temperature' not in requests[0]
    assert requests[0]['messages'][0]['content'][0]['video_url']['processing'] == 'static'
    group, = json.loads((args.output / 'summary.json').read_text())['groups']
    assert group['dimensions']['event_completion']['n'] == 1
    sample = owner.load_manifest(args.manifest)['requests'][0]
    saved = json.loads(owner.result_path(args.output, sample).read_text())['judge']
    assert saved['raw_response'] and saved['cost_usd'] and saved['ratings'] == response()['ratings']


def test_missing_usage_keeps_raw_response_and_blocks_implicit_retry(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    def transport(**request):
        raw = json.loads(json.dumps(fake_chat_completion(json.dumps(response()), prompt_tokens=2894, completion_tokens=1291), default=vars))
        raw['usage'] = None
        return raw
    install_transport(monkeypatch, transport)
    with pytest.raises(owner.OpenRouterResponseError):
        owner.score(args)
    sample = owner.load_manifest(args.manifest)['requests'][0]
    saved = json.loads(owner.result_path(args.output, sample).read_text())['judge']
    assert saved['status'] == 'response_failed' and saved['raw_response']
    assert saved['cost_usd'] is None
    with pytest.raises(ValueError, match='unresolved billed request'):
        owner.score(args)


def test_requirement_applicability_disagreement_is_reported_across_methods(tmp_path):
    rows = [row(), row(method='other')]
    for i, sample in enumerate(rows):
        data = response()
        if i:
            data['ratings'][0].update(status='not_applicable', score=None)
        owner.write_json(owner.result_path(tmp_path, sample), {
            **sample, 'judge': {**data, 'status': 'rated'}, 'media': {'duration_seconds': 17.75}})
    result = owner.summarize(manifest(rows), tmp_path)
    disagreement, = result['applicability_disagreements']
    assert disagreement['dimension'] == 'character_consistency'
    assert disagreement['methods'] == {'native': 'rated', 'other': 'not_applicable'}


def test_dropped_video_is_not_scored_even_when_the_judge_answers(tmp_path, monkeypatch):
    # 10-19 probe: an object-valued processing field made OpenRouter drop the video (video_tokens 0),
    # yet the judge still returned five confident ratings from the text alone.
    args = args_for(tmp_path)
    requests = []
    def transport(**request):
        requests.append(request)
        return json.loads(json.dumps(fake_chat_completion(json.dumps(response()), prompt_tokens=2640, completion_tokens=1945,
            prompt_tokens_details={'video_tokens': 0}), default=vars))
    install_transport(monkeypatch, transport)
    with pytest.raises(owner.VideoNotIngestedError):
        owner.score(args)
    sample = owner.load_manifest(args.manifest)['requests'][0]
    saved = json.loads(owner.result_path(args.output, sample).read_text())['judge']
    assert saved['status'] == 'video_not_ingested' and saved['cost_usd'] and 'ratings' not in saved
    owner.score(args)
    assert len(requests) == 1
    group, = json.loads((args.output / 'summary.json').read_text())['groups']
    assert group['dimensions']['event_completion']['status_counts'] == {'video_not_ingested': 1}
