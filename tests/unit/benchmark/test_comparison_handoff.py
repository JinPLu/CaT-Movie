import json

from mspc.comparison_methods import handoff


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_finished_worker_hands_its_card_to_the_next_method(tmp_path, monkeypatch):
    control, target = tmp_path / 'control', tmp_path / 'target'
    write(control / 'manifests/leases.json', {'leases': [{'lease_id': 'L', 'released_at': None}]})
    write(control / 'logs/workers/holo-6681-0.json', {'status': 'finished'})
    write(target / 'manifests/leases.json', {'leases': []})
    write(target / 'inputs/queue.json', {'methods': {'m': {'queue': str(target / 'inputs/q.json')}}})
    write(target / 'inputs/q.json', [{'story_id': 's1', 'generation_record_path': str(target / 'g1.json')}])
    plan = {'shared_root': '/shared', 'runs': {'sd': {'run_root': str(target), 'method': 'm', 'python': 'py',
                                                       'control_src': '/src'}},
            'slots': [{'lease_id': 'L', 'server_id': 'server', 'ssh_port': 6681, 'device': '0',
                       'after_worker': 'holo-6681-0', 'chain': ['sd']}]}
    launched = []
    monkeypatch.setattr(handoff.subprocess, 'Popen', lambda argv, **kwargs: launched.append((argv, kwargs['env'])))
    state = {}
    assert handoff.step(plan, control, 6681, state, lambda message: None)
    argv, env = launched[0]
    assert argv[-6:] == ['--device', '0', '--lease-id', 'L-0', '--worker', 'sd-6681-0']
    assert env['CUDA_VISIBLE_DEVICES'] == '0' and env['PYTHONPATH'] == '/src'
    row, = json.loads((target / 'manifests/leases.json').read_text())['leases']
    assert row['lease_id'] == 'L-0' and row['scheduler_lease'] == 'L' and row['released_at'] is None
    assert json.loads((control / 'manifests/leases.json').read_text())['leases'][0]['released_at'] is not None
    assert state['L-0'][-1]['worker'] == 'sd-6681-0'
    write(target / 'logs/workers/sd-6681-0.json', {'status': 'finished'})
    assert handoff.step(plan, control, 6681, state, lambda message: None)
    assert state['L-0'][-1]['done'] and json.loads((target / 'manifests/leases.json').read_text())['leases'][0]['released_at']
    assert not handoff.step(plan, control, 6681, state, lambda message: None)
