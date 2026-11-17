from datetime import datetime, timezone
import json
import sys

from mspc.comparison_methods import queue
from mspc.comparison_methods.queue import remaining_seconds
from mspc.evaluation.benchmark import write_json


def test_budget_counts_released_and_still_held_devices():
    rows = [
        {'method': 'm', 'held_since': '2026-10-26T10:00:00+00:00', 'released_at': '2026-10-26T11:00:00+00:00', 'gpu_count': 2},
        {'method': 'm', 'held_since': '2026-10-26T11:00:00+00:00', 'released_at': None, 'gpu_count': 2},
        {'method': 'other', 'held_since': '2026-10-26T10:00:00+00:00', 'released_at': None, 'gpu_count': 8},
    ]
    # 先前0.5卡时＋已释放2卡时＋在途1卡时；余量给两卡均分，留两分钟释放。
    value = remaining_seconds(rows, 'm', 5, .5, datetime(2026, 10, 26, 11, 30, tzinfo=timezone.utc))
    assert value == 2580


def test_same_host_lease_reallocation_selects_active_device_and_counts_prior_phases(tmp_path, monkeypatch):
    holocine, storymem, multishotmaster = (
        'official_holocine_14b_full', 'official_storymem_14b', 'official_multishotmaster_1_3b')
    lease_id = 'core-baselines-eight-a100'
    start, switch, current = ('2026-11-08T10:00:00+00:00', '2026-11-08T11:00:00+00:00',
                              datetime(2026, 11, 8, 11, 30, tzinfo=timezone.utc))
    # 同一个八卡Broker租约：原3+3+2，MSM排空后两个设备分别转入HoloCine与StoryMem。
    rows = [
        dict(lease_id=lease_id, method=holocine, devices=['GPU-0', 'GPU-1', 'GPU-2'],
             gpu_count=3, held_since=start, released_at=switch),
        dict(lease_id=lease_id, method=storymem, devices=['GPU-3', 'GPU-4', 'GPU-5'],
             gpu_count=3, held_since=start, released_at=switch),
        dict(lease_id=lease_id, method=multishotmaster, devices=['GPU-6', 'GPU-7'],
             gpu_count=2, held_since=start, released_at=switch),
        dict(lease_id=lease_id, method=holocine, devices=['GPU-0', 'GPU-1', 'GPU-2', 'GPU-6'],
             gpu_count=4, held_since=switch, released_at=None),
        dict(lease_id=lease_id, method=storymem, devices=['GPU-3', 'GPU-4', 'GPU-5', 'GPU-7'],
             gpu_count=4, held_since=switch, released_at=None),
    ]
    assert remaining_seconds(rows, holocine, 10, .5, current) == 3930
    assert remaining_seconds(rows, storymem, 10, 0, current) == 4380
    assert remaining_seconds(rows, multishotmaster, 5, 0, current) == 10800
    task = dict(story_id='reallocated-story', generation_record_path=str(tmp_path / 'story/generation.json'))
    queue_path = tmp_path / 'inputs/holocine-queue.json'
    write_json(queue_path, [task])
    write_json(tmp_path / 'inputs/queue.json', {'methods': {holocine: dict(
        authorized=True, queue=str(queue_path), max_workers=5, cap_gpu_hours=10,
        prior_gpu_hours=.5, story_limit_seconds=20000)}})
    write_json(tmp_path / 'inputs/dispatch.json', {})
    write_json(tmp_path / 'manifests/leases.json', {'leases': rows})
    (tmp_path / 'logs').mkdir()
    calls = []

    class Renderer:
        def __init__(self, dispatch, method, device):
            assert method == holocine and device == 'GPU-6'
        def close(self):
            calls.append('closed')

    def run(output, dispatch, selected, device, selected_lease, limit, budget, **kwargs):
        assert selected == task and device == 'GPU-6' and selected_lease == lease_id
        assert budget() == 3930
        calls.append('rendered')
        return {'status': 'completed'}

    monkeypatch.setattr(queue, 'ResidentRenderer', Renderer)
    monkeypatch.setattr(queue, 'run_task', run)
    monkeypatch.setattr(queue, 'now', lambda: current.isoformat())
    monkeypatch.setattr(sys, 'argv', ['queue', '--run-root', str(tmp_path), '--method', holocine,
                                    '--device', 'GPU-6', '--lease-id', lease_id, '--worker', 'moved-card'])
    queue.main()
    state = json.loads((tmp_path / 'logs/workers/moved-card.json').read_text())
    assert state['status'] == 'finished' and state['completed'] == 1
    assert calls == ['rendered', 'closed']
