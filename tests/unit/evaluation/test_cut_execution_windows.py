"""切镜准备同时输出整片判官复用的 TransNetV2 对齐时间窗。"""
from mspc.evaluation import cut_execution as c


def test_prepare_writes_aligned_windows_even_for_stories_without_cut_items(tmp_path):
    root, media_root = tmp_path / 'judge', tmp_path / 'media'
    items_path, cuts_path = tmp_path / 'items.json', tmp_path / 'cuts.json'
    row, sid = 'row', 'story'
    c.g.write_json(items_path, {'stories': {sid: {'items': []}}})
    c.g.write_json(cuts_path, {'rows': {row: {sid: {'detected_cut_times_seconds': [0.9, 2.1]}}}})
    c.g.write_json(media_root / 'rows' / row / 'prepared.json', {'stories': [{'story_id': sid}]})
    c.g.write_json(media_root / 'rows' / row / 'media-map.json', {
        sid: {'shot_frames': [10, 10, 10], 'probe': {'fps': 10, 'frames': 30},
              'remote_video_path': '/video.mp4'}})

    assert c.prepare(root, media_root, items_path, [cuts_path], [row], None) == {
        row: {'items': 0, 'undecided': 0, 'by': {}}}
    assert c.load(root / 'shot-windows' / f'{row}.json') == {sid: [[0.0, 0.9], [0.9, 2.1], [2.1, 3.0]]}
