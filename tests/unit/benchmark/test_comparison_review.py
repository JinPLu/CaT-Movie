"""视频拖动播放和验收意见的真实 HTTP 边界。"""
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from mspc.comparison_methods.review import Review, byte_range, handler


@pytest.mark.parametrize(('header', 'expected'), [
    (None, (0, 99)), ('bytes=10-19', (10, 19)),
    ('bytes=90-', (90, 99)), ('bytes=-12', (88, 99)),
    ('bytes=90-999', (90, 99)),
])
def test_browser_byte_ranges(header, expected):
    assert byte_range(header, 100) == expected


@pytest.mark.parametrize('header', ['bytes=100-', 'bytes=10-9', 'bytes=-0', 'bytes=', 'bytes=0-1,3-4'])
def test_invalid_ranges(header):
    with pytest.raises(ValueError):
        byte_range(header, 100)


def test_http_seek_and_separate_human_decisions(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(bytes(range(100)))
    decision_file = tmp_path / 'review/decisions.jsonl'
    review = Review({'decisions': str(decision_file)})
    review.assets = {'story/method': video}
    review.details = {'story/method': {'generation_path': 'original/generation.json'}}
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(review))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = '127.0.0.1:' + str(server.server_port)
    client = HTTPConnection(host)
    try:
        client.request('GET', '/media?key=story%2Fmethod', headers={'Range': 'bytes=20-29'})
        response = client.getresponse()
        assert response.status == 206
        assert response.getheader('Content-Range') == 'bytes 20-29/100'
        assert response.read() == bytes(range(20, 30))
        client.request('GET', '/media?key=/etc/passwd')
        response = client.getresponse()
        assert response.status == 400
        response.read()
        payload = json.dumps({'story_id': 'story', 'method': 'method', 'decision': 'needs_review', 'note': '真实验收另存'})
        client.request('POST', '/api/decisions', payload, {'Origin': 'http://another-site'})
        response = client.getresponse()
        assert response.status == 403
        response.read()
        assert not decision_file.exists()
        client.request('POST', '/api/decisions', payload, {'Origin': 'http://' + host})
        response = client.getresponse()
        assert response.status == 200
        row = json.loads(response.read())
        assert row['video_path'] == str(video)
        assert row['generation_path'] == 'original/generation.json'
        assert json.loads(decision_file.read_text()) == row
        assert video.read_bytes() == bytes(range(100))
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()
