"""同故事原片并排验收；仅服务显式目录中的产物，意见另存不改生成记录。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse


def read(path):
    return json.loads(Path(path).read_text())


def byte_range(value, size):
    if value is None:
        return 0, size - 1
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
    if not match or not any(match.groups()):
        raise ValueError('invalid byte range')
    first, last = match.groups()
    start = int(first) if first else max(0, size - int(last))
    end = min(size - 1, int(last)) if first and last else size - 1
    if start > end or start >= size:
        raise ValueError('range outside asset')
    return start, end


class Review:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.assets = {}
        self.details = {}

    def catalog(self):
        config = self.config
        source = read(config['stories'])
        stories = {s['story_id']: s for s in source['stories']}
        entries, details, assets = {}, {}, {}
        for origin in config['sources']:
            for path in sorted(Path(origin['root']).rglob('generation.json')):
                sid = path.parent.name
                if sid not in stories or stories[sid]['split'] != origin['split']:
                    continue
                record = read(path)
                key = sid + '/' + origin['method']
                video = path.parent / 'video.mp4'
                partial = path.parent / 'segments/0001.mp4'
                selected = video if record['status'] == 'completed' and video.is_file() else partial
                row = {'story_id': sid, 'method': origin['method'], 'status': record['status'],
                       'note': origin['note'], 'error': record.get('error', record.get('process_error')),
                       'source': str(path), 'video_path': str(selected) if selected.is_file() else None,
                       'partial': record['status'] != 'completed', 'attempts': []}
                if key in entries:
                    old = entries[key]
                    if old['status'] == 'completed' and row['status'] != 'completed':
                        old['attempts'].append(str(path))
                        continue
                    row['attempts'] = old['attempts'] + [old['source']]
                entries[key] = row
                detail = {'generation_path': str(path), 'generation': record,
                          'inputs': read(path.parent / 'inputs.json') if (path.parent / 'inputs.json').exists() else None,
                          'segments': read(path.parent / 'segments.json') if (path.parent / 'segments.json').exists() else None}
                details[key] = detail
        for key, row in entries.items():
            if row['video_path']:
                assets[key] = Path(row['video_path'])
                row['asset'] = key
            details[key]['previous_attempts'] = row['attempts']
        result = []
        for sid, story in stories.items():
            if story['split'] == 'dev' and sid not in config['development_stories']:
                continue
            methods = [m for key, m in entries.items() if m['story_id'] == sid]
            result.append({'story_id': sid, 'split': story['split'], 'fps': story['fps'],
                           'target_seconds': story['target_seconds'], 'request': story['inputs']['shots']['text'],
                           'methods': methods})
        with self.lock:
            self.assets, self.details = assets, details
        decisions = Path(config['decisions'])
        latest = {}
        if decisions.exists():
            for line in decisions.read_text().splitlines():
                d = json.loads(line)
                latest[d['story_id'] + '/' + d['method']] = d
        return {'stories': result, 'methods': config['methods'], 'decisions': latest,
                'updated_at': datetime.now(timezone.utc).isoformat()}

    def decide(self, data):
        key = data['story_id'] + '/' + data['method']
        if key not in self.details or data['decision'] not in ('accepted', 'rejected', 'needs_review'):
            raise ValueError('unknown result or decision')
        if not isinstance(data['note'], str) or len(data['note']) > 10000:
            raise ValueError('invalid review note')
        row = {k: data[k] for k in ('story_id', 'method', 'decision', 'note')}
        row.update(saved_at=datetime.now(timezone.utc).isoformat(),
                   video_path=str(self.assets[key]) if key in self.assets else None,
                   generation_path=self.details[key].get('generation_path'))
        path = Path(self.config['decisions'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row


def handler(review):
    class Handler(BaseHTTPRequestHandler):
        def json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/catalog':
                    self.json(review.catalog())
                elif parsed.path == '/api/detail':
                    self.json(review.details[query['key'][0]])
                elif parsed.path == '/media':
                    self.media(review.assets[query['key'][0]])
                elif parsed.path == '/':
                    body = Path(__file__).with_name('review.html').read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)
            except (KeyError, ValueError, OSError) as error:
                self.json({'error': str(error)}, 400)

        def do_POST(self):
            # 仅接受本工具同源的明确点击，不允许别的网页跨站提交验收。
            origin = self.headers.get('Origin')
            if origin != 'http://' + self.headers.get('Host', ''):
                self.send_error(403)
                return
            length = int(self.headers.get('Content-Length', '0'))
            if self.path != '/api/decisions' or not 0 < length < 16000:
                self.send_error(400)
                return
            try:
                self.json(review.decide(json.loads(self.rfile.read(length))))
            except (ValueError, KeyError, OSError) as error:
                self.json({'error': str(error)}, 400)

        def media(self, path):
            size = path.stat().st_size
            header = self.headers.get('Range')
            try:
                start, end = byte_range(header, size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
            self.send_response(206 if header else 200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if header:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with path.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        data = stream.read(min(256 * 1024, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # 浏览器主动取消旧的seek请求。

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(os.path.expandvars(args.config.read_text()))
    if '${' in json.dumps(config):
        raise ValueError('unresolved review paths')
    review = Review(config)
    review.catalog()
    ThreadingHTTPServer(('127.0.0.1', args.port), handler(review)).serve_forever()


if __name__ == '__main__':
    main()
