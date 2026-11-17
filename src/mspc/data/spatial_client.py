"""空间生产与在线实际帧观察共用的本地vLLM transport、坐标规范化与原子JSON写入。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 多个分片会同时写共用 config；各次写入必须拥有独立临时文件。
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=p.parent,
                                     prefix=p.name+'.', suffix='.partial', delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(obj, ensure_ascii=False))
    try:
        tmp.replace(p)
    finally:
        tmp.unlink(missing_ok=True)

def normalized_corners(box):
    """Qwen 的 0–1000 xyxy → SAM 实例框的 0–1 两角，不裁剪非法答案。"""
    import math
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                            or not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
        raise ValueError('invalid bbox_2d')
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError('bbox_2d must have positive width and height')
    return [[x1/1000, y1/1000], [x2/1000, y2/1000]]


class LocalClient:
    """仅访问本机 vLLM；原始回答与 usage 留在独立运行目录。"""
    def __init__(self, work, config, shard):
        from urllib.parse import urlsplit
        self.work, self.config = work, config['locate']
        self.endpoint = self.config['endpoints'][shard % len(self.config['endpoints'])]
        endpoint = urlsplit(self.endpoint)
        if endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('local locator requires loopback HTTP endpoint')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, tag, body):
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json'})
        tick = time.monotonic()
        try:
            with self.opener.open(req, timeout=self.config['timeout_seconds']) as response:
                obj = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            save(self.work/'responses'/(tag+'-error.json'), {'http_status':error.code, 'detail':detail})
            raise ValueError('local vLLM rejected request: '+detail) from error
        choice = obj['choices'][0]
        save(self.work/'responses'/(tag+'.json'), dict(response=obj, seconds=time.monotonic()-tick,
             content=choice['message'].get('content'), finish_reason=choice['finish_reason'],
             endpoint=self.endpoint, completed_at=datetime.now(timezone.utc).isoformat()))
        if choice['finish_reason'] != 'stop':
            raise ValueError('local locator unfinished: '+choice['finish_reason'])
        return choice['message']['content']


