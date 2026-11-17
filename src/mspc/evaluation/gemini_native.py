"""Gemini 3.8 Flash 的 Google 原生接口（OpenRouter 自 11-06 15:10 起返回 403）。

只用标准库走 REST：
- 认证头 ``x-goog-api-key``；密钥从本机文件读进内存，不进 URL、命令行、日志与回执（回执里出现即替换）。
- 经显式本机 HTTP 代理出网；不跟随重定向；不自动重试（Batch 创建不幂等，重发会多建一个任务）。
- 判官要看的成片与帧在主服务器主机上写判分桶、本机登记 ``gs://``（``Storage``，服务器出口过不了 Gemini 地区闸）；
  批任务输入 JSONL 走 Files API（可恢复上传，用完删除），生成走 ``generateContent``；大批离线判分走
  ``batchGenerateContent``（输入 JSONL 也经 Files 上传），按官方 Batch 价计费估算。
- 正文取 ``candidates[0].content.parts`` 里非 thought 的 text，要求 ``finishReason == STOP``；费用按
  ``usageMetadata`` 的 ``promptTokenCount`` 与 ``candidatesTokenCount + thoughtsTokenCount`` 估算，不是账单实付。
"""
from __future__ import annotations

import base64
import json
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

BASE = 'https://generativelanguage.googleapis.com'
HOST = 'generativelanguage.googleapis.com'
MODEL = 'gemini-3.8-flash'
# 美元／百万 token，官方价目至 2026-12-31 有效（2027-01-01 起翻倍）；输出含 thinking。
PRICES = {'standard': (D('0.75'), D('3.75')), 'batch': (D('0.375'), D('1.875'))}
PRICES_VALID_THROUGH = '2026-12-31'
FILE_STATES = ('PROCESSING', 'ACTIVE', 'FAILED')
BATCH_RUNNING = ('BATCH_STATE_PENDING', 'BATCH_STATE_RUNNING')
# Files API 每个 Google Cloud 项目的存储总额（官方 Files 页）；未结束批任务引用的上传都占着它。
FILES_STORAGE_BYTES = 20 * 10**9
# 主服务器主机上的项目代理文件：服务器写判分桶走它（该出口过不了 Gemini 地区闸，只用于写桶）。
HOST_PROXY_FILE = '${MSPC_SHARED_ROOT}/OoVMetric/resource/proxy.txt'
BATCH_DONE = ('BATCH_STATE_SUCCEEDED', 'BATCH_STATE_FAILED', 'BATCH_STATE_CANCELLED', 'BATCH_STATE_EXPIRED')


class GeminiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f'HTTP {status}: {body[:500]}')
        self.status, self.body = status, body


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_key(path: Path) -> str:
    key = Path(path).read_text().strip()
    if not key:
        raise ValueError(f'empty key file: {path}')
    return key


class Gemini:
    def __init__(self, key: str, proxy: str, timeout: int = 180):
        self._key = key
        self._client = build_opener(ProxyHandler({'https': proxy}), _NoRedirect())
        self.timeout = timeout

    def _scrub(self, raw: bytes) -> bytes:
        return raw.replace(self._key.encode(), b'[REDACTED]')

    def http(self, url: str, payload: dict | bytes | None = None, headers: dict | None = None,
             method: str | None = None, timeout: int | None = None) -> tuple[bytes, dict]:
        if urlsplit(url).hostname != HOST:
            raise ValueError(f'unexpected API host: {url}')
        sent = {'x-goog-api-key': self._key, **(headers or {})}
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode()
            sent.setdefault('Content-Type', 'application/json')
        request = Request(url, data=payload, headers=sent, method=method)
        try:
            with self._client.open(request, timeout=timeout or self.timeout) as response:
                return self._scrub(response.read()), {k.lower(): v for k, v in response.headers.items()}
        except HTTPError as error:
            raise GeminiError(error.code, self._scrub(error.read()).decode(errors='replace')) from None

    def json(self, url: str, payload: dict | None = None, method: str | None = None) -> dict:
        raw, _ = self.http(url, payload, method=method)
        return json.loads(raw) if raw else {}

    def upload(self, path: Path, mime: str, display_name: str) -> dict:
        """可恢复上传，一次传完；返回 Files 资源（含 name、uri、state）。"""
        size = Path(path).stat().st_size
        _, headers = self.http(BASE + '/upload/v1beta/files', {'file': {'display_name': display_name}},
                               {'X-Goog-Upload-Protocol': 'resumable', 'X-Goog-Upload-Command': 'start',
                                'X-Goog-Upload-Header-Content-Length': str(size),
                                'X-Goog-Upload-Header-Content-Type': mime}, method='POST')
        # 大文件（内联图片的批任务 JSONL 可达几百 MB）按经本机代理约 0.5 MB/s 的下限放宽超时。
        raw, _ = self.http(headers['x-goog-upload-url'], Path(path).read_bytes(),
                           {'Content-Type': mime, 'X-Goog-Upload-Offset': '0',
                            'X-Goog-Upload-Command': 'upload, finalize'}, method='POST',
                           timeout=max(self.timeout, size // 500_000))
        file = json.loads(raw)['file']
        if int(file['sizeBytes']) != size:
            raise ValueError(f"{path}: uploaded {file['sizeBytes']} of {size} bytes")
        return file

    def register(self, uris: list[str]) -> list[dict]:
        """把桶里的对象登记成 Files 资源（``files:register``）；返回与 ``uris`` 同序的资源。Gemini 接口不收服务账号
        令牌（实测 403「restricted with service accounts. Use authorization keys」），用 API 密钥；读桶的是密钥绑定的
        服务账号（AI Studio 建密钥时生成的 ``ais-gemini-key-…``），桶上要给它 Storage Object Viewer。"""
        files = self.json(f'{BASE}/v1beta/files:register', {'uris': uris}, method='POST')['files']
        if len(files) != len(uris):
            raise ValueError(f'registered {len(files)} of {len(uris)} objects')
        return files

    def register_all(self, uris: list[str], chunk: int = 100, workers: int = 8) -> dict[str, dict]:
        """去重后按 ``chunk`` 个一组并发登记，返回 ``gs://`` 地址 → Files 资源。官方：一组里有一个登记失败整组失败。"""
        unique = list(dict.fromkeys(uris))
        groups = [unique[i:i + chunk] for i in range(0, len(unique), chunk)]
        with ThreadPoolExecutor(workers) as pool:
            return dict(zip(unique, [f for files in pool.map(self.register, groups) for f in files]))

    def file(self, name: str) -> dict:
        return self.json(f'{BASE}/v1beta/{name}')

    def wait_active(self, name: str, timeout_seconds: int = 600, interval: int = 3) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while True:
            file = self.file(name)
            if file['state'] not in FILE_STATES:
                raise ValueError(f"{name}: unknown file state {file['state']}")
            if file['state'] != 'PROCESSING':
                if file['state'] != 'ACTIVE':
                    raise RuntimeError(f"{name}: file processing ended in {file['state']}")
                return file
            if time.monotonic() > deadline:
                raise TimeoutError(f'{name}: still PROCESSING after {timeout_seconds}s')
            time.sleep(interval)

    def list_files(self) -> list[dict]:
        files, token = [], ''
        while True:
            page = self.json(f'{BASE}/v1beta/files?pageSize=100' + (f'&pageToken={token}' if token else ''))
            files += page.get('files', [])
            token = page.get('nextPageToken')
            if not token:
                return files

    def delete(self, name: str) -> None:
        self.http(f'{BASE}/v1beta/{name}', method='DELETE')

    def generate(self, request: dict, model: str = MODEL) -> dict:
        return self.json(f'{BASE}/v1beta/models/{model}:generateContent', request, method='POST')

    def create_batch(self, input_file: str, display_name: str, model: str = MODEL) -> dict:
        """``input_file`` 是已上传 JSONL 的 Files 名（files/...）。创建不幂等：调用方须在发出前记账、发出后立刻落盘任务名。"""
        body = {'batch': {'displayName': display_name, 'inputConfig': {'fileName': input_file}}}
        return self.json(f'{BASE}/v1beta/models/{model}:batchGenerateContent', body, method='POST')

    def batch(self, name: str) -> dict:
        return self.json(f'{BASE}/v1beta/{name}')

    def download(self, file_name: str) -> bytes:
        raw, _ = self.http(f'{BASE}/download/v1beta/{file_name}:download?alt=media')
        return raw


class Storage:
    """判分用的 Cloud Storage 桶：成片在主服务器主机上写桶（``gcs_host_upload``），
    本机再用 API 密钥登记给 Gemini（``Gemini.register``）。上传服务账号的 JSON 密钥只在本机签写桶令牌；送上服务器的只有 1 小时有效的访问令牌，走 ssh 标准输入。
    桶里对象由桶的生命周期规则 2 天后删除。"""
    TOKEN_URL = 'https://oauth2.googleapis.com/token'
    SCOPE = 'https://www.googleapis.com/auth/devstorage.read_write'

    def __init__(self, service_account_file: Path, bucket: str, proxy: str):
        account = json.loads(Path(service_account_file).read_text())
        if account.get('type') != 'service_account':
            raise ValueError(f'{service_account_file}: not a service account key')
        self._account, self.bucket = account, bucket
        self._client = build_opener(ProxyHandler({'https': proxy}), _NoRedirect())
        self._token, self._expires, self._lock = '', 0.0, threading.Lock()

    def token(self) -> str:
        """缓存的访问令牌，剩余不足 20 分钟就重签（一次服务器上传给 15 分钟超时）。"""
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        with self._lock:
            if time.time() < self._expires - 1200:
                return self._token
            def b64(data: bytes) -> bytes:
                return base64.urlsafe_b64encode(data).rstrip(b'=')
            now = int(time.time())
            head = b64(json.dumps({'alg': 'RS256', 'typ': 'JWT', 'kid': self._account['private_key_id']}).encode())
            claim = b64(json.dumps({'iss': self._account['client_email'], 'scope': self.SCOPE, 'aud': self.TOKEN_URL,
                                    'iat': now, 'exp': now + 3600}).encode())
            key = serialization.load_pem_private_key(self._account['private_key'].encode(), None)
            signed = head + b'.' + claim + b'.' + b64(key.sign(head + b'.' + claim, padding.PKCS1v15(), hashes.SHA256()))
            body = urlencode({'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer', 'assertion': signed.decode()})
            try:
                with self._client.open(Request(self.TOKEN_URL, data=body.encode()), timeout=60) as response:
                    answer = json.loads(response.read())
            except HTTPError as error:
                raise GeminiError(error.code, error.read().decode(errors='replace')) from None
            self._token, self._expires = answer['access_token'], now + int(answer['expires_in'])
            return self._token

    def put_from_host(self, ssh: list[str], python: str, items: list[dict], cleanup: str | None = None,
                      workers: int = 16, timeout: int = 900) -> list[str]:
        """在远端主机经其项目代理并发写桶（``gcs_host_upload``），返回与 ``items``（{path, object, mime, size 或
        None}）同序的 ``gs://`` 地址；全部成功后删远端目录 ``cleanup``。远端 HTTP 错误抛 ``GeminiError``，
        ssh 或远端进程其他失败抛 ``OSError``（都可按上传失败整体重试，写同名对象会覆盖）。"""
        script = Path(__file__).with_name('gcs_host_upload.py').read_text()
        loader = ("import sys,json; a=json.loads(sys.stdin.readline()); "
                  "exec(compile(a.pop('script'),'gcs_host_upload','exec'),{'__name__':'__host__','ARGS':a})")
        for attempt in range(3):  # 写桶不计费、同名覆盖；ssh 或代理偶有断开
            token = self.token()
            payload = json.dumps({'script': script, 'token': token, 'proxy_file': HOST_PROXY_FILE,
                                  'bucket': self.bucket, 'items': items, 'cleanup': cleanup, 'workers': workers,
                                  'timeout': timeout})
            try:
                result = subprocess.run([*ssh, f'{python} -I -c {shlex.quote(loader)}'], input=payload + '\n',
                                        capture_output=True, text=True, timeout=timeout + 600)
            except subprocess.TimeoutExpired:
                if attempt == 2:
                    raise
                continue
            lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
            answer = json.loads(lines[-1]) if lines else {}
            if result.returncode == 0 and 'objects' in answer:
                if [o['name'] for o in answer['objects']] != [i['object'] for i in items]:
                    raise ValueError('host upload returned objects out of order')
                return [f"gs://{self.bucket}/{i['object']}" for i in items]
            error = (GeminiError(answer['status'], answer['body']) if answer.get('status') else
                     OSError(f"host upload of {len(items)} files exit {result.returncode}: "
                             f"{answer or result.stderr.replace(token, '[REDACTED]')[-500:]}"))
            if isinstance(error, GeminiError) and error.status < 500 or attempt == 2:
                raise error
        raise AssertionError('unreachable')


def video_part(file: dict, fps: float) -> dict:
    if not 0 < fps <= 24:
        raise ValueError(f'fps must be within (0, 24], got {fps}')
    return {'fileData': {'fileUri': file['uri'], 'mimeType': file['mimeType']}, 'videoMetadata': {'fps': fps}}


def file_part(file: dict) -> dict:
    """已上传或登记的文件（判官的帧走 ``gs://`` 登记，不再内联 base64）。"""
    return {'fileData': {'fileUri': file['uri'], 'mimeType': file['mimeType']}}


def json_request(parts: list[dict], schema: dict, *, system: str | None, thinking: str, max_output_tokens: int,
                 media_resolution: str | None) -> dict:
    """一条 user 消息、结构化 JSON 输出；``system`` 是任务、规则与输出要求（官方提示词指南：放 system instruction）。
    不传采样参数（3.8 迁移说明要求移除）。``media_resolution`` 为 None 时用模型默认。"""
    config = {'thinkingConfig': {'thinkingLevel': thinking}, 'maxOutputTokens': max_output_tokens,
              'responseMimeType': 'application/json', 'responseJsonSchema': schema}
    if media_resolution is not None:
        config['mediaResolution'] = media_resolution
    body = {'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': config}
    if system is not None:
        body['systemInstruction'] = {'parts': [{'text': system}]}
    return body


def answer_json(response: dict) -> dict:
    """正文 JSON；没有候选、没停在 STOP 或正文不是 JSON 都报错（调用方记格式失败）。"""
    candidates = response.get('candidates') or []
    if not candidates:
        raise ValueError(f"no candidates; promptFeedback={response.get('promptFeedback')}")
    candidate = candidates[0]
    if candidate.get('finishReason') != 'STOP':
        raise ValueError(f"finishReason {candidate.get('finishReason')}")
    text = ''.join(p['text'] for p in candidate['content']['parts'] if 'text' in p and not p.get('thought', False))
    return json.loads(text)


def modality_tokens(usage: dict, modality: str) -> int:
    return sum(d['tokenCount'] for d in usage.get('promptTokensDetails', []) if d['modality'] == modality)


def cost(usage: dict, tier: str) -> D:
    """按官方价目估算美元（非账单实付）。Batch 条目的 ``serviceTier`` 也写 STANDARD，档位要由调用方按提交方式给。"""
    inp, out = PRICES[tier]
    # 没有正文的回执（被拦下、没写出答案）usage 里不带 candidatesTokenCount，记 0。
    output = usage.get('candidatesTokenCount', 0) + usage.get('thoughtsTokenCount', 0)
    return (usage['promptTokenCount'] * inp + output * out) / D(1_000_000)


def reserve(input_tokens: int, output_tokens: int, tier: str) -> D:
    inp, out = PRICES[tier]
    return (input_tokens * inp + output_tokens * out) / D(1_000_000)


# ---- 运行根 ``batches/`` 下的批任务账：整片判官与四帧判官共用 ----
# 每个批任务一份记录，创建前就落盘（创建不幂等，崩溃后按记录续，不重发）。费用按批任务记：成功结束的按逐条 usage，
# 没成功结束的按预留额；历史重判的前一次费用仍留在批任务记录里，现行第 11 版不自动重发失败或缺失项。

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
    temp.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def batch_records(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((Path(root) / 'batches').glob('*.json'))]


def batch_spent(root: Path) -> D:
    return sum((D(b['cost_usd']) for b in batch_records(root) if b.get('closed')), D(0))


def batch_open_reserve(root: Path) -> D:
    return sum((D(b['reserve_usd']) for b in batch_records(root) if not b.get('closed')), D(0))


def batch_queued(root: Path) -> set[str]:
    return {k for b in batch_records(root) if not b.get('closed') for k in b['keys']}


def open_batch(root: Path, prefix: str, keys: list[str], reserve: D, cap: D, upload_bytes: int) -> Path:
    """查硬顶（已结束批任务的费用 + 未结束的预留 + 本批预留）与 Files 存储额（未结束批任务的上传 + 本批上传，
    含输入 JSONL 的估计），通过就写一份 ``uploading`` 记录。"""
    committed = batch_spent(root) + batch_open_reserve(root)
    if committed + reserve > cap:
        raise RuntimeError(f'cap {cap} USD: spent and reserved {committed} + this batch {reserve} would exceed it')
    stored = sum(b['upload_bytes'] for b in batch_records(root) if not b.get('closed'))
    if stored + upload_bytes > FILES_STORAGE_BYTES:
        raise RuntimeError(f'Files storage: open batches hold {stored} bytes + this batch {upload_bytes} > {FILES_STORAGE_BYTES}')
    name = f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    path = Path(root) / 'batches' / f'{name}.json'
    write_json(path, {'display_name': name, 'state': 'uploading', 'keys': keys, 'reserve_usd': str(reserve), 'uploads': {},
                  'upload_bytes': upload_bytes, 'started_at': now()})
    return path


def start_batch(client: Gemini, path: Path, requests: list[tuple[str, dict]], uploads: dict[str, str]) -> dict:
    """写 JSONL、上传、建批任务；``uploads`` 是本批引用的已上传文件（结束后一起删）。本地 JSONL 建完即删。"""
    record = json.loads(path.read_text())
    if [key for key, _ in requests] != record['keys']:
        raise ValueError(f'{path}: requests do not match the recorded keys')
    record['uploads'] = uploads
    jsonl = path.with_suffix('.jsonl')
    jsonl.write_text(''.join(json.dumps({'key': key, 'request': request}) + '\n' for key, request in requests))
    record['input_file'] = client.upload(jsonl, 'application/jsonl', record['display_name'])['name']
    record['state'] = 'creating'
    write_json(path, record)
    created = client.create_batch(record['input_file'], record['display_name'])
    record.update(name=created['name'], state=created['metadata']['state'], created_at=now())
    write_json(path, record)
    jsonl.unlink()
    return record


def collect_batches(client: Gemini, root: Path, handle) -> list[dict]:
    """查各未结束批任务。成功结束的下载结果交 ``handle(record, lines) -> (费用, 各状态条数)`` 写回执；
    结束的（无论成败）删掉上传的文件、记费用后关账。"""
    report = []
    for path in sorted((Path(root) / 'batches').glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('closed'):
            continue
        if 'name' not in record:  # 还在上传，或提交中途停了：这里分不出来，只报告，由提交方续或人工清理
            report.append({'batch': record['display_name'], 'state': record['state'], 'created': False})
            continue
        status = client.batch(record['name'])
        state = status['metadata']['state']
        if state in BATCH_RUNNING:
            report.append({'batch': record['name'], 'state': state, 'stats': status['metadata'].get('batchStats')})
            continue
        if state not in BATCH_DONE:
            raise ValueError(f"{record['name']}: unknown state {state}")
        cost, basis, results = D(record['reserve_usd']), 'reserve', {}
        if state == 'BATCH_STATE_SUCCEEDED':
            raw = client.download(status['response']['responsesFile'])
            lines = [json.loads(text) for text in raw.decode().splitlines() if text.strip()]
            (cost, results), basis = handle(record, lines), 'usage'
            missing = set(record['keys']) - {line['key'] for line in lines}
            if missing:
                results['missing'] = len(missing)
        with ThreadPoolExecutor(32) as pool:  # 登记的帧每行七百多张，16 路实测撤销约 72 秒
            list(pool.map(client.delete, [*record['uploads'].values(), record['input_file']]))
        record.update(state=state, closed=True, closed_at=now(), results=results, cost_usd=str(cost), cost_basis=basis)
        write_json(path, record)
        report.append({'batch': record['name'], 'state': state, 'results': results, 'cost_usd': str(cost)})
    return report
