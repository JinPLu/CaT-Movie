"""切镜执行项的判定（第 10 版题库起）：TransNetV2 → 中帧几乎相同判同机位 → 其余判官看四帧比图；只要求有切点的项只看 TransNetV2。

判定链在作者盲标 88 对上验证，与作者一致 80/88、
κ 0.81，留出 40 对 κ 0.78。题由 `question_writer.cut_items` 出，计分在 `checklist_scoring`。

- 定界：计划镜界 ±0.5 秒内有 TransNetV2 检出切点就以最近的检出切点为界，否则沿用计划镜界。相邻项没检出切点判
  ``no_cut``，后两段不做。
- 取帧：每对四帧，前镜中帧 a1、前镜界前最后一帧 a2、后镜界后第 3 帧 b1（不超过后镜末帧）、后镜中帧 b2。
- 帧相似度：96×54 灰度像素相关，记 corr(a1, b2) 与 corr(a2, b1)；只用一个方向：两镜中帧相关 corr(a1, b2) ≥0.9 判同机位。
  相似度低不再直接判新机位，交判官比图（11-06 随机抽查 StoryMem 直判误判 7/27）。
- 每一项都先看两镜之间任一计划镜界 ±0.5 秒内是否检出切点，没检出判 ``no_cut``；计划不要求两镜不同、但有题比较的
  镜对同样要换到新机位。
- 判官：`gemini-3.8-flash`，Google 原生 Batch（OpenRouter 自 11-06 15:10 起 403），不传采样参数，一项一条请求。第 3 版题面：任务、定义与输出要求进 system instruction；
  user 消息是四个文字标签各接一张原分辨率 JPEG（两镜各自首末帧：镜界后第 2 帧、镜界前第 2 帧；ffmpeg ``-q:v 2``；
  图片中档，每张 560 token）和最后一句问题；结构化 JSON（``shot_a``、``shot_b``、``setup``），描述排在答案前。
  ``unclear`` 只留给全黑一类什么都看不到的帧，前提不成立。不发请求原文、计划景别、人名、方法名。思考档在 ``submit`` 显式给，
  同一运行根只用一种（记在 ``judge.json``）。费用按 Batch 价从 usage 估算。

取帧与相似度在主服务器 CPU 机上跑（成片留在主服务器）；送判官的帧在主服务器上写判分 Cloud Storage 桶，本机登记 ``gs://`` 后在请求里
引用（服务器出口过不了 Gemini 地区闸，判官从本机经代理出网），密钥从本机
文件读进内存、不落盘。

运行根：``shot-windows/<行>.json``（每部成片的对齐镜头起止秒数）、``similarity/<行>.json``（相关系数与帧号缓存）、
``images/<行>.json``（交判官的四帧在判分桶里的 ``gs://`` 地址）、``pairs/<行>.json``
（每项是否计划切点、是否检出切点、两个相关系数）、``batches/``（批任务账，规则见 ``gemini_native``）、
``receipts/<行>/<项>.json``（判官回执；失败回执保留、不自动重发）、``verdicts/<行>.json``（每项判定，
``checklist_scoring`` 读）。

用法::

    PYTHONPATH=src python -m mspc.evaluation.cut_execution prepare --root RUN_ROOT --media-root ROOT_WITH_ROWS \\
        --items cut-items.json --cuts CUTS.json [...] --rows ROW ... --service-account-file SA_KEY \\
        --bucket storyboard-judge-upload --proxy http://127.0.0.1:7890
    PYTHONPATH=src python -m mspc.evaluation.cut_execution submit --root RUN_ROOT --rows ROW ... --max-requests 2000 \\
        --cap-usd 30 --thinking low --key-file GOOGLE_KEY --proxy http://127.0.0.1:7890
    PYTHONPATH=src python -m mspc.evaluation.cut_execution collect --root RUN_ROOT --key-file GOOGLE_KEY \\
        --proxy http://127.0.0.1:7890
"""
from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from decimal import Decimal as D
from itertools import accumulate
from pathlib import Path

from mspc.evaluation import gemini_native as g

WINDOW_SECONDS = 0.5
MIDDLE_SAME_THRESHOLD = 0.9
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-p', '4974',
       'root@HOST']
REMOTE_PYTHON = '/root/miniforge/bin/python3'
REMOTE_FFMPEG = '${MSPC_SHARED_ROOT}/env/ffmpeg7/bin/ffmpeg'
# 输出上限 1,024：实测每对约 80 token（两句描述加答案，low 不思考）。一条 Batch 请求的预留按四张中档图（各 560 token）、
# 文字约 700 token 与输出上限算，Batch 价。
MAX_OUTPUT_TOKENS = 1024
RESERVE_PER_REQUEST = g.reserve(4 * 560 + 700, MAX_OUTPUT_TOKENS, 'batch')

# 第 3 版题面：任务、定义与输出要求进 system instruction；
# 四帧改为两镜各自首末帧；看不清只留给全黑一类帧。
SYSTEM = """You check whether a cut in a generated multi-shot video moves to a new camera setup.
You will see four frames: the first and last frame of Shot A, then the first and last frame of Shot B.
Shot B directly follows Shot A in the video, or is a later shot that is compared with Shot A.

A camera setup is where the camera stands, which way it points, and how wide or close it frames the place.
Judge it only by the large fixed structures of the place (walls, windows, doors, building fronts, the horizon):
where they sit in the frame and how large they appear.
- new: the camera clearly moved, turned or zoomed, so these structures appear in different places or at a clearly
  different size, or a different part of the place is visible, or it is clearly a different place.
- same: these structures sit at about the same places and sizes. People moving closer or further away, entering,
  leaving or being replaced by someone else, and changes of small objects, lighting or surface detail do not make a
  new setup. If the camera position, viewing angle, and background framing remain the same, and the only difference
  is that a character moves closer or shifts position within the same fixed frame, answer same. This remains same
  when the moving foreground character covers part of the fixed background. A slight pan or small reframing is the
  same setup.
- unclear: only when a frame shows nothing usable, for example it is completely black. If a person or object fills
  the frame, still decide new or same from what you can see. A character appearing larger is not by itself evidence
  of a new setup; use the fixed background framing to distinguish camera reframing from foreground movement.

Use only the four frames. First describe each shot in one sentence: where the camera is and where the large fixed
structures sit in the frame. Then answer."""
LABELS = (('a_first', 'Shot A, first frame'), ('a_last', 'Shot A, last frame'),
          ('b_first', 'Shot B, first frame'), ('b_last', 'Shot B, last frame'))
QUESTION = ('Based on the four frames above, describe Shot A and Shot B, then answer: '
            'was Shot B filmed from a new camera setup?')
PROPERTIES = {
    'shot_a': {'type': 'string', 'description': 'One sentence on Shot A: camera position and where the large fixed '
                                                'structures sit in the frame.'},
    'shot_b': {'type': 'string', 'description': 'The same for Shot B.'},
    'setup': {'type': 'string', 'enum': ['new', 'same', 'unclear'],
              'description': 'New camera setup judged only by the large fixed structures; people, small objects and '
                             'lighting do not count.'},
}
RESPONSE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'new_camera_setup', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False, 'required': list(PROPERTIES), 'properties': PROPERTIES}}}


def boundaries(shot_frames: list[int], fps: float, detected_seconds: list[float]) -> tuple[list[int], list[bool]]:
    """各镜界帧号（含 0 与总帧数）与每个内部镜界是否检出切点；检出的以最近检出切点为界。"""
    planned = [0, *accumulate(shot_frames)]
    detected = [round(t * fps) for t in detected_seconds]
    bounds, found = [0], []
    for b in planned[1:-1]:
        near = [t for t in detected if abs(t - b) <= WINDOW_SECONDS * fps]
        bounds.append(min(near, key=lambda t: abs(t - b)) if near else b)
        found.append(bool(near))
    return bounds + [planned[-1]], found


def pair_frames(bounds: list[int], earlier: int, later: int, total: int) -> list[int]:
    """a1、a2、b1、b2 四帧的帧号。"""
    a0, a1 = bounds[earlier - 1], bounds[earlier]
    b0, b1 = bounds[later - 1], bounds[later]
    picked = [(a0 + a1) // 2, max(a0, a1 - 1), min(b0 + 3, b1 - 1), (b0 + b1) // 2]
    return [min(max(f, 0), total - 1) for f in picked]


def shot_edges(bounds: list[int], shot: int, total: int) -> list[int]:
    """一镜的首帧与末帧：镜界后第 2 帧、镜界前第 2 帧（避开切点上的过渡帧）；镜太短时退到镜内。"""
    s0, s1 = bounds[shot - 1], bounds[shot]
    picked = [min(s0 + 2, s1 - 1), max(s1 - 3, s0)]
    return [min(max(f, 0), total - 1) for f in picked]


def shot_triplet_frames(bounds: list[int], shot: int, total: int) -> list[int]:
    """一镜的首、中、末帧；不足五帧时用镜内端点，避免避让切点的偏移让首末反序。"""
    s0, s1 = bounds[shot - 1], bounds[shot]
    first, last = shot_edges(bounds, shot, total) if s1 - s0 >= 5 else (s0, s1 - 1)
    middle = min(max((s0 + s1) // 2, 0), total - 1)
    return [min(max(f, 0), total - 1) for f in (first, middle, last)]


def jobs(media: dict, items: list[dict], detected_seconds: list[float]) -> tuple[list[dict], dict[str, bool]]:
    """一部成片的取帧任务（``frames`` 算相似度，``judge_frames`` 是送判官的两镜首末帧）与切点检出：两镜之间的各个
    计划镜界里有没有任何一个检出切点（相邻项就是它那一个镜界）。"""
    bounds, found = boundaries(media['shot_frames'], media['probe']['fps'], detected_seconds)
    out, cut = [], {}
    for item in items:
        cut[item['item_id']] = any(found[b - 1] for b in range(item['reference_shot'], item['shot']))
        total = media['probe']['frames']
        out.append({'item_id': item['item_id'],
                    'frames': pair_frames(bounds, item['reference_shot'], item['shot'], total),
                    'judge_frames': [*shot_edges(bounds, item['reference_shot'], total),
                                     *shot_edges(bounds, item['shot'], total)]})
    return out, cut


# 主服务器 CPU 机上执行：每部成片按需要的帧号解码一次。``mode`` 为 similarity 时按 ``frames`` 回相关系数，为 jpeg 时
# 按 ``judge_frames`` 回送判官的四帧原分辨率 JPEG。
REMOTE = r'''
import json, subprocess, sys, tempfile, numpy as np
from pathlib import Path
from multiprocessing import Pool
P = json.load(sys.stdin)
FF, W, H = P['ffmpeg'], 96, 54
def decode(video, need, vf):
    sel = '+'.join(f'eq(n\\,{f})' for f in need)
    return subprocess.run([FF, '-nostdin', '-v', 'error', '-i', video, '-vf', f'select={sel},{vf}', '-vsync', '0']
                          + P['out'], capture_output=True, timeout=900, check=True).stdout
def corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6); return float((a * b).mean())
def similarity(job):
    need = sorted({f for p in job['pairs'] for f in p['frames']})
    raw = np.frombuffer(decode(job['video'], need, f'scale={W}:{H},format=gray'), dtype=np.uint8)
    if raw.size != len(need) * W * H:
        raise ValueError(f"{job['video']}: decoded {raw.size // (W * H)} of {len(need)} frames")
    fr = {f: raw[i * W * H:(i + 1) * W * H].astype(float) for i, f in enumerate(need)}
    return [{'key': p['key'], 'corr_mid': corr(fr[p['frames'][0]], fr[p['frames'][3]]),
             'corr_cut': corr(fr[p['frames'][1]], fr[p['frames'][2]])} for p in job['pairs']]
def jpeg(job):
    need = sorted({f for p in job['pairs'] for f in p['judge_frames']})
    if not need:
        return []
    # 一次顺序解码取唯一帧，image2 保留原 q:v 2 的独立 JPEG 编码；写到本次的远端目录，随后由
    # gcs_host_upload 写判分桶并删目录（不再经 ssh 回传 base64）。
    out = Path(P['out_dir']) / job['story']
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='judge-frames-', dir=P['out_dir']) as tmp:
        sel = '+'.join(f'eq(n\\,{f})' for f in need)
        subprocess.run(
            [FF, '-nostdin', '-v', 'error', '-i', job['video'], '-vf', f'select={sel}',
             '-fps_mode', 'passthrough', '-frames:v', str(len(need)), '-q:v', '2',
             '-f', 'image2', '-c:v', 'mjpeg', str(Path(tmp) / '%08d.jpg')],
            capture_output=True, timeout=900, check=True)
        files = sorted(Path(tmp).glob('*.jpg'))
        if len(files) != len(need):
            raise ValueError(f"{job['video']}: decoded {len(files)} of {len(need)} judge frames")
        for f, path in zip(need, files, strict=True):
            path.rename(out / f'{f:06d}.jpg')
    return [{'key': p['key'], 'story': job['story'], 'frames': p['judge_frames'],
             'bytes': {str(f): (out / f'{f:06d}.jpg').stat().st_size for f in p['judge_frames']}} for p in job['pairs']]
if P['mode'] == 'similarity':
    P['out'] = ['-f', 'rawvideo', '-']
    work = similarity
else:
    work = jpeg
with Pool(P['workers']) as pool:
    print(json.dumps([r for rs in pool.imap_unordered(work, P['jobs'], chunksize=1) for r in rs]))
'''


def remote(mode: str, video_jobs: list[dict], workers: int, out_dir: str | None = None) -> list[dict]:
    payload = json.dumps({'mode': mode, 'jobs': video_jobs, 'ffmpeg': REMOTE_FFMPEG, 'workers': workers,
                          'out_dir': out_dir})
    script = f'import sys, io; sys.stdin = io.StringIO({payload!r})\n' + REMOTE
    out = subprocess.run(SSH + [REMOTE_PYTHON, '-'], input=script, capture_output=True, text=True, check=True,
                         timeout=14400).stdout
    return json.loads(out.strip().splitlines()[-1])


def frames_to_bucket(storage: g.Storage, judge: str, row: str, video_jobs: list[dict], workers: int = 16) -> dict:
    """送判官的帧：主服务器上解码成原 q:v 2 JPEG，写判分桶，
    返回镜对或切点键 → 按 ``judge_frames`` 顺序的 ``gs://`` 地址。同一部片里重复用到的帧只写一次。"""
    out_dir = f'/tmp/judge-frames-{judge}-{uuid.uuid4().hex}'
    results = remote('jpeg', video_jobs, workers, out_dir)
    sizes = {(r['story'], f): n for r in results for f, n in ((int(k), v) for k, v in r['bytes'].items())}
    items = [{'path': f'{out_dir}/{story}/{f:06d}.jpg', 'object': f'{judge}/{row}/{story}/{f:06d}.jpg',
              'mime': 'image/jpeg', 'size': n} for (story, f), n in sorted(sizes.items())]
    uris = dict(zip(sorted(sizes), storage.put_from_host(SSH, REMOTE_PYTHON, items, cleanup=out_dir, workers=workers)))
    return {r['key']: [uris[(r['story'], f)] for f in r['frames']] for r in results}


# 图片中档（每张 560 token，与像素多少无关），省约一半。显式写上，不靠默认值。
MEDIA_RESOLUTION = 'MEDIA_RESOLUTION_MEDIUM'
THINKING = ('low', 'medium', 'high')


def judge_body(images: list[dict], thinking: str) -> dict:
    """``images`` 是登记好的四帧 Files 资源（``gs://`` 登记，按 ``LABELS`` 顺序）。"""
    if thinking not in THINKING:
        raise ValueError(f'thinking must be one of {THINKING}')
    parts = []
    for (_, label), file in zip(LABELS, images, strict=True):
        parts += [{'text': label}, g.file_part(file)]
    parts.append({'text': QUESTION})
    return g.json_request(parts, RESPONSE_FORMAT['json_schema']['schema'], system=SYSTEM, thinking=thinking,
                          max_output_tokens=MAX_OUTPUT_TOKENS, media_resolution=MEDIA_RESOLUTION)


def to_judge(pair: dict) -> bool:
    """交判官比图的项：两镜之间检出了切点、两镜中帧不算几乎相同。
    相似度低不再直接判新机位（11-06 随机抽查：StoryMem 27 个这样判的前提里作者判同机位 7 个）。"""
    return pair['cut_between'] and pair['corr_mid'] < MIDDLE_SAME_THRESHOLD


def safe(key: str) -> str:
    return key.replace('/', '_')


def load(path: Path):
    return json.loads(Path(path).read_text())


def prepare(root: Path, media_root: Path, items_path: Path, cut_paths: list[Path], rows: list[str],
            storage: g.Storage) -> dict:
    """逐行：相似度（缓存在 ``similarity/``）→ 交判官的项取帧写判分桶（地址记在 ``images/<行>.json``）→
    ``pairs/<行>.json`` → 按已有回执写判定。"""
    items = load(items_path)['stories']
    cuts = {}
    for path in cut_paths:
        cuts.update(load(path)['rows'])
    summary = {}
    for row in rows:
        media = load(media_root / 'rows' / row / 'media-map.json')
        stories = [s['story_id'] for s in load(media_root / 'rows' / row / 'prepared.json')['stories']]
        video_jobs, detected, by_key, shot_windows = [], {}, {}, {}
        for sid in stories:
            m = media[sid]
            fps = m['probe']['fps']
            bounds, _ = boundaries(m['shot_frames'], fps, cuts[row][sid]['detected_cut_times_seconds'])
            shot_windows[sid] = [[s0 / fps, s1 / fps] for s0, s1 in zip(bounds, bounds[1:])]
            story_items = items[sid]['items']
            if not story_items:
                continue
            pair_jobs, found = jobs(media[sid], story_items, cuts[row][sid]['detected_cut_times_seconds'])
            detected.update(found)
            for item in story_items:
                by_key[item['item_id']] = item
            if pair_jobs:
                video_jobs.append({'video': media[sid]['remote_video_path'], 'story': sid,
                                   'pairs': [{'key': j['item_id'], 'frames': j['frames'], 'judge_frames': j['judge_frames']}
                                             for j in pair_jobs]})
        g.write_json(root / 'shot-windows' / f'{row}.json', shot_windows)
        sim_path = root / 'similarity' / f'{row}.json'
        sims = load(sim_path) if sim_path.exists() else {}
        todo = [{**job, 'pairs': [p for p in job['pairs'] if p['key'] not in sims]} for job in video_jobs]
        todo = [job for job in todo if job['pairs']]
        if todo:  # 只算缓存里还没有的项（切镜项规则改了、多出新项时）
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            frames = {p['key']: p['frames'] for job in todo for p in job['pairs']}
            sims.update({r['key']: {**r, 'frames': frames[r['key']]} for r in remote('similarity', todo, 48)})
            sim_path.write_text(json.dumps(dict(sorted(sims.items())), indent=1) + '\n')
        if not set(by_key) <= set(sims):
            raise ValueError(f'{row}: similarity misses {len(set(by_key) - set(sims))} items')
        pairs = {k: {'planned_cut': by_key[k]['planned_cut'], 'cut_between': detected[k],
                     'corr_mid': sims[k]['corr_mid'], 'corr_cut': sims[k]['corr_cut']} for k in sorted(by_key)}
        image_path = root / 'images' / f'{row}.json'
        images = load(image_path) if image_path.exists() else {}
        missing = {k for k, p in pairs.items() if to_judge(p) and k not in images}
        if missing:
            jpeg_jobs = [{**j, 'pairs': [p for p in j['pairs'] if p['key'] in missing]} for j in video_jobs]
            images.update(frames_to_bucket(storage, 'cut', row, [j for j in jpeg_jobs if j['pairs']]))
            g.write_json(image_path, images)
        (root / 'pairs').mkdir(parents=True, exist_ok=True)
        (root / 'pairs' / f'{row}.json').write_text(json.dumps(pairs, indent=1) + '\n')
        summary[row] = decide(root, row)
    return summary


REFUSALS = ('PROHIBITED_CONTENT', 'SAFETY', 'BLOCKLIST', 'IMAGE_SAFETY')


def refused(record: dict) -> bool:
    """判官被安全过滤拒判（11-06 实测：两行 2,074 条里 1 条 PROHIBITED_CONTENT）。重发同一请求仍会被拒。"""
    raw = record.get('raw') or {}
    candidates = raw.get('candidates') or [{}]
    prompt_feedback = raw.get('promptFeedback') or {}
    return (record.get('status') == 'format_failed'
            and (candidates[0].get('finishReason') in REFUSALS or prompt_feedback.get('blockReason') in REFUSALS))


def decide(root: Path, row: str) -> dict:
    """由 ``pairs`` 与判官回执写 ``verdicts/<行>.json``；还没有有效回执的判官项判定记 None。"""
    verdicts = {}
    for k, pair in load(root / 'pairs' / f'{row}.json').items():
        sim = max(pair['corr_mid'], pair['corr_cut'])
        if not pair['cut_between']:
            verdicts[k] = {'verdict': 'no_cut', 'decided_by': 'transnet', 'similarity': sim}
        elif pair['corr_mid'] >= MIDDLE_SAME_THRESHOLD:
            verdicts[k] = {'verdict': 'same', 'decided_by': 'middle_frames', 'similarity': sim}
        else:
            path = root / 'receipts' / row / f'{safe(k)}.json'
            record = load(path) if path.exists() else {}
            setup = record.get('answer', {}).get('setup') if record.get('status') == 'answered' else None
            if refused(record):  # 安全过滤拒判：与看不清同样处理，前提不成立，另记 refused
                setup = 'refused'
            verdicts[k] = {'verdict': 'unclear' if setup == 'refused' else setup,  # new／same／unclear；unclear 前提不成立
                           'decided_by': 'judge', 'similarity': sim, 'judge_setup': setup}
    (root / 'verdicts').mkdir(exist_ok=True)
    (root / 'verdicts' / f'{row}.json').write_text(json.dumps(verdicts, indent=1) + '\n')
    return {'items': len(verdicts), 'undecided': sum(v['verdict'] is None for v in verdicts.values()),
            'by': {d: sum(v['decided_by'] == d for v in verdicts.values()) for d in {v['decided_by'] for v in verdicts.values()}}}


def pending(root: Path, rows: list[str]) -> list[tuple[str, str]]:
    """交判官且从未记入批任务、没有回执的项；失败和缺失不自动重发。"""
    submitted = {key for batch in g.batch_records(root) for key in batch['keys']}
    out = []
    for row in rows:
        for k, pair in load(root / 'pairs' / f'{row}.json').items():
            path = root / 'receipts' / row / f'{safe(k)}.json'
            if not to_judge(pair):
                continue
            if path.exists():
                status = load(path)['status']
                if status not in ('answered', 'format_failed', 'request_failed'):
                    raise ValueError(f'{path}: unknown receipt status {status}')
                continue
            if f'{row}|{k}' in submitted:
                continue
            out.append((row, k))
    return out


def submit(root: Path, rows: list[str], max_requests: int, cap: D, thinking: str, client: g.Gemini) -> dict | None:
    settings = {'model': g.MODEL, 'thinking': thinking, 'media_resolution': MEDIA_RESOLUTION, 'question_version': 4}
    if (root / 'judge.json').exists():
        if load(root / 'judge.json') != settings:
            raise ValueError(f"{root}: judge settings {load(root / 'judge.json')} differ from {settings}")
    else:
        g.write_json(root / 'judge.json', settings)
    todo = pending(root, rows)[:max_requests]
    if not todo:
        return None
    keys = [f'{row}|{k}' for row, k in todo]
    path = g.open_batch(root, 'setup', keys, RESERVE_PER_REQUEST * len(todo), cap, 10_000 * len(todo))
    images = {row: load(root / 'images' / f'{row}.json') for row in {r for r, _ in todo}}
    registered = client.register_all([u for row, k in todo for u in images[row][k]])
    requests = [(key, judge_body([registered[u] for u in images[row][k]], thinking)) for key, (row, k) in zip(keys, todo)]
    return g.start_batch(client, path, requests, {u: f['name'] for u, f in registered.items()})


def judge_record(line: dict) -> dict:
    """一条 Batch 结果变成回执：``answered``／``format_failed``／``request_failed``。"""
    if 'error' in line:
        return {'status': 'request_failed', 'error': line['error'], 'cost_usd': None}
    raw = line['response']
    usage = raw.get('usageMetadata') or {}
    cost = str(g.cost(usage, 'batch')) if 'promptTokenCount' in usage else None
    try:
        answer = g.answer_json(raw)
        valid = set(answer) == set(PROPERTIES) and answer['setup'] in ('new', 'same', 'unclear')
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        return {'status': 'format_failed', 'raw': raw, 'usage': usage, 'cost_usd': cost}
    return {'status': 'answered', 'answer': answer, 'api': 'batchGenerateContent',
            'model': raw.get('modelVersion', g.MODEL), 'usage': usage, 'cost_usd': cost, 'raw': raw}


def collect(root: Path, client: g.Gemini) -> dict:
    touched = set()

    def handle(record: dict, lines: list[dict]) -> tuple[D, dict]:
        cost, written = D(0), {}
        for line in lines:
            row, k = line['key'].split('|', 1)
            pair = load(root / 'pairs' / f'{row}.json')[k]
            data = {'item_id': k, 'row': row, 'pair': pair, 'batch': record['name'], 'finished_at': g.now(),
                    **judge_record(line)}
            path = root / 'receipts' / row / f'{safe(k)}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
            written[data['status']] = written.get(data['status'], 0) + 1
            cost += D(data['cost_usd'] or 0)
            touched.add(row)
        return cost, written
    report = g.collect_batches(client, root, handle)
    return {'batches': report, 'verdicts': {row: decide(root, row) for row in sorted(touched)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True, help='切镜执行运行根')
    p.add_argument('--media-root', type=Path, required=True, help='含 rows/<行>/media-map.json 与 prepared.json 的判分运行根')
    p.add_argument('--items', type=Path, required=True, help='question_bank_v6 写出的切镜执行项')
    p.add_argument('--cuts', type=Path, nargs='+', required=True, help='checklist_scoring stored-cuts／detect-cuts 的切点表')
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--service-account-file', type=Path, required=True, help='判分桶上传服务账号 JSON 密钥（只在本机）')
    p.add_argument('--bucket', required=True)
    p.add_argument('--proxy', required=True, help='本机 HTTP 代理（签令牌用）')
    for name in ('submit', 'collect'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, required=True)
        sub.add_argument('--key-file', type=Path, required=True, help='Google API 密钥文件（只读进内存）')
        sub.add_argument('--proxy', required=True, help='本机 HTTP 代理')
        if name == 'submit':
            sub.add_argument('--rows', nargs='+', required=True)
            sub.add_argument('--max-requests', type=int, required=True)
            sub.add_argument('--cap-usd', type=D, required=True, help='本运行根判官费用硬顶')
            sub.add_argument('--thinking', choices=THINKING, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        storage = g.Storage(args.service_account_file, args.bucket, args.proxy)
        print(json.dumps(prepare(args.root, args.media_root, args.items, args.cuts, args.rows, storage), indent=1))
        return
    client = g.Gemini(g.read_key(args.key_file), args.proxy)
    if args.command == 'submit':
        record = submit(args.root, args.rows, args.max_requests, args.cap_usd, args.thinking, client)
        print(json.dumps({k: record[k] for k in ('name', 'state', 'reserve_usd')} | {'requests': len(record['keys'])}
                         if record else {'submitted': 0}))
    else:
        print(json.dumps(collect(args.root, client), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
