"""整片判官（事件与状态两列）走 Google 原生 Batch（OpenRouter 自 11-06 15:10 起 403）。

当前判官合同：system instruction 讲任务、作答规则与输出要求（``SYSTEM``）；user 消息是视频（Files
API 传原片，FPS 在 ``config.json`` 写明，现行 2 FPS）在前，登记表、该视频各镜时间段、编号问题
与一句总问（``CLOSING``）在后，不发送逐镜剧本；逐题字段顺序与 thinking 档在 prepare 时显式给
（现行先写时间与初态→末态／过程证据再答）；12,288 输出上限、视频中档。只送事件与状态题，前提不成立的状态复查题不送。回执与旧根同形
（``rows/<行>/output/metrics/first-read/...``，``status: rated`` 与 ``result.answers``）。

运行根：
- ``config.json``：冻结的判官设置、题库路径、FPS；``rows/<行>/prepared.json``：题换成给定题库的题，时间轴与请求沿用
  源判分根；``rows/<行>/media-map.json``：成片在主服务器的路径与字节数。
- ``batches/<名>.json``：每个批任务一份（账目规则见 ``gemini_native`` 的批任务账）。
- 成片在主服务器主机上核字节后写判分 Cloud Storage 桶、本机登记 ``gs://``（不经本机中转）；批任务结束后撤销登记、删掉输入
  JSONL，桶里对象由桶的生命周期规则 2 天后删除。

用法（本机；密钥文件只读进内存）::

    python -m mspc.evaluation.gemini_checklist_batch prepare --source-root V6_ROOT --bank questions.v11.json \\
        --cut-items cut-items.v11.json --cut-root CUT_ROOT --rows ROW ... --fps 2 --thinking low \\
        --answer-order reason_first --output ROOT
    python -m mspc.evaluation.gemini_checklist_batch submit --root ROOT --rows ROW ... --max-requests 300 \\
        --cap-usd 20 --workers 8 --service-account-file SA_KEY --bucket storyboard-judge-upload \\
        --key-file KEY --proxy http://127.0.0.1:7890
    python -m mspc.evaluation.gemini_checklist_batch collect --root ROOT --key-file KEY --proxy http://127.0.0.1:7890
    python -m mspc.evaluation.gemini_checklist_batch status --root ROOT
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal as D
from itertools import accumulate
from pathlib import Path

from mspc.evaluation import gemini_native as g
from mspc.evaluation.cut_execution import boundaries
from mspc.evaluation.story_checklist import ANSWER, request_path, validate_response, write

# 成片所在的主服务器主机：成片在那里经项目代理写判分桶（``submit``）。
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-p', '4974',
       'root@HOST']
REMOTE_PYTHON = '/root/miniforge/bin/python3'
# 逐题字段顺序：answer_first 是旧通路沿用的 question → answer → reason → time；reason_first 改成
# question → time → reason → answer（11-06 配置复查：thinking low 实测 0 token，先答后写理由时理由帮不了判断）。
ORDERS = {'answer_first': ('question', 'answer', 'reason', 'time'), 'reason_first': ('question', 'time', 'reason', 'answer')}
THINKING = ('low', 'medium', 'high')
# 输出上限 12,288：实测 low 约 1,100、medium 约 1,100 正文加 4,000 思考；费用预留按这个上限算，硬顶覆盖单条最坏情况。
JUDGE = {'model': g.MODEL, 'max_output_tokens': 12288, 'media_resolution': 'MEDIA_RESOLUTION_MEDIUM'}
# 整片判官只判事件与状态（人物、物体、地点改由镜对图片判官判）；先讲清任务再问；任务与规则进 system instruction，视频在前、文字在后、问题放最后。
DIMENSIONS = ('event', 'state')
SYSTEM = """# Task
You check visible events and states in a generated multi-shot video. You will receive a registry identifying
characters, objects and locations, each numbered shot's time range, and a numbered list of yes/no questions.
Each question asks about one thing that should be visible in its given time range. The registry identifies
entities; it does not establish that any requested action or state appears in the video.

# How to answer each question
- yes: you can clearly see it in the video within the given time range.
- no: the required action or state is absent, or the visible evidence is insufficient to establish it.
- For an event asking for an action or transition, observe the process within the time range; an end result alone
  is insufficient. If the person or object already has the target pose or state at the start and remains static
  throughout, with no required transition, answer no. A state question can be yes when that state is visibly
  present at the requested time without a transition.
- The registry only tells you which character, object or location a question is about. Find it by the role the
  question gives it in this time range (who performs the action, which object is used, where it happens). Do not
  answer no because the person, object or place looks different from the registry description or from other shots;
  identity across shots is checked separately. Answer no for a missing entity only when nothing in the time range
  plays that role. A character shown from behind or in profile still counts.
- An action near the frame edge counts as yes when the action trajectory and interaction are clearly visible,
  even if part of an object is cropped (for example pouring into a container partly below the bottom edge).
  If the required interaction itself cannot be seen, answer no.
- Judge every question independently and use only evidence inside its specified time range. Treat ranges as
  start-inclusive and end-exclusive: the first frame of the next shot is not evidence for the preceding shot.
- For both yes and no, describe the observed initial and final states and the observed transition or state
  features. State when something is not visible; never invent a trajectory from an expected outcome.

# Output
For each question, give the time (m:ss.d) where you looked and a brief reason in this order:
[Initial -> Final] the observed state at the start and at the end of the time range;
[Transition] the specific observed action trajectory or state features, including any missing transition.
Then answer yes or no."""
CLOSING = 'Based on the video, registry and time ranges above, answer every question.'


def schema(order: str) -> dict:
    fields = ORDERS[order]
    item = {**ANSWER, 'required': list(fields), 'properties': {f: ANSWER['properties'][f] for f in fields}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['answers'],
            'properties': {'answers': {'type': 'array', 'items': item}}}
# 每请求的费用预留（Batch 价）：视频中档按每采样帧 258 token，文字题面按 4,000 token 预留；
# 输出按上限预留。只用于硬顶检查，实际按回执 usage 记。
FRAME_TOKENS, TEXT_TOKENS, OUTPUT_RESERVE = 258, 4000, JUDGE['max_output_tokens']


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path):
    return json.loads(Path(path).read_text())


def _shot_bounds(shot_seconds: list[float], cut_bounds: list[float] | None = None) -> list[float]:
    """秒数边界；实际切点可替换计划累计镜长，保留原始精度到请求格式化时。"""
    edges = list(cut_bounds) if cut_bounds is not None else [0.0, *accumulate(shot_seconds)]
    if (not shot_seconds or len(edges) != len(shot_seconds) + 1 or edges[0] != 0
            or not all(math.isfinite(t) for t in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError('shot boundaries must start at zero and strictly increase, with one range per shot')
    return edges


def _window_bounds(windows: list[list[float]]) -> list[float]:
    """prepared／切镜根保存的连续时间窗转换为秒数边界。"""
    if not windows or any(len(w) != 2 for w in windows) or any(a[1] != b[0] for a, b in zip(windows, windows[1:])):
        raise ValueError('shot_windows must contain contiguous start/end pairs')
    return [windows[0][0], *(w[1] for w in windows)]


def _clock(seconds: float) -> str:
    tenths = int(round(seconds * 10))
    return f'{tenths // 600}:{tenths // 10 % 60:02d}.{tenths % 10}'


def video_judge_questions_text(request: str, shot_seconds: list[float], rows: list[dict],
                               cut_bounds: list[float] | None = None) -> str:
    """整片判官专用：只给 SUBJECTS／LOCATIONS 登记表、0.1 秒时间窗和题面，不给逐镜剧本。"""
    registry, headings, active = [], set(), False
    for line in request.splitlines():
        text = line.strip()
        if text in ('SUBJECTS', 'LOCATIONS'):
            headings.add(text)
            registry.append(text)
            active = True
        elif active and text.startswith('- '):
            registry.append(line)
        elif text:
            active = False
    if headings != {'SUBJECTS', 'LOCATIONS'}:
        raise ValueError('request must contain SUBJECTS and LOCATIONS registry blocks')
    edges = _shot_bounds(shot_seconds, cut_bounds)
    windows = {k: f'{_clock(a)}–{_clock(b)}' for k, (a, b) in enumerate(zip(edges, edges[1:]), 1)}
    lines = ['## Registry', *registry, '', '## Shots in this video',
             ' · '.join(f'Shot {k}: {w}' for k, w in windows.items()), '', '## Questions']
    for number, row in enumerate(rows, 1):
        text = row['question']
        for k, window in windows.items():
            text = text.replace('{shot %d}' % k, window)
        if '{shot' in text:
            raise ValueError(f'unresolved shot placeholder: {row["requirement_id"]}')
        prefix = f"({'end of ' if row['anchor'] == 'end' else ''}{windows[row['shot']]}) "
        lines.append(f'{number}. {prefix}{text}')
    return '\n'.join(lines)


def premise_failed(items: list[dict], verdicts: dict) -> set[str]:
    """前提不成立的题：切镜项判定不是 ``new``（两镜之间切开且换了新机位）时，它的 ``dependents``。"""
    undecided = [i['item_id'] for i in items if verdicts[i['item_id']]['verdict'] is None]
    if undecided:
        raise ValueError(f'undecided cut items: {undecided[:3]}')
    return {d for i in items if verdicts[i['item_id']]['verdict'] != 'new' for d in i['dependents']}


def prepare(source_root: Path, bank_path: Path, cut_items_path: Path, cut_root: Path, rows: list[str], fps: float,
            thinking: str, order: str, output: Path, cut_paths: list[Path] | None = None) -> dict:
    """运行根：判官设置冻结进 ``config.json``；各行的题换成 ``bank`` 里该故事的事件与状态题，其余字段照源根。

    实际时间窗优先取显式 cuts，其次取切镜根已保存的 shot-windows；未获得检测时间窗时沿用源时间轴。
    切镜执行要先判完：前提不成立的状态复查题不送判官，记在
    ``skipped_premise_failed``，计分时记不过；人物、物体、地点题归镜对判官。"""
    g.video_part({'uri': 'x', 'mimeType': 'video/mp4'}, fps)  # 校验 FPS 范围
    if thinking not in THINKING:
        raise ValueError(f'thinking must be one of {THINKING}')
    bank, items = load(bank_path), load(cut_items_path)['stories']
    cuts = {}
    for path in cut_paths or []:
        cuts.update(load(path)['rows'])
    config = {'judge': {**JUDGE, 'thinking': thinking, 'answer_order': order, 'fps': fps, 'schema': schema(order),
                        'system': SYSTEM, 'closing': CLOSING, 'prompt_version': 5},
              'question_bank': str(bank_path), 'question_bank_version': bank['version'],
              'cut_items': str(cut_items_path), 'cut_root': str(cut_root), 'cuts': [str(p) for p in cut_paths or []],
              'source_root': str(source_root),
              'rows': rows, 'prices_usd_per_million': {k: [str(x) for x in v] for k, v in g.PRICES.items()},
              'prices_valid_through': g.PRICES_VALID_THROUGH, 'created_at': now()}
    counts = {}
    if (output / 'config.json').exists():  # 往已有运行根加行：判官设置、题库、切镜项必须相同，已有的行不重写
        earlier = load(output / 'config.json')
        same = ('judge', 'question_bank', 'cut_items', 'cut_root', 'cuts', 'source_root')
        if any(earlier[k] != config[k] for k in same):
            raise ValueError(f'{output}: judge settings or inputs differ from the existing root')
        if set(rows) & set(earlier['rows']):
            raise ValueError(f'{output}: rows already prepared: {sorted(set(rows) & set(earlier["rows"]))}')
        counts, config = dict(earlier['counts']), {**earlier, 'rows': earlier['rows'] + rows}
    for row in rows:
        prepared = load(source_root / 'rows' / row / 'prepared.json')
        media = load(source_root / 'rows' / row / 'media-map.json')
        aligned_path = cut_root / 'shot-windows' / f'{row}.json'
        aligned = load(aligned_path) if aligned_path.exists() else {}
        verdicts = load(cut_root / 'verdicts' / f'{row}.json')
        stories, skipped = [], 0
        for story in prepared['stories']:
            sid = story['story_id']
            if cut_paths:
                m = media[sid]
                bounds, _ = boundaries(m['shot_frames'], m['probe']['fps'], cuts[row][sid]['detected_cut_times_seconds'])
                edges = _shot_bounds(story['shot_seconds'], [b / m['probe']['fps'] for b in bounds])
            elif sid in aligned:
                edges = _shot_bounds(story['shot_seconds'], _window_bounds(aligned[sid]))
            else:
                windows = story.get('shot_windows')
                edges = _shot_bounds(story['shot_seconds'], _window_bounds(windows) if windows is not None else None)
            failed = premise_failed(items[story['story_id']]['items'], verdicts)
            ours = [q for q in bank['stories'][story['story_id']]['requirements'] if q['dimension'] in DIMENSIONS]
            kept = [q for q in ours if q['requirement_id'] not in failed]
            skipped += len(ours) - len(kept)
            stories.append({**story, 'requirements': kept, 'shot_windows': [list(w) for w in zip(edges, edges[1:])],
                            'shot_seconds': [b - a for a, b in zip(edges, edges[1:])],
                            'skipped_premise_failed': [q['requirement_id'] for q in ours if q['requirement_id'] in failed]})
        out = {**prepared, 'stories': stories, 'rereads': [],
               'source': {**prepared['source'], 'questions': str(bank_path)},
               'judge': {'max_questions_per_request': max(len(s['requirements']) for s in stories), **config['judge']}}
        write(output / 'rows' / row / 'prepared.json', out)
        write(output / 'rows' / row / 'media-map.json', media)
        counts[row] = {'stories': len(stories), 'questions': sum(len(s['requirements']) for s in stories),
                       'skipped_premise_failed': skipped, 'unavailable': len(prepared['unavailable_stories'])}
    write(output / 'config.json', {**{k: v for k, v in config.items() if k not in ('counts', 'created_at')},
                                   'created_at': config['created_at'], 'counts': counts})
    return counts


def receipt_path(root: Path, row: str, story: dict) -> Path:
    return request_path(root / 'rows' / row / 'output', 'first-read', story)


def pending(root: Path, rows: list[str]) -> list[tuple[str, dict]]:
    """只提交有题、从未记入批任务且没有回执的故事；失败或缺失回执保留作测量缺失。"""
    submitted = {key for batch in g.batch_records(root) for key in batch['keys']}
    out = []
    for row in rows:
        for story in load(root / 'rows' / row / 'prepared.json')['stories']:
            path = receipt_path(root, row, story)
            if path.exists():
                status = load(path)['status']
                if status not in ('rated', 'format_failed', 'request_failed'):
                    raise ValueError(f'{path}: unknown receipt status {status}')
                continue
            if f"{row}|{story['story_id']}" in submitted:
                continue
            if not story['requirements']:
                continue  # 无事件／状态题：没有整片判官要测的内容，不制造空题请求。
            out.append((row, story))
    return out


def request_for(story: dict, file: dict, judge: dict) -> dict:
    windows = story.get('shot_windows')
    text = video_judge_questions_text(story['request'], story['shot_seconds'], story['requirements'],
                                      _window_bounds(windows) if windows is not None else None) + '\n\n' + judge['closing']
    return g.json_request([g.video_part(file, judge['fps']), {'text': text}], judge['schema'], system=judge['system'],
                          thinking=judge['thinking'], max_output_tokens=judge['max_output_tokens'],
                          media_resolution=judge['media_resolution'])


def story_reserve(story: dict, fps: float) -> D:
    return g.reserve(int(story['duration_seconds'] * fps + 1) * FRAME_TOKENS + TEXT_TOKENS, OUTPUT_RESERVE, 'batch')


def submit(root: Path, rows: list[str], max_requests: int, cap: D, client: g.Gemini, storage: g.Storage,
           workers: int) -> dict | None:
    """取一批待判故事：查硬顶、成片在主服务器主机上经项目代理写判分桶并本机登记（服务器出口过不了 Gemini 地区闸，但能写桶），写 JSONL、建一个批任务。登记的对象不占 Files
    20 GB 存储额，只有输入 JSONL 计入。"""
    judge = load(root / 'config.json')['judge']
    if judge['prompt_version'] != 5:
        raise ValueError(f'{root}: prepare a new root for the role-based event and state judge (prompt_version=5)')
    todo = pending(root, rows)[:max_requests]
    if not todo:
        return None
    keys = [f"{r}|{s['story_id']}" for r, s in todo]
    path = g.open_batch(root, 'checklist', keys, sum((story_reserve(s, judge['fps']) for _, s in todo), D(0)), cap,
                        50_000 * len(todo))  # 每条 JSONL 请求约 10 KB，按 50 KB 留
    media = [load(root / 'rows' / r / 'media-map.json')[s['story_id']] for r, s in todo]
    uris = storage.put_from_host(SSH, REMOTE_PYTHON, [
        {'path': m['remote_video_path'], 'object': f"video/{r}/{s['story_id']}.mp4", 'mime': 'video/mp4',
         'size': m['video_bytes']} for (r, s), m in zip(todo, media)], workers=workers)
    registered = client.register_all(uris)
    with ThreadPoolExecutor(workers) as pool:
        files = list(pool.map(lambda uri: client.wait_active(registered[uri]['name']), uris))
    requests = [(key, request_for(story, file, judge)) for key, (_, story), file in zip(keys, todo, files)]
    return g.start_batch(client, path, requests, {key: f['name'] for key, f in zip(keys, files)})

def receipt(row: str, story: dict, key: str, batch: dict, line: dict, judge: dict) -> dict:
    base = {'method': story['method'], 'input_tier': story['input_tier'], 'seed': story['seed'],
            'story_id': story['story_id'], 'part': None,
            'requirement_ids': [q['requirement_id'] for q in story['requirements']], 'phase': 'first-read',
            'model': judge['model'], 'fps': judge['fps'], 'api': 'batchGenerateContent', 'batch': batch['name'],
            'key': key, 'finished_at': now()}
    if 'error' in line:
        return {**base, 'status': 'request_failed', 'error': line['error'], 'cost_usd': None}
    response = line['response']
    usage = response.get('usageMetadata', {})
    base.update(usage=usage, raw_response=response,
                cost_usd=str(g.cost(usage, 'batch')) if 'promptTokenCount' in usage else None)
    try:
        result = validate_response(g.answer_json(response), story)
        if g.modality_tokens(usage, 'VIDEO') <= 0:
            raise ValueError('no video tokens in usage')
    except (AssertionError, KeyError, ValueError, json.JSONDecodeError) as error:
        return {**base, 'status': 'format_failed', 'error': f'{type(error).__name__}: {error}'}
    return {**base, 'status': 'rated', 'result': result}


def collect(root: Path, client: g.Gemini) -> list[dict]:
    """结束的批任务取回结果、写回执，再删掉上传的视频与 JSONL。"""
    config = load(root / 'config.json')
    stories = {f"{row}|{story['story_id']}": (row, story)
               for row in config['rows'] for story in load(root / 'rows' / row / 'prepared.json')['stories']}

    def handle(record: dict, lines: list[dict]) -> tuple[D, dict]:
        cost, written = D(0), {}
        for line in lines:
            row, story = stories[line['key']]
            data = receipt(row, story, line['key'], record, line, config['judge'])
            write(receipt_path(root, row, story), data)
            written[data['status']] = written.get(data['status'], 0) + 1
            cost += D(data['cost_usd'] or 0)
        return cost, written
    return g.collect_batches(client, root, handle)


def status(root: Path) -> dict:
    config = load(root / 'config.json')
    receipts = [load(p) for p in (root / 'rows').glob('*/output/metrics/first-read/**/*.json')]
    return {'rows': len(config['rows']), 'pending': len(pending(root, config['rows'])),
            'receipts': {s: sum(r['status'] == s for r in receipts) for s in {r['status'] for r in receipts}},
            'spent_usd_estimate': str(g.batch_spent(root)),
            'open_batches': [b.get('name', b['display_name']) for b in g.batch_records(root) if not b.get('closed')],
            'open_reserve_usd': str(g.batch_open_reserve(root))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source-root', type=Path, required=True, help='有各行 prepared.json 与 media-map.json 的判分根')
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--cut-items', type=Path, required=True, help='第 11 版切镜项（question_bank_v6 refresh-cut-items）')
    p.add_argument('--cut-root', type=Path, required=True, help='cut_execution 运行根（verdicts/<行>.json 已判完）')
    p.add_argument('--cuts', nargs='+', type=Path, help='TransNetV2 切点；省略时读取 cut-root 保存的 shot-windows')
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--fps', type=float, required=True)
    p.add_argument('--thinking', choices=THINKING, required=True)
    p.add_argument('--answer-order', choices=tuple(ORDERS), required=True)
    p.add_argument('--output', type=Path, required=True)
    for name in ('submit', 'collect'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, required=True)
        sub.add_argument('--key-file', type=Path, required=True)
        sub.add_argument('--proxy', required=True)
        if name == 'submit':
            sub.add_argument('--rows', nargs='+', required=True)
            sub.add_argument('--max-requests', type=int, required=True)
            sub.add_argument('--cap-usd', type=D, required=True)
            sub.add_argument('--workers', type=int, required=True, help='并发写桶与登记数')
            sub.add_argument('--service-account-file', type=Path, required=True, help='判分桶上传服务账号 JSON 密钥（只在本机）')
            sub.add_argument('--bucket', required=True)
    s = commands.add_parser('status')
    s.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.source_root, args.bank, args.cut_items, args.cut_root, args.rows, args.fps,
                                 args.thinking, args.answer_order, args.output, args.cuts), ensure_ascii=False))
    elif args.command == 'status':
        print(json.dumps(status(args.root), ensure_ascii=False, indent=1))
    else:
        client = g.Gemini(g.read_key(args.key_file), args.proxy)
        if args.command == 'submit':
            storage = g.Storage(args.service_account_file, args.bucket, args.proxy)
            record = submit(args.root, args.rows, args.max_requests, args.cap_usd, client, storage, args.workers)
            print(json.dumps(record and {k: record[k] for k in ('name', 'state', 'reserve_usd')} | {'requests': len(record['keys'])}
                             if record else {'submitted': 0}, ensure_ascii=False))
        else:
            print(json.dumps(collect(args.root, client), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
