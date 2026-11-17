"""人物、物体、地点一致性的镜对图片判官（第 11 版起），走 Google 原生 Batch。

人物、物体、地点一致性不看动态，所以不整片判，改送每镜首帧与末帧的原分辨率图片；图片比视频看得到更多细节。

- 判哪些：给定题库里人物、物体、地点题，按比较的镜对分组，一对镜头一次请求，问这对镜头上的全部这类题；前提不成立
  （``cut_execution`` 判定不是新机位或没检出切点）的题不送，计分时记不过。
- 送什么：system instruction 讲任务与规则；user 消息依次为登记表（本对题目点到的人、物、地点在请求里的描述）、六张带
  标签的图（前镜首、中、末帧，后镜首、中、末帧；首末取镜界后、前第 2 帧，中帧取镜界中点，镜界按 TransNetV2 检出切点调整）、
  编号题目与一句总问。图片原分辨率（主服务器上解码成 JPEG 写判分 Cloud Storage 桶、本机登记 ``gs://`` 后引用）、中档（每张 560 token）；思考档在 ``submit`` 显式给，一个运行根一种。
- 输出：人物、物体、地点三组，每题先按 EntityBench 的判据逐项填 same／different／not visible，再答 yes／no；题号须与
  给定的相同。

运行根：``pairs/<行>.json``（每个镜对的题号与帧号）、``skipped/<行>.json``（前提不成立、不送的题）、``images/<行>.json``（六帧的
``gs://`` 地址）、
``batches/``（批任务账）、``receipts/<行>/<镜对>.json``（逐题 yes／no 与原答）。

用法（本机；取帧在主服务器 CPU 上）::

    PYTHONPATH=src python -m mspc.evaluation.consistency_pair_batch prepare --root ROOT --media-root V6_ROOT \\
        --bank questions.v11.json --cut-items cut-items.v11.json --cut-root CUT_ROOT --cuts CUTS.json [...] --rows ROW ... \\
        --service-account-file SA_KEY --bucket storyboard-judge-upload --proxy http://127.0.0.1:7890
    PYTHONPATH=src python -m mspc.evaluation.consistency_pair_batch submit --root ROOT --rows ROW ... \\
        --max-requests 2000 --cap-usd 10 --thinking low --key-file KEY --proxy http://127.0.0.1:7890
    PYTHONPATH=src python -m mspc.evaluation.consistency_pair_batch collect --root ROOT --key-file KEY --proxy ...
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal as D
from pathlib import Path

from mspc.evaluation import gemini_native as g
from mspc.evaluation.cut_execution import boundaries, frames_to_bucket, refused, shot_triplet_frames
from mspc.evaluation.gemini_checklist_batch import premise_failed

DIMENSIONS = ('character', 'object', 'place')
MEDIA_RESOLUTION = 'MEDIA_RESOLUTION_MEDIUM'
THINKING = ('low', 'medium', 'high')
# 输出上限 2,048：每题逐项判据加答案约 40 token，一对镜头通常不到 10 道题。预留按六张图各 560 token、文字约 800 token
# 与输出上限算，Batch 价。
MAX_OUTPUT_TOKENS = 2048
RESERVE_PER_REQUEST = g.reserve(6 * 560 + 800, MAX_OUTPUT_TOKENS, 'batch')
POSITIONS = ('first', 'middle', 'last')
SCHEMA_REVISION = 'per_group_question_numbers_v2'
_ENTITY = re.compile(r'(?:Character|Object|Location) \d+')

SYSTEM = """# Task
You check a generated multi-shot video for consistency between two of its shots.
You will see the first, middle and last frame of each shot, and a numbered list of questions.
Each question asks whether one character, object or location
in the later shot is the same individual as in the earlier shot.

# Rules
- Use only the frames. Each frame is labeled with its shot and position before the image.
- Compare the individuals in the two shots with each other. The registry description only tells you which
  character, object or location a question is about; do not judge how well a shot matches the description.
- Same character: the same person, judged by the face, hairstyle and build. Matching clothing, accessories or role
  alone is not evidence of the same person; a different face in the same outfit is a different person. Clothing may
  change between shots and does not by itself make two people different. If the face cannot be seen in one of the
  shots, the same hairstyle, build and clothing together count as the same person.
- Same object: the same specific item, judged by its shape, color, texture and distinctive details; another object
  of the same kind is not the same object.
- Same location: the same specific place, judged by fixed background features visible in both shots, such as walls,
  windows, furniture layout or landmarks. A different viewing angle of the same place is still the same place. The
  same type of place is not enough.
- A change of rendering style between live action, 2D animation and 3D animation means a different individual.
- If the character, object or location cannot be found in either shot, or the frames do not show enough to tell
  (for example it is out of frame, hidden, or the frame is filled by something else), the answer is no.

# Output
For each question, first compare the listed features (same, different, or not visible), then answer yes or no.
Use the question numbers exactly as given."""
CLOSING = 'Based on the frames above, answer each question in the group for its type.'

_FEATURE = {'type': 'string', 'enum': ['same', 'different', 'not visible']}


def _group(name: str, features: dict[str, str], verdict: str, verdict_description: str) -> dict:
    props = {'id': {'type': 'integer', 'description': 'The question number exactly as given.'}}
    props.update({f: {**_FEATURE, 'description': d} for f, d in features.items()})
    props[verdict] = {'type': 'string', 'enum': ['yes', 'no'], 'description': verdict_description}
    return {'type': 'array', 'description': f'One entry per {name} question.',
            'items': {'type': 'object', 'additionalProperties': False, 'required': list(props), 'properties': props}}


SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['characters', 'objects', 'locations'],
          'properties': {
              'characters': _group('character', {
                  'face': 'Face shape and facial features of this character in the two shots.',
                  'hair': 'Hair color, hairstyle, length and facial hair.',
                  'build': 'Body build and proportions.',
                  'clothing': 'Clothing and accessories; evidence of the same person only when the face is not visible.'},
                  'same_person', 'yes if the face, hair and build show the same person; if the face is not visible in '
                                 'one shot, the hair, build and clothing must all match; otherwise no.'),
              'objects': _group('object', {
                  'shape': 'Shape, form and silhouette.',
                  'color_texture': 'Color, texture and material.',
                  'details': 'Distinctive markings and fine details.'},
                  'same_object', 'yes only if it is the same specific item in both shots; otherwise no.'),
              'locations': _group('location', {
                  'layout': 'Architecture, room layout and fixed structures visible in both shots.',
                  'landmarks': 'Distinctive background features or objects that identify the place.',
                  'color_mood': 'Color palette and lighting of the place.'},
                  'same_place', 'yes only if fixed background features show the same specific place; otherwise no.')}}
GROUP_OF = {'character': ('characters', 'same_person'), 'object': ('objects', 'same_object'),
            'place': ('locations', 'same_place')}
TAG = {'character': 'character', 'object': 'object', 'place': 'location'}


def load(path):
    return json.loads(Path(path).read_text())


def safe(key: str) -> str:
    return key.replace('/', '_').replace('|', '__')


def pair_questions(requirements: list[dict], failed: set[str]) -> dict[tuple[int, int], list[dict]]:
    """(前镜, 后镜) → 该镜对上前提成立的人物／物体／地点题，按题号排。"""
    out = defaultdict(list)
    for q in requirements:
        if q['dimension'] in DIMENSIONS and q['requirement_id'] not in failed:
            if q['reference_shot'] in (None, q['shot']):
                raise ValueError(f"{q['requirement_id']}: consistency question without a second shot")
            out[tuple(sorted((q['shot'], q['reference_shot'])))].append(q)
    return dict(out)


def registry(request_text: str, questions: list[dict]) -> str:
    """登记表里本对题目点到的人、物、地点各一行（请求原文）。"""
    names = set(_ENTITY.findall(' '.join(q['question'] for q in questions)))
    lines = [line for line in request_text.splitlines()
             if (m := re.match(r'- ((?:Character|Object|Location) \d+)\b', line)) and m.group(1) in names]
    if len(lines) != len(names):
        raise ValueError(f'registry lines {len(lines)} for {sorted(names)}')
    return '\n'.join(lines)


def response_schema(questions: list[dict]) -> dict:
    """每组只允许本请求中该组的题号，并要求相应条数；逐号完整性仍由 map_answers 检查。"""
    schema = copy.deepcopy(SCHEMA)
    for dimension, (group, _) in GROUP_OF.items():
        numbers = [n for n, question in enumerate(questions, 1) if question['dimension'] == dimension]
        spec = schema['properties'][group]
        spec['minItems'] = spec['maxItems'] = len(numbers)
        if numbers:
            spec['items']['properties']['id']['enum'] = numbers
    return schema


def request_body(request_text: str, pair: tuple[int, int], questions: list[dict], images: list[dict],
                 thinking: str) -> dict:
    """``images`` 是登记好的六帧 Files 资源（前镜首中末、后镜首中末）。"""
    parts = [{'text': '## Registry\n' + registry(request_text, questions)}]
    labels = [f'Shot {shot} ({which} shot), {position} frame'
              for shot, which in zip(pair, ('earlier', 'later')) for position in POSITIONS]
    for label, file in zip(labels, images, strict=True):
        parts += [{'text': label}, g.file_part(file)]
    lines = [f"{n}. [{TAG[q['dimension']]}] " + re.sub(r'\{shot (\d+)\}', r'Shot \1', q['question'])
             for n, q in enumerate(questions, 1)]
    parts.append({'text': '## Questions\n' + '\n'.join(lines) + '\n\n' + CLOSING})
    if thinking not in THINKING:
        raise ValueError(f'thinking must be one of {THINKING}')
    return g.json_request(parts, response_schema(questions), system=SYSTEM, thinking=thinking, max_output_tokens=MAX_OUTPUT_TOKENS,
                          media_resolution=MEDIA_RESOLUTION)


def map_answers(questions: list[dict], answer: dict) -> dict[str, str]:
    """题号 → yes／no；每一组的题号必须恰好是这一类题在本请求里的编号，否则报格式错。"""
    out = {}
    for dimension, (group, verdict) in GROUP_OF.items():
        numbers = {n for n, q in enumerate(questions, 1) if q['dimension'] == dimension}
        given = [item['id'] for item in answer[group]]
        if sorted(given) != sorted(numbers):
            raise ValueError(f'{group}: ids {given} differ from {sorted(numbers)}')
        for item in answer[group]:
            out[questions[item['id'] - 1]['requirement_id']] = item[verdict]
    return out


def judge_record(line: dict, questions: list[dict]) -> dict:
    if 'error' in line:
        return {'status': 'request_failed', 'error': line['error'], 'cost_usd': None}
    raw = line['response']
    usage = raw.get('usageMetadata') or {}
    cost = str(g.cost(usage, 'batch')) if 'promptTokenCount' in usage else None
    try:
        answers = map_answers(questions, g.answer_json(raw))
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return {'status': 'format_failed', 'error': f'{type(error).__name__}: {error}', 'raw': raw, 'usage': usage,
                'cost_usd': cost}
    return {'status': 'answered', 'answers': answers, 'api': 'batchGenerateContent',
            'model': raw.get('modelVersion', g.MODEL), 'usage': usage, 'cost_usd': cost, 'raw': raw}


def prepare(root: Path, media_root: Path, bank_path: Path, cut_items_path: Path, cut_root: Path,
            cut_paths: list[Path], rows: list[str], storage: g.Storage) -> dict:
    """逐行：按前提筛题、按镜对分组、取帧写判分桶（主服务器），写 ``pairs/``、``skipped/`` 与 ``images/``。"""
    bank, items = load(bank_path)['stories'], load(cut_items_path)['stories']
    cuts = {}
    for path in cut_paths:
        cuts.update(load(path)['rows'])
    summary = {}
    for row in rows:
        media = load(media_root / 'rows' / row / 'media-map.json')
        verdicts = load(cut_root / 'verdicts' / f'{row}.json')
        stories = [s['story_id'] for s in load(media_root / 'rows' / row / 'prepared.json')['stories']]
        pairs, skipped, video_jobs = {}, {}, []
        for sid in stories:
            failed = premise_failed(items[sid]['items'], verdicts)
            requirements = bank[sid]['requirements']
            skipped[sid] = [q['requirement_id'] for q in requirements
                            if q['dimension'] in DIMENSIONS and q['requirement_id'] in failed]
            groups = pair_questions(requirements, failed)
            if not groups:
                continue
            m = media[sid]
            bounds, _ = boundaries(m['shot_frames'], m['probe']['fps'], cuts[row][sid]['detected_cut_times_seconds'])
            jobs = []
            for (a, b), questions in sorted(groups.items()):
                key = f'{row}|{sid}|{a}-{b}'
                frames = [*shot_triplet_frames(bounds, a, m['probe']['frames']),
                          *shot_triplet_frames(bounds, b, m['probe']['frames'])]
                pairs[key] = {'story': sid, 'shots': [a, b], 'questions': [q['requirement_id'] for q in questions],
                              'frames': frames}
                jobs.append({'key': key, 'frames': frames, 'judge_frames': frames})
            video_jobs.append({'video': m['remote_video_path'], 'story': sid, 'pairs': jobs})
        image_path = root / 'images' / f'{row}.json'
        images = load(image_path) if image_path.exists() else {}
        missing = [{**j, 'pairs': [p for p in j['pairs'] if p['key'] not in images]} for j in video_jobs]
        if any(j['pairs'] for j in missing):
            images.update(frames_to_bucket(storage, 'pairs', row, [j for j in missing if j['pairs']]))
            g.write_json(image_path, images)
        g.write_json(root / 'pairs' / f'{row}.json', pairs)
        g.write_json(root / 'skipped' / f'{row}.json', skipped)
        summary[row] = {'pairs': len(pairs), 'questions': sum(len(p['questions']) for p in pairs.values()),
                        'skipped_premise_failed': sum(len(v) for v in skipped.values())}
    return summary


def receipt_path(root: Path, row: str, key: str) -> Path:
    return root / 'receipts' / row / f'{safe(key)}.json'


def pending(root: Path, rows: list[str]) -> list[tuple[str, str]]:
    """只提交从未记入批任务且没有回执的镜对；失败或缺失回执保留作测量缺失。"""
    submitted = {key for batch in g.batch_records(root) for key in batch['keys']}
    out = []
    for row in rows:
        for key in load(root / 'pairs' / f'{row}.json'):
            path = receipt_path(root, row, key)
            if path.exists():
                status = load(path)['status']
                if status not in ('answered', 'format_failed', 'request_failed'):
                    raise ValueError(f'{path}: unknown receipt status {status}')
                continue
            if key in submitted:
                continue
            out.append((row, key))
    return out


def submit(root: Path, rows: list[str], max_requests: int, cap: D, thinking: str, client: g.Gemini) -> dict | None:
    settings = {'model': g.MODEL, 'thinking': thinking, 'media_resolution': MEDIA_RESOLUTION, 'prompt_version': 2}
    if (root / 'judge.json').exists():
        if load(root / 'judge.json') != settings:
            raise ValueError(f"{root}: judge settings {load(root / 'judge.json')} differ from {settings}")
    else:
        g.write_json(root / 'judge.json', settings)
    todo = pending(root, rows)[:max_requests]
    if not todo:
        return None
    bank = load(load_config_bank(root))['stories']
    path = g.open_batch(root, 'consistency', [k for _, k in todo], RESERVE_PER_REQUEST * len(todo), cap,
                        10_000 * len(todo))
    g.write_json(path, {**load(path), 'schema_revision': SCHEMA_REVISION})
    images = {row: load(root / 'images' / f'{row}.json') for row in {r for r, _ in todo}}
    registered = client.register_all([u for row, key in todo for u in images[row][key]])
    requests = []
    for row, key in todo:
        pair = load(root / 'pairs' / f'{row}.json')[key]
        rows_of = {q['requirement_id']: q for q in bank[pair['story']]['requirements']}
        requests.append((key, request_body(bank[pair['story']]['request'], tuple(pair['shots']),
                                           [rows_of[r] for r in pair['questions']],
                                           [registered[u] for u in images[row][key]], thinking)))
    return g.start_batch(client, path, requests, {u: f['name'] for u, f in registered.items()})


def load_config_bank(root: Path) -> Path:
    return Path(load(root / 'config.json')['bank'])


def collect(root: Path, client: g.Gemini) -> list[dict]:
    bank = load(load_config_bank(root))['stories']

    def handle(record: dict, lines: list[dict]) -> tuple[D, dict]:
        cost, written = D(0), Counter()
        for line in lines:
            row = line['key'].split('|', 1)[0]
            pair = load(root / 'pairs' / f'{row}.json')[line['key']]
            rows_of = {q['requirement_id']: q for q in bank[pair['story']]['requirements']}
            data = {'key': line['key'], 'row': row, 'batch': record['name'], 'finished_at': g.now(),
                    **judge_record(line, [rows_of[r] for r in pair['questions']])}
            g.write_json(receipt_path(root, row, line['key']), data)
            written[data['status']] += 1
            cost += D(data['cost_usd'] or 0)
        return cost, dict(written)
    return g.collect_batches(client, root, handle)


def answers(root: Path, row: str) -> dict[str, str]:
    """该行全部已判的题：题号 → yes／no；判官安全过滤拒收的镜对，其题记 ``refused``（计不过）。
    拒收沿用切镜判官的既有规则（``cut_execution.refused``：同看不清，不重发、不声称判官看到了画面）；
    11-15 测试集 cinedance-014083-b0000-shot01 的三组镜对在输入阶段被 PROHIBITED_CONTENT 拒收，同一故事此前切镜也被拒。"""
    out = {}
    for key, pair in load(root / 'pairs' / f'{row}.json').items():
        path = receipt_path(root, row, key)
        if not path.exists():
            continue
        record = load(path)
        if record['status'] == 'answered':
            out.update(record['answers'])
        elif refused(record):
            out.update(dict.fromkeys(pair['questions'], 'refused'))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--media-root', type=Path, required=True, help='含 rows/<行>/media-map.json 与 prepared.json 的判分根')
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--cut-items', type=Path, required=True)
    p.add_argument('--cut-root', type=Path, required=True, help='cut_execution 运行根（verdicts/<行>.json 已判完）')
    p.add_argument('--cuts', type=Path, nargs='+', required=True, help='checklist_scoring stored-cuts／detect-cuts 的切点表')
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--service-account-file', type=Path, required=True, help='判分桶上传服务账号 JSON 密钥（只在本机）')
    p.add_argument('--bucket', required=True)
    p.add_argument('--proxy', required=True, help='本机 HTTP 代理（签令牌用）')
    for name in ('submit', 'collect'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, required=True)
        sub.add_argument('--key-file', type=Path, required=True)
        sub.add_argument('--proxy', required=True)
        if name == 'submit':
            sub.add_argument('--rows', nargs='+', required=True)
            sub.add_argument('--max-requests', type=int, required=True)
            sub.add_argument('--cap-usd', type=D, required=True)
            sub.add_argument('--thinking', choices=THINKING, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not (args.root / 'config.json').exists():
            g.write_json(args.root / 'config.json', {'bank': str(args.bank), 'cut_items': str(args.cut_items),
                                                     'cut_root': str(args.cut_root), 'media_root': str(args.media_root)})
        elif load(args.root / 'config.json')['bank'] != str(args.bank):
            raise ValueError(f'{args.root}: prepared with another bank')
        storage = g.Storage(args.service_account_file, args.bucket, args.proxy)
        print(json.dumps(prepare(args.root, args.media_root, args.bank, args.cut_items, args.cut_root, args.cuts,
                                 args.rows, storage), indent=1))
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
