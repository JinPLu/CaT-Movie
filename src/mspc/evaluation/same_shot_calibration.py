"""两镜前提题（question_writer 规则 4）的校准材料与读数：真片行与照抄臂行。

真片行：开发集每个故事按源片逐镜截取（第 5 版对源片复核时的源片窗口：源视频、帧率、裁切框、逐镜起止帧），每镜重采样到
给定分镜的帧数，拼成 16fps、832×480 的整片；镜界就是计划切点，所有前提题都应答 no。
照抄臂：同一批源片，每个故事至多换两镜。第一个要求变化的相邻计划切点 (k, k+1)，把第 k+1 镜换成第 k 镜的画面；第一个
不相邻、且不碰这两镜的前提镜对 (i, j)，把第 j 镜换成第 i 镜的画面。换上的画面按目标镜帧数重采样，沿用 10-05 构图题复验
的照抄臂（同一画面重采样）。被换的镜对应答 yes；不涉及被换镜的镜对仍应答 no；涉及被换镜但不是被换镜对的不计。
两行的目录与方法成片同构（``generation.json``、``video.mp4``、``segments.json``），由 ``checklist_rolling`` 照常判分。

用法::

    # 本机：构造计划（源片窗口、逐镜帧数、照抄目标与每道前提题的期望答案）
    PYTHONPATH=src python -m mspc.evaluation.same_shot_calibration plan --windows FRAMES_DEV \\
        --storyboards configs/benchmark/benchmark_storyboards_v4.json --bank questions.same-shot-additions.json \\
        --split dev --output PLAN.json
    # 本机发起，在主服务器 CPU 机上只用 ffmpeg 出片
    PYTHONPATH=src python -m mspc.evaluation.same_shot_calibration build --plan PLAN.json --ssh 'ssh -p 4974 root@HOST' \\
        --ffmpeg FFMPEG --ffprobe FFPROBE --output-root REMOTE_DIR --workers 8
    # 判分后
    PYTHONPATH=src python -m mspc.evaluation.same_shot_calibration readout --root ROLLING_ROOT --plan PLAN.json \\
        --source-row ROW --copy-row ROW --output O.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections import Counter
from itertools import accumulate
from pathlib import Path

WIDTH, HEIGHT, FPS = 832, 480, 16
#: TransNetV2 拼接判定：第 i 镜末尾与第 j 镜开头各取至多这么多秒；拼接点前后这么多秒内检出切点算不同镜头。
SPLICE_SECONDS, WINDOW_SECONDS = 2.0, 0.5
CALIBRATION_ROWS = {'source-film': 'dev30-same-shot-calibration-source-film',
                    'copy-arm': 'dev30-same-shot-calibration-copy-arm'}


def copy_targets(premises: list[dict]) -> dict[int, int]:
    """被换镜 → 换上哪一镜的画面；按前提题顺序取第一个相邻计划切点与第一个不碰它的不相邻镜对。"""
    copies: dict[int, int] = {}
    adjacent = next((r for r in premises if r['planned_cut']), None)
    if adjacent is not None:
        copies[adjacent['shot']] = adjacent['reference_shot']
    for r in premises:
        i, j = r['reference_shot'], r['shot']
        if r['planned_cut'] or j in copies or j in copies.values() or i in copies:
            continue
        copies[j] = i
        break
    return copies


def expected(premise: dict, copies: dict[int, int]) -> str:
    i, j = premise['reference_shot'], premise['shot']
    if copies.get(j) == i:
        return 'yes'
    if i in copies or j in copies:
        return 'excluded'
    return 'no'


def plan(windows: Path, boards: dict, bank: dict, split: str) -> dict:
    stories = {}
    for sid, story in sorted(bank['stories'].items()):
        if story['split'] != split:
            continue
        window = json.loads((windows / sid / 'frames.json').read_text())
        shots = boards[sid]['storyboard']['shots']
        if len(shots) != len(window['shots']):
            raise ValueError(f'{sid}: window and storyboard shot counts differ')
        premises = story['requirements']
        copies = copy_targets(premises)
        stories[sid] = {
            'source_video_path': window['source_video_path'], 'fps': window['fps'], 'crop_box': window['crop_box'],
            'container_preroll_frames': window['container_preroll_frames'] or 0,
            'shots': [{'shot': s['shot'], 'start_frame': w['start_frame'], 'end_frame': w['end_frame'],
                       'frames': s['frame_count']} for s, w in zip(shots, window['shots'])],
            'copies': {str(j): i for j, i in copies.items()},
            'expected': {r['requirement_id']: expected(r, copies) for r in premises}}
    return {'split': split, 'width': WIDTH, 'height': HEIGHT, 'fps': FPS, 'stories': stories}


# 在主服务器上执行：只用标准库与 ffmpeg；读计划与参数（stdin 的 JSON），每个故事写真片与照抄臂两个目录。
REMOTE_BUILDER = r'''
import json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
P = json.load(sys.stdin)
plan, ffmpeg, ffprobe, out_root = P['plan'], P['ffmpeg'], P['ffprobe'], P['output_root']

def frames_of(video):
    s = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames', '-of', 'json', video]))['streams'][0]
    return int(s['nb_read_frames'])

def build(sid, story):
    shots, fps, pre = story['shots'], story['fps'], story['container_preroll_frames']
    first, last = shots[0]['start_frame'], shots[-1]['end_frame']
    x0, y0, x1, y1 = story['crop_box']
    tmp = tempfile.mkdtemp(prefix='same-shot-' + sid + '-')
    try:
        lead = max(0, int(first + pre - 2 * fps))
        a, b = first + pre - lead, last + pre - lead - 1
        vf = (f"select=between(n\\,{a}\\,{b}),crop={x1 - x0}:{y1 - y0}:{x0}:{y0},"
              f"scale={plan['width']}:{plan['height']}")
        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-y', '-ss', f'{lead / fps:.6f}', '-i',
                        story['source_video_path'], '-vf', vf, '-fps_mode', 'passthrough', '-q:v', '2',
                        f'{tmp}/src%06d.jpg'], check=True)
        extracted = len([n for n in os.listdir(tmp) if n.startswith('src')])
        if extracted != last - first:
            raise ValueError(f'{sid}: extracted {extracted} frames, expected {last - first}')
        def picks(shot, count):
            span = shot['end_frame'] - shot['start_frame']
            return [shot['start_frame'] - first + t * span // count for t in range(count)]
        result = {}
        for variant, copies in (('source-film', {}), ('copy-arm', story['copies'])):
            seq = os.path.join(tmp, variant)
            os.makedirs(seq)
            index = 0
            for shot in shots:
                source = shots[copies[str(shot['shot'])] - 1] if str(shot['shot']) in copies else shot
                for frame in picks(source, shot['frames']):
                    index += 1
                    os.symlink(f'{tmp}/src{frame + 1:06d}.jpg', f'{seq}/{index:06d}.jpg')
            directory = os.path.join(out_root, variant, sid)
            os.makedirs(directory, exist_ok=True)
            video = os.path.join(directory, 'video.mp4')
            subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-y', '-framerate', str(plan['fps']), '-i',
                            f'{seq}/%06d.jpg', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                            '-pix_fmt', 'yuv420p', video], check=True)
            total = sum(s['frames'] for s in shots)
            if frames_of(video) != total:
                raise ValueError(f'{sid}/{variant}: encoded frame count differs from {total}')
            json.dump([{'start_shot': s['shot'] - 1, 'end_shot': s['shot'], 'frame_counts': [s['frames']],
                        'num_frames': s['frames']} for s in shots], open(os.path.join(directory, 'segments.json'), 'w'))
            json.dump({'status': 'completed', 'stage': 'same_shot_calibration', 'variant': variant,
                       'source_video_path': story['source_video_path'], 'copies': copies},
                      open(os.path.join(directory, 'generation.json'), 'w'))
            result[variant] = total
        return sid, result
    finally:
        shutil.rmtree(tmp)

with ThreadPoolExecutor(P['workers']) as pool:
    done = list(pool.map(lambda item: build(*item), plan['stories'].items()))
print(json.dumps({'built': len(done), 'frames': dict(done)}))
'''


def build(plan_path: Path, ssh: str, ffmpeg: str, ffprobe: str, output_root: str, workers: int) -> dict:
    payload = json.dumps({'plan': json.loads(plan_path.read_text()), 'ffmpeg': ffmpeg, 'ffprobe': ffprobe,
                          'output_root': output_root, 'workers': workers})
    script = f'import sys, io; sys.stdin = io.StringIO({payload!r})\n' + REMOTE_BUILDER
    out = subprocess.run(shlex.split(ssh) + ['python3', '-'], input=script, capture_output=True, text=True,
                         check=True, timeout=7200).stdout
    return json.loads(out.strip().splitlines()[-1])


def splice_ranges(shot_frames: list[int], fps: float, earlier: int, later: int) -> list[list[int]]:
    """第 ``earlier`` 镜末尾与第 ``later`` 镜开头的帧号区间（左闭右开，整片帧号），各至多 ``SPLICE_SECONDS`` 秒。"""
    starts = [0, *accumulate(shot_frames)]
    take = round(SPLICE_SECONDS * fps)
    return [[max(starts[earlier - 1], starts[earlier] - take), starts[earlier]],
            [starts[later - 1], min(starts[later], starts[later - 1] + take)]]


def calibration_jobs(plan_data: dict, bank: dict, remote_root: str) -> list[dict]:
    """真片行与照抄臂行的全部前提镜对（相邻与不相邻都拼接）。"""
    jobs = []
    for sid, story in plan_data['stories'].items():
        frames = [shot['frames'] for shot in story['shots']]
        for q in bank['stories'][sid]['requirements']:
            for variant, row in CALIBRATION_ROWS.items():
                jobs.append({'key': f"{row}|{q['requirement_id']}", 'video': f'{remote_root}/{variant}/{sid}/video.mp4',
                             'fps': plan_data['fps'],
                             'ranges': splice_ranges(frames, plan_data['fps'], q['reference_shot'], q['shot'])})
    return jobs


def generated_jobs(main_root: Path, rows: list[str], bank: dict) -> list[dict]:
    """生成片的不相邻前提镜对；镜时与远端成片取判分行的 ``media-map.json``（与判官看的同一部、同一张镜时表）。"""
    jobs = []
    for row in rows:
        media = json.loads((main_root / 'rows' / row / 'media-map.json').read_text())
        split = json.loads((main_root / 'rows' / row / 'prepared.json').read_text())['stories']
        for story in split:
            sid = story['story_id']
            entry = media[sid]
            for q in bank['stories'][sid]['requirements']:
                if q['planned_cut']:
                    continue
                jobs.append({'key': f"{row}|{q['requirement_id']}", 'video': entry['remote_video_path'],
                             'fps': entry['probe']['fps'],
                             'ranges': splice_ranges(entry['shot_frames'], entry['probe']['fps'],
                                                     q['reference_shot'], q['shot'])})
    return jobs


def adjacent_from_stored(main_root: Path, rows: dict[str, str], bank: dict, metrics_root: Path) -> dict[str, dict]:
    """生成片相邻前提镜对：整片已存的 TransNetV2 检出切点在该镜界 ±``WINDOW_SECONDS`` 秒内即不同镜头。"""
    out = {}
    for row, method in rows.items():
        for story in json.loads((main_root / 'rows' / row / 'prepared.json').read_text())['stories']:
            sid = story['story_id']
            record = json.loads((metrics_root / method / method / 'shots' / 'seed-42' / f'{sid}.json').read_text())
            cuts = record['non_vlm']['detected_cut_times_seconds']
            ends = list(accumulate(story['shot_seconds']))
            for q in bank['stories'][sid]['requirements']:
                if not q['planned_cut']:
                    continue
                boundary = ends[q['reference_shot'] - 1]
                out[f"{row}|{q['requirement_id']}"] = {
                    'different': any(abs(c - boundary) <= WINDOW_SECONDS for c in cuts), 'source': 'stored_full_video'}
    return out


# 在主服务器 GPU 机上用 TF-only 解释器执行：模型只加载一次；峰值与切点规则用 renderer_qualification_public_metrics 原函数。
REMOTE_SPLICE = r'''
import json, sys
from collections import defaultdict
P = json.load(sys.stdin)
sys.path.insert(0, P['repo'] + '/tools/run')
import transnet_worker
official = transnet_worker.load_backend_without_torch()
public = sys.modules['mspc.evaluation.renderer_qualification_public_metrics']
import cv2
from pathlib import Path
engine = official._TransNetEngine(Path(P['resource_root']), P['device'])
by_video = defaultdict(list)
for job in P['jobs']:
    by_video[job['video']].append(job)
out = {}
for video, jobs in by_video.items():
    need = sorted({f for job in jobs for a, b in job['ranges'] for f in range(a, b)})
    capture, frames, index = cv2.VideoCapture(video), {}, 0
    wanted = set(need)
    while index <= need[-1]:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f'{video}: decoded only {index} frames, need {need[-1] + 1}')
        if index in wanted:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    for job in jobs:
        clip = [frames[f] for a, b in job['ranges'] for f in range(a, b)]
        splice = job['ranges'][0][1] - job['ranges'][0][0]
        probabilities = engine.predict(clip)
        peaks = public.transition_run_peaks(probabilities, threshold=P['threshold'])
        boundaries = list(public.cut_boundaries_from_peaks(peaks, frame_count=len(clip)))
        out[job['key']] = {'different': any(abs(b - splice) <= P['window_seconds'] * job['fps'] for b in boundaries),
                           'boundaries': boundaries, 'splice': splice, 'frames': len(clip),
                           'max_probability_near_splice': max(probabilities[max(0, splice - 3):splice + 3])}
print(json.dumps(out))
'''


def run_splice(jobs: list[dict], ssh: str, python: str, repo: str, resource_root: str, cuda_devices: str | None) -> dict:
    """``cuda_devices`` 为 None 时在 CPU 上跑（11-04 CPU 整片重跑 5 部与 A100 已存检出逐个相同）。"""
    payload = json.dumps({'jobs': jobs, 'repo': repo, 'resource_root': resource_root,
                          'device': 'cpu' if cuda_devices is None else 'cuda',
                          'threshold': 0.5, 'window_seconds': WINDOW_SECONDS})
    script = f'import sys, io; sys.stdin = io.StringIO({payload!r})\n' + REMOTE_SPLICE
    command = (f'CUDA_VISIBLE_DEVICES= {shlex.quote(python)} -' if cuda_devices is None else
               f'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={shlex.quote(cuda_devices)} {shlex.quote(python)} -')
    out = subprocess.run(shlex.split(ssh) + [command], input=script, capture_output=True, text=True,
                         check=True, timeout=14400).stdout
    return json.loads(out.strip().splitlines()[-1])


def answers(root: Path, row: str) -> dict[str, str]:
    out = {}
    for path in (root / 'rows' / row / 'output' / 'metrics' / 'first-read').rglob('*.json'):
        for answer in (json.loads(path.read_text()).get('result') or {}).get('answers', []):
            out[answer['requirement_id']] = answer['answer']
    return out


def buckets(premise: dict, shot_seconds: list[float]) -> dict:
    """预先写定的读数分档：景别差、相邻与否、差异类型（可重叠）、两镜中较短那镜的时长。"""
    difference = premise['planned_difference']
    shorter = min(shot_seconds[premise['reference_shot'] - 1], shot_seconds[premise['shot'] - 1])
    return {'size_gap': f"{min(difference['size_gap'], 2)}{'+' if difference['size_gap'] >= 2 else ''}",
            'adjacency': 'adjacent' if premise['planned_cut'] else 'apart',
            'types': [t for t, on in (('place', difference['place']), ('cast', difference['cast']),
                                      ('angle', difference['angle']), ('size', difference['size_gap'] > 0)) if on],
            'shorter_shot': '<1s' if shorter < 1 else '1-2.5s' if shorter < 2.5 else '>=2.5s'}


def calibration_pairs(root: Path, plan_data: dict, bank: dict, splice: dict) -> list[dict]:
    """真片行与照抄臂行的逐镜对表：期望答案、判官与 TransNetV2 拼接各自判"同一镜头"与否、分档。"""
    premises = {r['requirement_id']: r for s in bank['stories'].values() for r in s['requirements']}
    out = []
    for variant, row in CALIBRATION_ROWS.items():
        judged = answers(root, row)
        for sid, story in plan_data['stories'].items():
            seconds = [shot['frames'] / plan_data['fps'] for shot in story['shots']]
            for rid, want in story['expected'].items():
                truth = 'no' if variant == 'source-film' else want
                if truth == 'excluded':
                    continue
                out.append({'row': row, 'requirement_id': rid, 'truth_same': truth == 'yes',
                            'vlm_same': None if rid not in judged else judged[rid] == 'yes',
                            'transnet_same': not splice[f'{row}|{rid}']['different'],
                            **buckets(premises[rid], seconds)})
    return out


def rates(pairs: list[dict], truth_same: bool) -> dict:
    """在真值为 ``truth_same`` 的镜对上，两种方法判成"同一镜头"的比例；按预定各分档分开报。"""
    chosen = [p for p in pairs if p['truth_same'] == truth_same]

    def one(group):
        out = {}
        for method in ('vlm', 'transnet'):
            judged = [p[f'{method}_same'] for p in group if p[f'{method}_same'] is not None]
            out[method] = {'same': sum(judged), 'total': len(judged), 'rate': sum(judged) / len(judged) if judged else None}
        return out

    result = {'all': one(chosen)}
    for key in ('size_gap', 'adjacency', 'shorter_shot'):
        for value in sorted({p[key] for p in chosen}):
            result[f'{key}={value}'] = one([p for p in chosen if p[key] == value])
    for value in ('place', 'cast', 'angle', 'size'):
        result[f'types∋{value}'] = one([p for p in chosen if value in p['types']])
    return result


def readout(root: Path, plan_data: dict, bank: dict, splice: dict) -> dict:
    pairs = calibration_pairs(root, plan_data, bank, splice)
    source = [p for p in pairs if p['row'] == CALIBRATION_ROWS['source-film']]
    copied = [p for p in pairs if p['row'] == CALIBRATION_ROWS['copy-arm']]
    return {'source_film_false_same': rates(source, False),
            'copy_arm_caught': rates(copied, True),
            'copy_arm_untouched_false_same': rates(copied, False),
            'pairs': pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('plan')
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--storyboards', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--split', choices=('dev', 'test'), required=True)
    p.add_argument('--output', type=Path, required=True)
    b = commands.add_parser('build')
    b.add_argument('--plan', type=Path, required=True)
    b.add_argument('--ssh', required=True)
    b.add_argument('--ffmpeg', required=True)
    b.add_argument('--ffprobe', required=True)
    b.add_argument('--output-root', required=True)
    b.add_argument('--workers', type=int, required=True)
    s = commands.add_parser('splice')
    s.add_argument('--bank', type=Path, required=True)
    s.add_argument('--plan', type=Path, help='校准构造计划：判真片行与照抄臂行')
    s.add_argument('--calibration-remote-root', help='主服务器上校准片的根目录')
    s.add_argument('--main-root', type=Path, help='第 6 版主运行根：判这些生成行的不相邻镜对')
    s.add_argument('--rows', nargs='+', default=[])
    s.add_argument('--ssh', required=True)
    s.add_argument('--python', required=True, help='主服务器上 TF-only 的 TransNetV2 解释器')
    s.add_argument('--repo', required=True, help='主服务器上与本机同一提交的代码目录')
    s.add_argument('--resource-root', required=True)
    s.add_argument('--cuda-devices', required=True)
    s.add_argument('--output', type=Path, required=True)
    a = commands.add_parser('adjacent')
    a.add_argument('--bank', type=Path, required=True)
    a.add_argument('--main-root', type=Path, required=True)
    a.add_argument('--metrics-root', type=Path, required=True)
    a.add_argument('--pair', nargs=2, action='append', metavar=('ROW', 'METHOD'), required=True)
    a.add_argument('--output', type=Path, required=True)
    r = commands.add_parser('readout')
    r.add_argument('--root', type=Path, required=True)
    r.add_argument('--plan', type=Path, required=True)
    r.add_argument('--bank', type=Path, required=True)
    r.add_argument('--splice', type=Path, required=True, help='splice 命令对校准两行的输出')
    r.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        boards = {s['story_id']: s for s in json.loads(args.storyboards.read_text())['stories']}
        result = plan(args.windows, boards, json.loads(args.bank.read_text()), args.split)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
        expectations = Counter(w for s in result['stories'].values() for w in s['expected'].values())
        print(json.dumps({'stories': len(result['stories']), 'expected': dict(expectations)}))
    elif args.command == 'build':
        print(json.dumps(build(args.plan, args.ssh, args.ffmpeg, args.ffprobe, args.output_root, args.workers)))
    elif args.command == 'splice':
        bank = json.loads(args.bank.read_text())
        jobs = []
        if args.plan is not None:
            jobs += calibration_jobs(json.loads(args.plan.read_text()), bank, args.calibration_remote_root)
        if args.rows:
            jobs += generated_jobs(args.main_root, args.rows, bank)
        result = run_splice(jobs, args.ssh, args.python, args.repo, args.resource_root, args.cuda_devices)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
        print(json.dumps({'jobs': len(jobs), 'different': sum(v['different'] for v in result.values())}))
    elif args.command == 'adjacent':
        result = adjacent_from_stored(args.main_root, dict(args.pair), json.loads(args.bank.read_text()),
                                      args.metrics_root)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
        print(json.dumps({'pairs': len(result), 'different': sum(v['different'] for v in result.values())}))
    else:
        result = readout(args.root, json.loads(args.plan.read_text()), json.loads(args.bank.read_text()),
                         json.loads(args.splice.read_text()))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
        print(json.dumps({k: v['all'] for k, v in result.items() if k != 'pairs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
