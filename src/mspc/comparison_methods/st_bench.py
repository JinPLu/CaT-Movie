"""社区基准 ST-Bench 上对比方法的原生输入与逐故事出片：角色登记 → 各方法原生输入 → 出片。

题集读取与评分归 mspc.evaluation.st_bench。
Wan 逐镜与 StoryMem 直接吃 ST-Bench 原文（StoryMem 跑官方 pipeline.py）。HoloCine、MultiShotMaster 与两条
关键帧路线要角色标签：每个故事一次 LLM 转换，只从原文摘出角色外貌与每镜的原文指称，冻结后四家共用
（仿 StoryMem 论文用 GPT-5 把 ST-Bench 转成 HoloCine 格式）。
"""
from __future__ import annotations

import argparse
from decimal import ROUND_CEILING, Decimal
from itertools import accumulate
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mspc.comparison_methods.prompts import holocine_inputs, pack_segments
from mspc.evaluation.benchmark import write_json

RENDERER_FAMILIES = ('wan_single', 'multishotmaster', 'holocine')
KEYFRAME_METHODS = {'storydiffusion_wan_a14b': 'storydiffusion', 'ic_lora_wan_a14b': 'ic_lora'}

REGISTRY_INSTRUCTIONS = """You prepare a fixed multi-shot story script for video generators that refer to recurring characters by tags. Work only from the given text; never add, infer or embellish anything.

A character is one individual person, animal, robot or creature that acts or is shown as a subject in at least one shot. Groups (guards, soldiers, parents, siblings, a herd), objects and places are not characters. Number characters "Character 1", "Character 2", ... in order of first appearance; two different individuals are always two characters.

For each character write `description`: a noun phrase saying what the character is and how it looks when it first appears, reusing only words the script itself uses (for example "a fluffy orange-and-white cat" or "a young man in a black T-shirt"). It must start with what the character is (a man, a girl, a dog, a robot...), never with clothing alone or a bare name, and it must not describe later changes of costume or state. If the script states nothing about the look, use the plainest noun phrase it uses (for example "a man"). Different characters must get different descriptions.

Also copy `name`: the proper name the script gives this character (for example "Elon Musk", "Harry Potter", "Lin Daiyu"), in its fullest form, exactly as written in the script; null if the script never names it. Keep the name out of `description`.

For every shot, in the given order, list the characters visible in it. For each listed character copy:
- `video_mention`: the exact span of that shot's video prompt where the character is first named or described in that prompt, copied character for character with the same capitalization and including a leading article, or null if that prompt refers to it only by a pronoun or not at all;
- `first_frame_mention`: the same, from that shot's first-frame prompt.
Spans of different characters in the same prompt must not overlap."""


def registry_schema() -> dict:
    mention = {'type': ['string', 'null']}
    present = {'type': 'object', 'additionalProperties': False, 'required': ['tag', 'video_mention', 'first_frame_mention'],
               'properties': {'tag': {'type': 'string'}, 'video_mention': mention, 'first_frame_mention': mention}}
    return {'type': 'json_schema', 'json_schema': {'name': 'st_bench_character_registry', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False, 'required': ['characters', 'shots'],
        'properties': {
            'characters': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False, 'required': ['tag', 'name', 'description'],
                'properties': {'tag': {'type': 'string'}, 'name': {'type': ['string', 'null']},
                               'description': {'type': 'string'}}}},
            'shots': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False, 'required': ['shot', 'present'],
                'properties': {'shot': {'type': 'integer'}, 'present': {'type': 'array', 'items': present}}}}}}}}


def registry_user_text(story: dict) -> str:
    return json.dumps({'story_overview': story['overview'],
                       'shots': [{'shot': i, 'video_prompt': shot['prompt'], 'first_frame_prompt': shot['first_frame_prompt']}
                                 for i, shot in enumerate(story['shots'], 1)]}, ensure_ascii=False, indent=1)


def _spans(text: str, mentions: list[str]) -> list[tuple[int, int]]:
    """指称在原文里的位置：整词匹配、大小写不计（实测登记把句首 The 写成 the），替换时用原文区间。

    长指称先占位，各取第一个与已占区间不重叠的出现（实测 'the prince' 会先撞上 'The princess' 的前缀）。
    """
    if any(not mention.strip() for mention in mentions):
        raise ValueError('empty mention')
    lowered = text.lower()
    taken: dict[int, tuple[int, int]] = {}
    for index in sorted(range(len(mentions)), key=lambda i: -len(mentions[i])):
        needle = mentions[index].lower()
        start = lowered.find(needle)
        while start >= 0:
            end = start + len(needle)
            # 只在指称本身以字母数字起止时要求词界（实测 santa 的指称以破折号收尾，后面紧跟 sits）。
            whole = ((start == 0 or not needle[0].isalnum() or not text[start - 1].isalnum())
                     and (end == len(text) or not needle[-1].isalnum() or not text[end].isalnum()))
            if whole and all(end <= a or start >= b for a, b in taken.values()):
                taken[index] = (start, end)
                break
            start = lowered.find(needle, start + 1)
        if index not in taken:
            raise ValueError(f'mention {mentions[index]!r} is not a free whole-word span of {text!r}')
    return [taken[i] for i in range(len(mentions))]


def validate_registry(story: dict, registry: dict) -> dict:
    """标签连续、每镜一行、指称逐字是原文片段且互不重叠、每个角色至少出场一次。"""
    tags = [row['tag'] for row in registry['characters']]
    if not tags or tags != [f'Character {i}' for i in range(1, len(tags) + 1)]:
        raise ValueError(f'character tags must be Character 1..N, got {tags}')
    texts = [story['overview'], *(shot[k] for shot in story['shots'] for k in ('prompt', 'first_frame_prompt'))]
    for row in registry['characters']:
        if row['name'] is not None and not any(_free(text, row['name']) for text in texts):
            raise ValueError(f"{row['tag']}: name {row['name']!r} is not a whole-word span of the script")
    # 区分性按方法实际看到的「名字＋描述」判：两个同为 "an ethereal vision" 的角色靠名字就能分开。
    identities = [character_text(row).strip().lower() if row['description'].strip() else ''
                  for row in registry['characters']]
    if not all(identities) or len(set(identities)) != len(identities):
        raise ValueError('character descriptions must be non-empty and distinct')
    if [row['shot'] for row in registry['shots']] != list(range(1, len(story['shots']) + 1)):
        raise ValueError('registry must list every shot once, in order')
    seen = set()
    for row, shot in zip(registry['shots'], story['shots']):
        present = [entry['tag'] for entry in row['present']]
        if len(set(present)) != len(present) or not set(present) <= set(tags):
            raise ValueError(f"shot {row['shot']}: unknown or repeated character tags {present}")
        seen.update(present)
        for field, text in (('video_mention', shot['prompt']), ('first_frame_mention', shot['first_frame_prompt'])):
            _spans(text, [entry[field] for entry in row['present'] if entry[field] is not None])
    if seen != set(tags):
        raise ValueError(f'characters never present: {sorted(set(tags) - seen)}')
    return registry


def _free(text: str, mention: str) -> bool:
    try:
        _spans(text, [mention])
        return True
    except ValueError:
        return False


def character_text(row: dict) -> str:
    """角色在标签方法里的全部身份信息：原文名字＋首次外貌（公平性修正：名字本身是文本编码器里的
    身份锚，吃原文的方法看得到，换成标签的方法也必须看得到）。描述里已含名字时不重复。"""
    description = row['description'].rstrip('.')
    name = row['name']
    if name is None or name.lower() in description.lower():
        return description
    return f'{name}, {description}'


def tag_text(text: str, mentions: dict[str, str | None], token) -> str:
    """把每个角色在该句的首次原文指称换成 token(角色号)；没有原文指称的不改。"""
    named = [(tag, mention) for tag, mention in mentions.items() if mention is not None]
    spans = _spans(text, [mention for _, mention in named])
    for (tag, _), (start, end) in sorted(zip(named, spans), key=lambda item: -item[1][0]):
        gap = ' ' if end < len(text) and text[end].isalnum() else ''  # 指称以标点收尾时补回词间空格
        text = text[:start] + token(int(tag.split()[-1])) + gap + text[end:]
    return text


def _mentions(registry: dict, index: int, field: str) -> dict:
    return {entry['tag']: entry[field] for entry in registry['shots'][index]['present']}


def _global_caption(story: dict, registry: dict, start: int, end: int, token) -> str:
    referenced = {entry['tag'] for row in registry['shots'][start:end] for entry in row['present']}
    lines = [f"{token(int(row['tag'].split()[-1]))}: {character_text(row)}."
             for row in registry['characters'] if row['tag'] in referenced]
    return ' '.join([story['overview'], *lines])


def native_segment(story: dict, registry: dict | None, family: str, start: int, end: int, frames: int) -> dict:
    counts = [frames] * (end - start)
    row = {'start_shot': start, 'end_shot': end, 'frame_counts': counts}
    if family == 'wan_single':
        return {**row, 'prompts': [shot['prompt'] for shot in story['shots'][start:end]], 'num_frames': sum(counts)}
    token = (lambda k: f'[character{k}]') if family == 'holocine' else (lambda k: f'Subject {k}')
    caption = _global_caption(story, registry, start, end, token)
    shots = [tag_text(story['shots'][i]['prompt'], _mentions(registry, i, 'video_mention'), token)
             for i in range(start, end)]
    if family == 'holocine':
        return {**row, **holocine_inputs(caption, shots, counts)}
    if family != 'multishotmaster':
        raise ValueError(f'no ST-Bench native renderer input for {family}')
    boundaries = [0, *accumulate(counts)]
    return {**row, 'global_caption': caption, 'shot_captions': shots,
            'prompts': [f'Story: {caption} Now: {shot}' for shot in shots],
            'shot_groups': [[a, b] for a, b in zip(boundaries, boundaries[1:])], 'num_frames': sum(counts)}


def native_segments(story: dict, registry: dict | None, settings: dict, frames: int, count_tokens) -> list[dict]:
    family = settings['family']
    return pack_segments(len(story['shots']), lambda s, e: native_segment(story, registry, family, s, e, frames),
                         family=family, limits=settings['limits'], count_tokens=count_tokens)


def segment_shot_frames(segment: dict) -> list[int]:
    """各段实际出片的逐镜帧数；HoloCine 用官方取整后的切点。"""
    if 'shot_cut_frames' in segment:
        bounds = [0, *segment['shot_cut_frames'], segment['num_frames']]
        return [b - a for a, b in zip(bounds, bounds[1:])]
    return list(segment['frame_counts'])


def keyframe_source(story: dict, registry: dict, fps: int, frames: int) -> dict:
    """与 keyframe_video.prepare_keyframe_story 同形：开镜文字用 ST-Bench 自带的首帧提示，I2V 用逐镜原文。"""
    token = lambda k: f'[C{k}]'
    characters = {token(i): character_text(row) for i, row in enumerate(registry['characters'], 1)}
    shots = []
    for i, shot in enumerate(story['shots']):
        tagged = tag_text(shot['first_frame_prompt'], _mentions(registry, i, 'first_frame_mention'), token)
        expanded = tagged
        for tag, description in characters.items():
            expanded = expanded.replace(tag, description)
        shots.append({'tagged': tagged, 'expanded': expanded,
                      'characters': [token(int(e['tag'].split()[-1])) for e in registry['shots'][i]['present']],
                      'video_prompt': shot['prompt'], 'frames': frames})
    return {'story_id': story['story_id'], 'split': 'test', 'fps': fps, 'characters': characters,
            'identity_sources': {token(i): {'name': row['tag'], 'outfit': ''}
                                 for i, row in enumerate(registry['characters'], 1)},
            'shots': shots}


def storymem_shot_frames(story: dict, frames: int) -> list[int]:
    """官方 MI2V 续接镜以上一镜末帧为首帧，生成后去掉这一帧（pipeline.py: video[:, 1:]）。"""
    return [frames if index == 0 or shot['cut'] else frames - 1 for index, shot in enumerate(story['shots'])]


def storymem_shot_files(story: dict, directory: Path) -> list[Path]:
    """官方按「场景号_场景内镜号」命名逐镜文件。"""
    files, numbers = [], {}
    for shot in story['shots']:
        numbers[shot['scene']] = numbers.get(shot['scene'], 0) + 1
        files.append(directory / f"{shot['scene']:02d}_{numbers[shot['scene']]:02d}.mp4")
    return files


# ---------------------------------------------------------------- 配置与登记


def load_config(path: Path) -> dict:
    config = json.loads(os.path.expandvars(path.read_text()))
    for method, entry in config['methods'].items():
        entry['settings'] = json.loads(os.path.expandvars(Path(entry['settings']).read_text()))
        if '${' in json.dumps(entry):
            raise ValueError(f'{method}: unresolved environment variable')
    return config


def registry_path(output: Path, story_id: str) -> Path:
    return output / 'inputs' / 'registry' / f'{story_id}.json'


def build_registry(args) -> None:
    from mspc.data.openrouter_client import OpenRouterClient, OpenRouterCostCounter, load_openrouter_api_key
    from mspc.evaluation.st_bench import load_config as load_bench, load_stories

    config = load_config(args.config)
    stories = load_stories(load_bench(Path(config['benchmark_config'])))
    llm = config['registry']
    calls = args.output / 'inputs' / 'registry-calls.jsonl'
    calls.parent.mkdir(parents=True, exist_ok=True)
    spent = sum(Decimal(json.loads(line)['cost_usd']) for line in calls.read_text().splitlines()) if calls.exists() else Decimal(0)
    counter = OpenRouterCostCounter(hard_cap_usd=args.budget_usd, prices={llm['model']: llm['prices']},
                                    already_spent_usd=str(spent), precall_prompt_tokens=llm['precall_prompt_tokens'],
                                    precall_completion_tokens=llm['max_tokens'])
    client = OpenRouterClient(api_key=load_openrouter_api_key(args.api_key_file), base_url='https://openrouter.ai/api/v1',
                              cost_counter=counter, timeout_seconds=300, proxy=args.proxy)
    for story in stories:
        target = registry_path(args.output, story['story_id'])
        if target.exists():
            previous = json.loads(target.read_text())
            if previous['instructions'] == REGISTRY_INSTRUCTIONS:
                # 同一指令下的已有回答用现行校验重判；过了就不重抽，没过才重新请求。
                judged = {**previous, **judge_registry(story, previous['text'])}
                if judged['status'] == 'valid':
                    if judged != previous:
                        write_json(target, judged)
                    continue
        response = client.complete_messages(model=llm['model'], messages=[
            {'role': 'system', 'content': REGISTRY_INSTRUCTIONS}, {'role': 'user', 'content': registry_user_text(story)}],
            temperature=llm['temperature'], max_tokens=llm['max_tokens'],
            request_options=llm['request_options'], response_format=registry_schema())
        price = llm['prices']
        cost = (Decimal(response.usage['prompt_tokens']) * Decimal(price['input_usd_per_mtok'])
                + Decimal(response.usage['completion_tokens']) * Decimal(price['output_usd_per_mtok'])) / 1_000_000
        reported = response.served.get('reported_cost_usd')
        if reported is not None:
            cost = max(cost, Decimal(str(reported)))
        cost = cost.quantize(Decimal('0.000001'), ROUND_CEILING)
        with calls.open('a') as handle:
            handle.write(json.dumps({'story_id': story['story_id'], 'cost_usd': str(cost), 'usage': response.usage}) + '\n')
        record = {'story_id': story['story_id'], 'model': llm['model'], 'instructions': REGISTRY_INSTRUCTIONS,
                  'text': response.text, 'cost_usd': str(cost), **judge_registry(story, response.text)}
        write_json(target, record)
        print(json.dumps({'story_id': story['story_id'], 'status': record['status'], 'cost_usd': str(cost)}), flush=True)


def judge_registry(story: dict, text: str) -> dict:
    try:
        return {'status': 'valid', 'registry': validate_registry(story, json.loads(text)), 'error': None}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {'status': 'invalid', 'registry': None, 'error': f'{type(error).__name__}: {error}'}


def load_registry(output: Path, story: dict) -> dict:
    record = json.loads(registry_path(output, story['story_id']).read_text())
    if record['status'] != 'valid':
        raise ValueError(f"{story['story_id']}: character registry is {record['status']}")
    return validate_registry(story, record['registry'])


# ---------------------------------------------------------------- 出片


def video_directory(output: Path, method: str, story_id: str) -> Path:
    return output / 'inference' / 'raw' / 'videos' / method / story_id


def count_frames(path: Path, ffprobe: str) -> int:
    result = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                             '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    return int(json.loads(result.stdout)['streams'][0]['nb_read_frames'])


def render_wan_shots(runtime, segments: list[dict], directory: Path, seed: int, stride: int) -> list[dict]:
    """无参考的原生 Wan：与 segment_story.generate_story 无历史时同一调用。"""
    from mspc.inference.continuity_dev_32 import wan22_shot_plan_from_frame_counts

    units = []
    for ordinal, segment in enumerate(segments):
        label = f'g{ordinal + 1}'
        plan = wan22_shot_plan_from_frame_counts(segment['frame_counts'], label=label)
        path = directory / 'segments' / f'{label}.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        runtime.generate_case(prompts=segment['prompts'], shot_groups=plan['shot_groups'],
                              latent_shot_groups=plan['latent_shot_groups'], shot_frame_counts=segment['frame_counts'],
                              output_path=path, seed=seed + ordinal * stride, memory=None, capture_history=False)
        units.append({**segment, 'seed': seed + ordinal * stride, 'output': str(path),
                      'seconds': time.monotonic() - started})
    return units


def render_storymem_official(settings: dict, story: dict, directory: Path, seed: int) -> None:
    """官方 pipeline.py 原样跑两遍（--t2v_first_shot 出首镜后返回，再出其余镜），只注入本地 CLIP 与 HPS 权重。"""
    upstream = str(Path(settings['upstream']).resolve())
    sys.path.insert(0, upstream)
    import clip
    import extract_keyframes
    import pipeline
    import torch
    from hpsv3 import HPSv3RewardInferencer

    weights = settings['weights']
    model, _ = clip.load(weights['clip'], device='cpu', jit=False)
    extract_keyframes._CLIPCtx.model = model.eval().requires_grad_(False)
    extract_keyframes._CLIPCtx.device, extract_keyframes._CLIPCtx.dtype = 'cpu', torch.float32
    quality = []
    extract_keyframes.HPSv3RewardInferencer = lambda device: quality[0]
    original = pipeline.save_keyframes

    def save_keyframes(path):  # 与 renderer.StoryMemRuntime 相同：同一个 HPS 只在选帧时上卡。
        if quality:
            quality[0].model.to('cuda')
        else:
            quality.append(HPSv3RewardInferencer(config_path=weights['hps_config'], checkpoint_path=weights['hps'],
                                                 device='cuda'))
        try:
            original(path)
        finally:
            quality[0].model.to('cpu')
            torch.cuda.empty_cache()

    pipeline.save_keyframes = save_keyframes
    if Path(settings['weights']['low_noise_lora']).parent != Path(settings['weights']['high_noise_lora']).parent:
        raise ValueError('official pipeline reads both LoRA files from one directory')
    official = directory / 'official'
    sample, memory = settings['sampling'], settings['memory']
    argv = ['pipeline.py', '--story_script_path', story['source_file'],
            '--t2v_model_path', weights['t2v'], '--i2v_model_path', weights['m2v'],
            '--lora_weight_path', str(Path(weights['low_noise_lora']).parent), '--lora_rank', str(settings['lora_rank']),
            '--size', f"{sample['width']}*{sample['height']}", '--max_memory_size', str(memory['max_frames']),
            '--fix', str(memory['fixed_first_frames']), '--output_dir', str(official),
            '--log_file', str(directory / 'official.log'), '--seed', str(seed), '--offload_model', '--mi2v',
            '--t5_cpu', '--convert_model_dtype']
    for extra in (['--t2v_first_shot'], []):
        sys.argv = argv + extra
        pipeline.main(pipeline._parse_args())


def render(args) -> None:
    from mspc.evaluation.benchmark import inspect_video
    from mspc.evaluation.st_bench import load_config as load_bench, load_stories

    config = load_config(args.config)
    bench = load_bench(Path(config['benchmark_config']))
    stories = {story['story_id']: story for story in load_stories(bench)}
    entry = config['methods'][args.method]
    settings, frames, fps, seed = entry['settings'], bench['shot_frames'], bench['fps'], config['seed']
    selected = [stories[story_id] for story_id in args.story_ids]
    if args.method in KEYFRAME_METHODS:
        from mspc.comparison_methods.keyframe_story import run_stages, story_config

        env = dict(os.environ, PYTHONPATH=':'.join([*entry['python_path'], os.environ['PYTHONPATH']]))
        template = {key: value for key, value in settings.items()
                    if key not in ('story_id', 'given_plan_source', 'raw_request_source')}
        template.update(seed=seed, output_root=str(args.output), video_method=args.method)
        for story in selected:
            source = keyframe_source(story, load_registry(args.output, story), fps, frames)
            path = args.output / 'inputs' / 'keyframe-sources' / f"{story['story_id']}.json"
            if path.exists() and json.loads(path.read_text()) != source:
                raise FileExistsError(f'refusing to change the frozen keyframe input {path}')
            write_json(path, source)
            run_stages(args.output, args.method,
                       {**story_config(template, story['story_id'], source['characters']), 'st_bench_source': str(path)}, env)
        return
    family = settings['family']
    if family == 'storymem':
        for story in selected:
            directory = video_directory(args.output, args.method, story['story_id'])
            record_path = directory / 'generation.json'
            if record_path.exists():
                continue
            write_json(record_path, {'status': 'generation_started', 'source': story['source_file']})
            started = time.monotonic()
            render_storymem_official(settings, story, directory, seed)
            files = storymem_shot_files(story, directory / 'official')
            counts = [count_frames(path, settings['ffprobe']) for path in files]
            if counts != storymem_shot_frames(story, frames):
                raise ValueError(f'official StoryMem shot frames {counts} differ from the cut-flag rule')
            from mspc.core.video import concat_shot_videos
            concat_shot_videos(files, output_path=directory / 'video.mp4', ffmpeg=settings['ffmpeg'])
            write_json(record_path, {'status': 'completed', 'shot_files': [str(p) for p in files],
                                     'shot_frame_counts': counts, 'elapsed_seconds': time.monotonic() - started,
                                     'media': inspect_video(directory / 'video.mp4', settings['ffprobe'])})
        return
    if family not in RENDERER_FAMILIES:
        raise ValueError(f'{args.method}: no ST-Bench route for family {family}')
    if family == 'wan_single' and (settings['model']['memory_prefix'] is not None or settings['communication'] != 'normal'):
        raise ValueError('the ST-Bench Wan row is the native per-shot model without references')
    from mspc.comparison_methods.renderer import load_backend, render_story, select_runtime

    count_tokens, provenance = select_runtime(settings)
    pending = []
    for story in selected:
        directory = video_directory(args.output, args.method, story['story_id'])
        if (directory / 'generation.json').exists():
            continue
        registry = None if family == 'wan_single' else load_registry(args.output, story)
        segments = native_segments(story, registry, settings, frames, count_tokens)
        write_json(directory / 'inputs.json', {'story_id': story['story_id'], 'settings': settings,
                                               'segments': segments, 'runtime': provenance})
        pending.append((story, directory, segments))
    if not pending:
        return
    runtime, embed = load_backend(settings, fps)
    for story, directory, segments in pending:
        record_path = directory / 'generation.json'
        write_json(record_path, {'status': 'generation_started'})
        started = time.monotonic()
        task = {'video_path': str(directory / 'video.mp4'), 'seed': seed}
        if family == 'wan_single':
            units = render_wan_shots(runtime, segments, directory, seed, settings['seed_stride'])
            from mspc.core.video import concat_shot_videos
            concat_shot_videos([Path(u['output']) for u in units], output_path=directory / 'video.mp4', ffmpeg=settings['ffmpeg'])
        else:
            units = render_story(settings, task, None, segments, (runtime, embed), count_tokens, fps, [])
        counts = [count for segment in segments for count in segment_shot_frames(segment)]
        write_json(record_path, {'status': 'completed', 'segments': units, 'shot_frame_counts': counts,
                                 'elapsed_seconds': time.monotonic() - started,
                                 'media': inspect_video(directory / 'video.mp4', settings['ffprobe'])})
        print(json.dumps({'method': args.method, 'story_id': story['story_id'], 'status': 'completed'}), flush=True)


def score_manifest(args) -> None:
    """把各方法生成记录整理成 evaluation.st_bench score 的清单；未完成的记 video_path=null。"""
    from mspc.evaluation.st_bench import load_config as load_bench, load_stories

    config = load_config(args.config)
    stories = load_stories(load_bench(Path(config['benchmark_config'])))
    rows = []
    for story in stories:
        directory = video_directory(args.output, args.method, story['story_id'])
        record_path = directory / 'generation.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {'status': 'pending'}
        if record['status'] != 'completed':
            rows.append({'story_id': story['story_id'], 'video_path': None, 'shot_frame_counts': None,
                         'generation_status': record['status']})
            continue
        if args.method in KEYFRAME_METHODS:  # 关键帧路线由 keyframe_video.assemble_videos 写逐镜记录
            counts = [shot['num_frames'] for shot in json.loads((directory / 'segments.json').read_text())]
        else:
            counts = record['shot_frame_counts']
        rows.append({'story_id': story['story_id'], 'video_path': str(directory / 'video.mp4'),
                     'shot_frame_counts': counts, 'generation_status': 'completed'})
    write_json(args.manifest, {'method': args.method, 'rows': rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    registry = commands.add_parser('registry', help='one LLM conversion per story into a frozen character registry')
    registry.add_argument('--config', type=Path, required=True)
    registry.add_argument('--output', type=Path, required=True)
    registry.add_argument('--budget-usd', required=True)
    registry.add_argument('--api-key-file', type=Path, required=True)
    registry.add_argument('--proxy')
    rendering = commands.add_parser('render', help='render the listed stories with one method')
    rendering.add_argument('--config', type=Path, required=True)
    rendering.add_argument('--output', type=Path, required=True)
    rendering.add_argument('--method', required=True)
    rendering.add_argument('--story-ids', nargs='+', required=True)
    manifest = commands.add_parser('manifest', help='write the scoring manifest for one method')
    manifest.add_argument('--config', type=Path, required=True)
    manifest.add_argument('--output', type=Path, required=True)
    manifest.add_argument('--method', required=True)
    manifest.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    {'registry': build_registry, 'render': render, 'manifest': score_manifest}[args.command](args)


if __name__ == '__main__':
    main()
