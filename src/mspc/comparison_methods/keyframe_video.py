"""自产关键帧到普通Wan I2V的唯一执行与拼接入口。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import time

from mspc.data.shot_text import opening_frame_text
from mspc.comparison_methods.prompts import segment_inputs


def prepare_keyframe_story(story: dict) -> dict:
    """同一给定计划的开镜事实；角色bank的原生条件由各图像方法消费。"""
    board = story['storyboard']
    registry = {c['name']: c['appearance'] for c in board['characters']}
    present = [{p['name'] for p in shot['present']} & registry.keys() for shot in board['shots']]
    used = set().union(*present)
    names = [name for name in registry if name in used]
    tags = {name: f'[C{index+1}]' for index, name in enumerate(names)}
    descriptions = {tags[name]: registry[name] for name in names}
    first = {name: next(p for shot in board['shots'] for p in shot['present'] if p['name'] == name) for name in names}
    outfits = {entry['name']: entry.get('outfit', '') for entry in board['characters']}
    sources = {tags[name]: {'name': name, 'outfit': first[name].get('outfit', outfits[name])} for name in names}
    prompts = []
    for i in range(len(board['shots'])):
        text = opening_frame_text(board, i)
        for name in names:
            text = text.replace(name + ', ' + registry[name], tags[name])
        text = re.sub(r'Character [1-9]\d*', lambda m: tags[m.group()], text)
        expanded = text
        for tag, description in descriptions.items():
            expanded = expanded.replace(tag, description + ' ')
        prompts.append({'tagged': text, 'expanded': expanded,
                        'characters': [tags[name] for name in names if name in present[i]],
                        'video_prompt': segment_inputs(board, 'storymem', i, i+1)['prompts'][0],
                        'frames': board['shots'][i]['frame_count']})
    return {'story_id': story['story_id'], 'split': story['split'], 'fps': story['fps'],
            'characters': descriptions, 'identity_sources': sources, 'shots': prompts}


KEYFRAME_SOURCES = ('given_plan_source', 'st_bench_source')


def keyframe_source(config: dict) -> tuple[str, dict]:
    """开镜输入恰有一个来源：共同给定计划，或 comparison_methods.st_bench 已冻结的 ST-Bench 原生输入。"""
    keys = [key for key in KEYFRAME_SOURCES if key in config]
    if len(keys) != 1:
        raise ValueError(f'keyframe config requires exactly one of {KEYFRAME_SOURCES}')
    key, = keys
    if key == 'st_bench_source':
        source = read_json(config[key])
        if source['story_id'] != config['story_id']:
            raise ValueError('frozen ST-Bench input story differs from configuration')
        return key, source
    story = next(s for s in read_json(config[key])['stories'] if s['story_id'] == config['story_id'])
    return key, prepare_keyframe_story(story)


def encoder_text_audit(tokenizer, text: str, length: int) -> dict:
    """记录原生encoder实际保留的文本，不在调用前人工截字或拒收故事。"""
    raw = tokenizer(text, truncation=False)['input_ids']
    actual = tokenizer(text, padding='max_length', max_length=length,
                       truncation=True)['input_ids']
    removed = raw[length-1:-1] if len(raw) > length else []
    return {'raw_token_count': len(raw), 'raw_token_ids': raw,
            'encoded_length': len(actual), 'encoded_token_ids': actual,
            'truncated': bool(removed),
            'retained_text': tokenizer.decode(actual, skip_special_tokens=True),
            'removed_text': tokenizer.decode(removed, skip_special_tokens=True)}


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    from mspc.config.shared_root import expand_shared_root_token
    return json.loads(expand_shared_root_token(Path(path).read_text(),str(path)))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def wan_output_size(max_area: int, image_size: tuple[int, int], vae_stride, patch_size) -> tuple[int, int]:
    """固定上游的面积量化公式，用于在加载模型前核真实图片尺寸。"""
    import numpy as np
    width, height = image_size
    aspect_ratio = height / width
    latent_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    latent_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    return latent_w * vae_stride[2], latent_h * vae_stride[1]


def generate_videos(config: dict, prepared: dict, shot_indices: list[int]):
    if not shot_indices or len(set(shot_indices)) != len(shot_indices):
        raise ValueError('video worker requires a nonempty set of distinct shot indices')
    if any(type(i) is not int or not 0 <= i < len(prepared['shots']) for i in shot_indices):
        raise ValueError('video worker shot index outside the frozen story')
    import sys
    import torch
    from PIL import Image
    upstream = Path(config['wan_i2v']['upstream'])
    sys.path.insert(0, str(upstream))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video
    sample = config['wan_i2v']
    output = Path(config['output_root'])/'inference/raw/videos'/config['video_method']/prepared['story_id']
    if (output/'generation.json').exists():
        raise FileExistsError('video generation already recorded; inspect and reuse it')
    native_config = WAN_CONFIGS['i2v-A14B']
    for i in shot_indices:
        path = Path(config['output_root'])/'inference/raw/images'/config['image_method']/prepared['story_id']/f'{i+1:04d}.png'
        with Image.open(path) as image:
            size = wan_output_size(sample['max_area'], image.size, native_config.vae_stride, native_config.patch_size)
        if size != (sample['width'], sample['height']):
            raise ValueError(f'shot {i+1} native area quantization produces {size}, not the configured resolution')
    started=time.monotonic()
    model=wan.WanI2V(config=deepcopy(native_config),checkpoint_dir=sample['weights'],
                     device_id=0,rank=0,t5_fsdp=False,dit_fsdp=False,use_sp=False,t5_cpu=True,convert_model_dtype=True)
    load_seconds=time.monotonic()-started
    torch.cuda.reset_peak_memory_stats()
    segments=[]
    for i in shot_indices:
        row = prepared['shots'][i]
        image_path=Path(config['output_root'])/'inference/raw/images'/config['image_method']/prepared['story_id']/f'{i+1:04d}.png'
        path=output/'segments'/f'{i+1:04d}.mp4'
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        began=time.monotonic()
        video=model.generate(row['video_prompt'],Image.open(image_path).convert('RGB'),
                   max_area=sample['max_area'],frame_num=row['frames'],shift=sample['shift'],
                   sample_solver='unipc',sampling_steps=sample['steps'],guide_scale=tuple(sample['cfg']),
                   n_prompt=sample['negative_prompt'],seed=config['seed']+i*1009,offload_model=True)
        if tuple(video.shape[1:])!=(row['frames'],sample['height'],sample['width']):
            raise ValueError(f'unexpected I2V output shape {tuple(video.shape)}')
        save_video(tensor=video[None],save_file=str(path),fps=prepared['fps'],nrow=1,normalize=True,value_range=(-1,1))
        del video
        record = {'shot':i+1,'num_frames':row['frames'],'seconds':time.monotonic()-began,
                  'seed':config['seed']+i*1009,'output':str(path),'first_frame':str(image_path)}
        segments.append(record)
        write_json(path.with_suffix('.json'),record)
    worker_name = 'shots-' + '-'.join(str(i+1) for i in shot_indices)
    write_json(output/'workers'/f'{worker_name}.json',{'status':'completed',
               'shot_indices':shot_indices,'load_seconds':load_seconds,
               'elapsed_seconds':time.monotonic()-started,'shots':segments,
               'peak_allocated_bytes':torch.cuda.max_memory_allocated()})


def assemble_videos(config: dict, prepared: dict):
    """各镜独立完成后只拼接一次；原镜号、seed与帧数不因并行分工变化。"""
    from mspc.evaluation.benchmark import inspect_video
    from mspc.core.video import concat_shot_videos
    output = Path(config['output_root'])/'inference/raw/videos'/config['video_method']/prepared['story_id']
    if (output/'generation.json').exists():
        raise FileExistsError('video generation already recorded; inspect and reuse it')
    segments = []
    for i,row in enumerate(prepared['shots']):
        path = output/'segments'/f'{i+1:04d}.mp4'
        record = read_json(path.with_suffix('.json'))
        if (record['shot'],record['num_frames'],record['seed']) != (i+1,row['frames'],config['seed']+i*1009):
            raise ValueError(f'shot {i+1} metadata differs from the frozen input')
        if Path(record['output']) != path or not path.is_file():
            raise ValueError(f'shot {i+1} output is missing or points elsewhere')
        segments.append(record)
    started = time.monotonic()
    concat_shot_videos([Path(s['output']) for s in segments],output_path=output/'video.mp4',ffmpeg=config['ffmpeg'])
    write_json(output/'segments.json',segments)
    write_json(output/'generation.json',{'status':'completed','assembly_seconds':time.monotonic()-started,
               'shot_generation_seconds':sum(s['seconds'] for s in segments),
               'workers':[read_json(p) for p in sorted((output/'workers').glob('*.json'))],
               'media':inspect_video(output/'video.mp4',config['ffprobe']),'shots':len(segments)})


