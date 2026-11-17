"""固定官方 CSA 的无界面生图；视频交给共享普通Wan I2V入口。"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import time

from mspc.comparison_methods.keyframe_video import (
    assemble_videos, encoder_text_audit, generate_videos, keyframe_source, load_config,
    read_json, write_json,
)


def prepare_story(prepared: dict) -> dict:
    if not prepared['characters']:
        raise ValueError('identity-bank generation requires at least one used character')
    refs = {}
    identities = []
    prompt_characters = [[tag for tag in prepared['characters'] if tag in row['tagged']]
                         for row in prepared['shots']]
    for tag in prepared['characters']:
        solo = [i for i, tags in enumerate(prompt_characters) if tags == [tag]]
        refs[tag] = solo[:1]
        if solo:
            identities.append({'character': tag, 'story_shot_index': solo[0],
                               'prompt': prepared['shots'][solo[0]]['expanded'],
                               'source': 'given_story_solo_shot'})
        else:
            source = prepared['identity_sources'][tag]
            prompt = 'Portrait of ' + prepared['characters'][tag].rstrip('.') + '.'
            if source['outfit']:
                outfit = re.sub(r'^(?:wearing|wears) ', '', source['outfit'], flags=re.IGNORECASE)
                prompt += ' The character is wearing ' + outfit.rstrip('.') + '.'
            identities.append({'character': tag, 'story_shot_index': None,
                               'prompt': prompt, 'source': 'given_character_description',
                               'source_character': source['name']})
    return {**prepared, 'reference_indices': refs, 'identity_references': identities,
            'prompt_characters': prompt_characters}


def token_audit(prepared: dict, sdxl_root: Path) -> dict:
    from transformers import CLIPTokenizer
    audits = []
    for folder in ['tokenizer', 'tokenizer_2']:
        tokenizer = CLIPTokenizer.from_pretrained(sdxl_root / folder, local_files_only=True)
        audits.append([encoder_text_audit(tokenizer, row['expanded'], tokenizer.model_max_length)
                       for row in prepared['shots']])
    identity_audits = []
    for folder in ['tokenizer', 'tokenizer_2']:
        tokenizer = CLIPTokenizer.from_pretrained(sdxl_root / folder, local_files_only=True)
        identity_audits.append([encoder_text_audit(tokenizer, row['prompt'], tokenizer.model_max_length)
                                for row in prepared['identity_references']])
    return {'clip_token_counts': [[row['raw_token_count'] for row in series] for series in audits],
            'clip_limit': 77, 'encoders': audits,
            'identity_encoders': identity_audits,
            'truncation_behavior': 'official SDXL tokenizer truncation; full prompt passed unchanged'}


def _official_attention(upstream: Path, torch, sample: dict, prepared: dict):
    """只执行固定上游的 attention 定义；避开 UI 顶层自动下载、加载与监听。"""
    import random
    import copy
    env = {'torch': torch, 'F': torch.nn.functional, 'random': random, 'copy': copy,
           'device': 'cuda', 'height': sample['height'], 'width': sample['width'],
           'sa32': sample['sa32'], 'sa64': sample['sa64'], 'id_length': 1,
           'total_length': 2, 'cur_step': 0, 'attn_count': 0, 'total_count': 0,
           'write': True, 'cur_character': [], 'character_dict': prepared['characters'],
           'ref_indexs_dict': prepared['reference_indices'], 'ref_totals': [],
           'character_index_dict': {}, 'invert_character_index_dict': {}}
    needed = {'AttnProcessor2_0', 'cal_attn_indice_xl_effcient_memory'}
    helper = ast.parse((upstream / 'utils/gradio_utils.py').read_text())
    exec(compile(ast.Module([n for n in helper.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in needed], []), str(upstream / 'utils/gradio_utils.py'), 'exec'), env)
    env['AttnProcessor'] = env['AttnProcessor2_0']
    main = ast.parse((upstream / 'gradio_app_sdxl_specific_id_low_vram.py').read_text())
    needed = {'SpatialAttnProcessor2_0', 'set_attention_processor'}
    exec(compile(ast.Module([n for n in main.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in needed], []), str(upstream / 'gradio_app_sdxl_specific_id_low_vram.py'), 'exec'), env)
    return env


def generate_images(config: dict, prepared: dict):
    import random
    import numpy as np
    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline
    sample = config['storydiffusion']
    output = Path(config['output_root']) / 'inference/raw/images/storydiffusion' / prepared['story_id']
    if (output / 'generation.json').exists():
        raise FileExistsError('image generation already recorded; inspect and reuse it')
    audit = token_audit(prepared, Path(sample['weights']))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    env = _official_attention(Path(sample['upstream']), torch, sample, prepared)
    pipe = StableDiffusionXLPipeline.from_pretrained(sample['weights'], variant='fp16', torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    env['set_attention_processor'](pipe.unet, 1)
    env['total_count'] = sum(isinstance(p, env['SpatialAttnProcessor2_0']) for p in pipe.unet.attn_processors.values())
    load_seconds = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    timings = []
    calls = []
    written_shots = set()
    for identity in prepared['identity_references']:
        i = identity['story_shot_index']
        target = output / f'{i+1:04d}.png' if i is not None else output / 'identity-bank' / (identity['character'].strip('[]') + '.png')
        calls.append({'prompt': identity['prompt'], 'characters': [identity['character']],
                      'write': True, 'shot_index': i, 'output': target,
                      'source': identity['source']})
        if i is not None:
            written_shots.add(i)
    for i, row in enumerate(prepared['shots']):
        if i not in written_shots:
            calls.append({'prompt': row['expanded'], 'characters': prepared['prompt_characters'][i],
                          'write': False, 'shot_index': i, 'output': output / f'{i+1:04d}.png',
                          'source': 'given_story_shot'})
    for call in calls:
        env['write'] = call['write']
        env['cur_character'] = call['characters']
        env['cur_step'] = env['attn_count'] = 0
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        began = time.monotonic()
        image = pipe(call['prompt'], width=sample['width'], height=sample['height'],
                     guidance_scale=sample['cfg'], num_inference_steps=sample['steps'],
                     negative_prompt=sample['negative_prompt'],
                     generator=torch.Generator(device='cuda').manual_seed(config['seed'])).images[0]
        call['output'].parent.mkdir(parents=True, exist_ok=True)
        image.save(call['output'])
        timings.append({'shot': call['shot_index'] + 1 if call['shot_index'] is not None else None,
                        'seconds': time.monotonic()-began, 'identity_bank_write': env['write'],
                        'characters': call['characters'], 'prompt': call['prompt'],
                        'source': call['source'], 'output': str(call['output'])})
        write_json(output/'progress.json', {'completed':len(timings),'expected':len(calls),'calls':timings})
    write_json(output/'generation.json', {'status':'completed','load_seconds':load_seconds,
               'elapsed_seconds':time.monotonic()-started,'shots':timings,'token_audit':audit,
               'scheduler':type(pipe.scheduler).__name__,
               'identity_references':prepared['identity_references'],
               'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'upstream_commit':read_json(Path(sample['upstream'])/'SOURCE.json')['commit']})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','images','videos','assemble'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--shot-indices',type=int,nargs='+',help='0-based original shot indices assigned to this video worker')
    args=parser.parse_args()
    config=load_config(args.config)
    _,source=keyframe_source(config)
    prepared=prepare_story(source)
    target=Path(config['output_root'])/'inputs/prepared'/f"{config['story_id']}.json"
    if args.command=='prepare':
        prepared['token_audit']=token_audit(prepared,Path(config['storydiffusion']['weights']))
        write_json(target,prepared)
        print(json.dumps({'story_id':prepared['story_id'],'shots':len(prepared['shots']),'audit':prepared['token_audit']}))
    elif args.command=='images': generate_images(config,prepared)
    elif args.command=='videos':
        if args.shot_indices is None:parser.error('videos requires explicit --shot-indices')
        generate_videos(config,prepared,args.shot_indices)
    else: assemble_videos(config,prepared)


if __name__=='__main__':main()
