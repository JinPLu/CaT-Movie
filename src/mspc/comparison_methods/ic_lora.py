"""官方film-storyboard LoRA：三格联合生成，历史自产格经原生SDEdit传递。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time

from mspc.comparison_methods.keyframe_video import (
    assemble_videos, encoder_text_audit, generate_videos, keyframe_source, load_config,
    read_json, write_json,
)


def three_panel_windows(shot_count: int) -> list[dict]:
    """首次三格，之后沿时间前进；末窗多保留一格，不补虚构镜或重画候选。"""
    if type(shot_count) is not int or shot_count < 3:
        raise ValueError('three-panel storyboard requires at least three shots')
    windows = [{'shot_indices': [0, 1, 2], 'preserved_indices': [],
                'generated_indices': [0, 1, 2]}]
    covered = 3
    while covered < shot_count:
        start = min(covered - 1, shot_count - 3)
        indices = list(range(start, start + 3))
        windows.append({'shot_indices': indices,
                        'preserved_indices': [i for i in indices if i < covered],
                        'generated_indices': [i for i in indices if i >= covered]})
        covered = indices[-1] + 1
    return windows


def prepare_story(source: dict, sample: dict) -> dict:
    if sample['width'] % 16 or sample['panel_height'] % 16:
        raise ValueError('fixed panel boundaries must align to FLUX packed latents')
    names = sample['character_names']
    if set(names) != set(source['characters']) or len(set(names.values())) != len(names):
        raise ValueError('each frozen character requires one distinct explicit name')
    global_prompt = '[MOVIE-SHOTS] Three vertically stacked movie shots. ' + ' '.join(
        f'<{names[tag]}> is {description}.' for tag, description in source['characters'].items())
    windows = three_panel_windows(len(source['shots']))
    for window in windows:
        scenes = []
        for panel, index in enumerate(window['shot_indices']):
            text = source['shots'][index]['tagged']
            for tag, name in names.items():
                text = text.replace(tag, f'<{name}>')
            scenes.append(f'[SCENE-{panel + 1}] {text}')
        caption = global_prompt + ' ' + ' '.join(scenes)
        window.update(prompt=caption, prompt_2=caption)
    return {'story_id': source['story_id'], 'split': source['split'], 'fps': source['fps'],
            'characters': deepcopy(source['characters']), 'character_names': names,
            'shots': deepcopy(source['shots']), 'windows': windows,
            'panel_size': [sample['width'], sample['panel_height']],
            'canvas_size': [sample['width'], sample['panel_height'] * 3]}


def token_audit(prepared: dict, weights: Path, max_sequence_length: int) -> dict:
    from transformers import AutoTokenizer
    clip = AutoTokenizer.from_pretrained(weights / 'tokenizer', local_files_only=True)
    t5 = AutoTokenizer.from_pretrained(weights / 'tokenizer_2', local_files_only=True)
    clip_audits = [encoder_text_audit(clip, w['prompt'], clip.model_max_length)
                   for w in prepared['windows']]
    t5_counts = [len(t5(w['prompt_2'], truncation=False)['input_ids']) for w in prepared['windows']]
    return {'clip_token_counts': [row['raw_token_count'] for row in clip_audits],
            'clip_limit': 77, 'clip_pooled_inputs': clip_audits,
            't5_token_counts': t5_counts,
            't5_encoded_lengths': [max(max_sequence_length, count) for count in t5_counts],
            't5_minimum_padding': max_sequence_length, 't5_truncation': False}


def encode_window(pipe, window: dict, encoded_length: int, device):
    """使用原生公共编码入口保留完整T5；短输入保持原512 padding。"""
    import torch
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=window['prompt'], prompt_2=window['prompt_2'],
            device=device, max_sequence_length=encoded_length)
    if prompt_embeds.shape[1] != encoded_length:
        raise ValueError('FLUX text encoder changed the declared full T5 sequence length')
    return {'prompt_embeds': prompt_embeds, 'pooled_prompt_embeds': pooled_prompt_embeds}


def conditioning_canvas(window: dict, canonical_images: Path, panel_size: tuple[int, int]):
    from PIL import Image
    width, height = panel_size
    canvas = Image.new('RGB', (width, height * 3))
    mask = Image.new('L', canvas.size, 255)
    for panel, index in enumerate(window['shot_indices']):
        if index in window['preserved_indices']:
            with Image.open(canonical_images / f'{index+1:04d}.png') as image:
                if image.size != panel_size:
                    raise ValueError('previous canonical panel has incompatible dimensions')
                canvas.paste(image.convert('RGB'), (0, panel * height))
            mask.paste(0, (0, panel * height, width, (panel + 1) * height))
    return canvas, mask


def load_image_pipelines(sample: dict):
    """CPU和GPU阶段共用同一完整加载；逐张量检查官方LoRA，没有替代权重。"""
    import torch
    from diffusers import FluxInpaintPipeline, FluxPipeline
    from safetensors.torch import load_file
    pipe = FluxPipeline.from_pretrained(sample['weights'], torch_dtype=torch.bfloat16,
                                        use_safetensors=True, local_files_only=True)
    if pipe.transformer.config.in_channels != 64:
        raise ValueError('film-storyboard uses the original FLUX.1-dev model')
    pipe.load_lora_weights(str(Path(sample['lora']).parent),
                           weight_name=Path(sample['lora']).name, adapter_name='film_storyboard',
                           local_files_only=True)
    pipe.set_adapters('film_storyboard', adapter_weights=sample['lora_scale'])
    state = load_file(sample['lora'], device='cpu')
    parameters = dict(pipe.transformer.named_parameters())
    for key, value in state.items():
        target = key.removeprefix('transformer.').replace('.lora_A.', '.lora_A.film_storyboard.').replace('.lora_B.', '.lora_B.film_storyboard.')
        if target not in parameters or not torch.equal(parameters[target].detach().cpu(), value.to(parameters[target].dtype)):
            raise ValueError(f'official LoRA tensor was not loaded correctly: {key}')
    lora_tensor_count = len(state)
    del state, parameters
    inpaint = FluxInpaintPipeline(scheduler=pipe.scheduler, vae=pipe.vae,
                                 text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
                                 text_encoder_2=pipe.text_encoder_2, tokenizer_2=pipe.tokenizer_2,
                                 transformer=pipe.transformer)
    return pipe, inpaint, lora_tensor_count


def generate_images(config: dict, prepared: dict):
    import torch
    sample = config['ic_lora']
    output = Path(config['output_root']) / 'inference/raw/images' / config['image_method'] / prepared['story_id']
    if (output / 'generation.json').exists() or list(output.glob('*.png')):
        raise FileExistsError('image results already exist; inspect and reuse them')
    audit = token_audit(prepared, Path(sample['weights']), sample['max_sequence_length'])
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    pipe, inpaint, lora_tensor_count = load_image_pipelines(sample)
    pipe.to('cuda')
    load_seconds = time.monotonic() - started
    torch.cuda.reset_peak_memory_stats()
    width, height = prepared['panel_size']
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for number, window in enumerate(prepared['windows']):
        prefix = output / 'windows' / f'{number+1:04d}'
        if prefix.with_suffix('.png').exists():
            raise FileExistsError(prefix.with_suffix('.png'))
        seed = config['seed'] + number * 1009
        began = time.monotonic()
        encoded = encode_window(pipe, window, audit['t5_encoded_lengths'][number], 'cuda')
        args = dict(**encoded, width=width,
                    height=height * 3, num_inference_steps=sample['steps'],
                    guidance_scale=sample['guidance'],
                    generator=torch.Generator(device='cuda').manual_seed(seed))
        if window['preserved_indices']:
            canvas, mask = conditioning_canvas(window, output, (width, height))
            prefix.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(prefix.with_name(prefix.name + '-conditioning.png'))
            mask.save(prefix.with_name(prefix.name + '-mask.png'))
            image = inpaint(image=canvas, mask_image=mask, strength=sample['strength'], **args).images[0]
        else:
            image = pipe(**args).images[0]
        if image.size != tuple(prepared['canvas_size']):
            raise ValueError('FLUX returned a different canvas geometry')
        prefix.parent.mkdir(parents=True, exist_ok=True)
        image.save(prefix.with_suffix('.png'))
        panels = []
        for panel, index in enumerate(window['shot_indices']):
            if index not in window['generated_indices']:
                continue
            target = output / f'{index+1:04d}.png'
            if target.exists():
                raise FileExistsError(target)
            bounds = [0, panel * height, width, (panel + 1) * height]
            image.crop(tuple(bounds)).save(target)
            panels.append({'shot': index+1, 'panel': panel+1, 'bounds': bounds, 'output': str(target)})
        record = {**window, 'window': number+1, 'seed': seed,
                  'text_encoding': {'clip': audit['clip_pooled_inputs'][number],
                                    't5_raw_tokens': audit['t5_token_counts'][number],
                                    't5_encoded_length': audit['t5_encoded_lengths'][number]},
                  'seconds': time.monotonic()-began, 'canvas': str(prefix.with_suffix('.png')),
                  'canonical_panels': panels}
        records.append(record)
        write_json(prefix.with_suffix('.json'), record)
        elapsed = sum(r['seconds'] for r in records)
        write_json(output / 'progress.json', {'completed_windows': len(records), 'expected_windows': len(prepared['windows']),
                   'completed_shots': sum(len(r['canonical_panels']) for r in records),
                   'elapsed_seconds': time.monotonic()-started,
                   'estimated_remaining_image_seconds': elapsed / len(records) * (len(prepared['windows'])-len(records)),
                   'windows': records})
    write_json(output / 'generation.json', {'status': 'completed', 'started_at': started_at,
               'finished_at': datetime.now(timezone.utc).isoformat(), 'load_seconds': load_seconds,
               'elapsed_seconds': time.monotonic()-started, 'windows': records,
               'lora_tensor_count': lora_tensor_count, 'token_audit': audit,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'upstream_commit': read_json(Path(sample['upstream']) / 'SOURCE.json')['commit']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'validate-assets', 'images', 'videos', 'assemble'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--shot-indices', type=int, nargs='+')
    args = parser.parse_args()
    config = load_config(args.config)
    target = Path(config['output_root']) / 'inputs/prepared' / f"{config['story_id']}.json"
    if args.command == 'prepare':
        key, source = keyframe_source(config)
        prepared = prepare_story(source, config['ic_lora'])
        prepared[key] = config[key]
        prepared['token_audit'] = token_audit(prepared, Path(config['ic_lora']['weights']), config['ic_lora']['max_sequence_length'])
        if target.exists() and read_json(target) != prepared:
            raise FileExistsError('refusing to change the frozen input')
        write_json(target, prepared)
        print(prepared['story_id'], prepared['token_audit'])
    else:
        prepared = read_json(target)
        if prepared['story_id'] != config['story_id']:
            raise ValueError('frozen input story differs from configuration')
        if args.command == 'validate-assets':
            started = time.monotonic()
            pipe, inpaint, count = load_image_pipelines(config['ic_lora'])
            report = {'checked_at': datetime.now(timezone.utc).isoformat(),
                      'scope': 'CPU complete model and LoRA loading; no generated image',
                      'load_seconds': time.monotonic() - started, 'lora_tensor_count': count,
                      'transformer_in_channels': pipe.transformer.config.in_channels,
                      'inpaint_pipeline': type(inpaint).__name__,
                      'shared_components': all(getattr(inpaint, name) is getattr(pipe, name) for name in
                                               ('scheduler', 'vae', 'text_encoder', 'tokenizer',
                                                'text_encoder_2', 'tokenizer_2', 'transformer')),
                      'loaded_adapters': pipe.get_list_adapters(),
                      'scheduler': dict(pipe.scheduler.config),
                      'token_audit': prepared['token_audit']}
            write_json(Path(config['output_root']) / 'inputs/ic-lora-model-validation.json', report)
            print(report)
        elif args.command == 'images':
            generate_images(config, prepared)
        elif args.command == 'videos':
            if args.shot_indices is None:
                parser.error('videos requires explicit --shot-indices')
            generate_videos(config, prepared, args.shot_indices)
        else:
            assemble_videos(config, prepared)


if __name__ == '__main__':
    main()
