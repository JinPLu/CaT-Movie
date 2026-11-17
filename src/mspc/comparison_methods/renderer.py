"""统一故事出片消费者；一个进程只加载一种原生runtime，不裁掉HoloCine取整后的画面。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from mspc.benchmark.generation import plan_path
from mspc.evaluation.benchmark import inspect_video, load_manifest, write_json
from mspc.comparison_methods.prompts import OFFICIAL_WAN, adapt_storyboard, segment_inputs


# 前面各镜一张关键帧都没选出（记忆库为空）时，用前一镜末帧作唯一记忆帧照常走 M2V；镜记录 ``memory_fallback`` 记这个值。
MEMORY_FALLBACK = 'previous_shot_last_frame'


def fallback_frame(unit: dict) -> Path:
    """前一镜末帧的固定副本：上游每镜把末帧写到同一个 last_frame.jpg，下一镜出完就被覆盖。"""
    output = Path(unit['output'])
    return output.with_name(f'{output.stem}_last_frame.jpg')


def storymem_memory(units: list[dict], limits: dict) -> list[str]:
    """按镜头顺序消费已登记关键帧，保持原来的首部／尾部上限规则。

    一张关键帧都没有时退到前一镜末帧作唯一记忆帧（M2V 至少要一张记忆帧，这是放宽「至少一帧」的最小退法）。
    """
    memory = [path for unit in units for path in unit['keyframes']]
    if len(memory) != len(set(memory)):
        raise ValueError('duplicate StoryMem keyframe in completed prefix')
    if units and not memory:
        return [str(fallback_frame(units[-1]))]
    size, fixed = limits['max_frames'], limits['fixed_first_frames']
    if len(memory) > size:
        memory = memory[:fixed] + memory[-(size - fixed):]
    return memory


def select_runtime(settings: dict):
    root = Path(settings['upstream']).resolve()
    package = 'wan' if settings['family'] in ('storymem', *OFFICIAL_WAN) else 'diffsynth'
    if not (root / package).is_dir():
        raise FileNotFoundError(root / package)
    sys.path.insert(0, str(root))
    if settings['family'] in ('storymem', *OFFICIAL_WAN):
        # wan.__init__ 的 T5 默认参数会初始化 CUDA；CPU 容量检查只加载官方 tokenizer 文件。
        spec = importlib.util.spec_from_file_location('storymem_tokenizers', root / 'wan/modules/tokenizers.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tokenizer = module.HuggingfaceTokenizer(name=settings['weights']['tokenizer'], seq_len=None, clean='whitespace')
        counter = lambda text: int(tokenizer(text, return_mask=True, add_special_tokens=True)[1].sum())
    else:
        module = importlib.import_module(package)
        if root not in Path(module.__file__).resolve().parents:
            raise RuntimeError(f'expected {root}/{package}; got {module.__file__}; use a fresh process per method')
    if settings['family'] == 'holocine':
        from diffsynth.prompters.wan_prompter import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(name=settings['weights']['tokenizer'], seq_len=None, clean='whitespace')
        counter = lambda text: int(tokenizer(text, return_mask=True, add_special_tokens=True)['attention_mask'].sum())
    elif settings['family'] not in ('storymem', *OFFICIAL_WAN):
        from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(name=settings['weights']['tokenizer'], seq_len=None, clean='whitespace')
        counter = lambda text: int(tokenizer(text, return_mask=True, add_special_tokens=True)[1].sum())
    if (root / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    else:
        revision = settings['code_revision']
        if not isinstance(revision, str) or not revision:
            raise RuntimeError(f'{root} has no git metadata; set code_revision')
    return counter, {'upstream': str(root), 'revision': revision, package: str(module.__file__)}


class FixedNegativeTextEncoder:
    """只复用当前冻结 StoryMem encoder 的固定负提示词；正提示词仍走原 batch。"""

    def __init__(self, encoder, negative_prompt: str):
        self.encoder = encoder
        self.negative_prompt = negative_prompt
        self.context = None
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def __call__(self, texts, device):
        if texts != [self.negative_prompt]:
            return self.encoder(texts, device)
        if self.context is None:
            self.context = [value.detach().clone() for value in self.encoder(texts, device)]
            self.misses += 1
        else:
            self.hits += 1
        return [value.clone() for value in self.context]


def cache_pipeline_negative_prompt(pipe, family: str, negative_prompt: str) -> None:
    """固定模型生命周期内缓存原 batch 的负 context，不改变原文字编码几何。"""
    from types import MethodType
    units = [unit for unit in pipe.units if type(unit).__name__ == 'WanVideoUnit_PromptEmbedder']
    if len(units) != 1:
        raise ValueError('expected one native prompt embedder')
    unit, = units
    original = unit.process
    cache = {}
    unit.negative_prompt_cache_hits = unit.negative_prompt_cache_misses = 0

    def process(self, pipeline, prompt, **kwargs):
        fixed = (prompt == negative_prompt if family == 'holocine' else
                 isinstance(prompt, list) and bool(prompt) and all(text == negative_prompt for text in prompt))
        # HoloCine 的原 __call__ 没有在两侧 inputs 中放 positive，原 UnitRunner
        # 因而向 encoder 原样传 None；不能替它改成 False 或据此漏掉固定负词。
        negative = fixed and (family != 'holocine' or kwargs['positive'] is False or kwargs['positive'] is None)
        if not negative:
            return original(pipeline, prompt, **kwargs)
        key = (1, kwargs['positive']) if family == 'holocine' else len(prompt)
        if key not in cache:
            cache[key] = deepcopy(original(pipeline, prompt, **kwargs))
            self.negative_prompt_cache_misses += 1
        else:
            self.negative_prompt_cache_hits += 1
        return deepcopy(cache[key])

    unit.process = MethodType(process, unit)


class StoryMemRuntime:
    """官方 T2V/M2V 与选帧；两个 stage 首次加载后保留 CPU 权重跨故事复用。"""

    def __init__(self, settings: dict, fps: int, stage: str):
        self.settings, self.fps = settings, fps
        if stage not in ('t2v', 'm2v'):
            raise ValueError('StoryMem model stage must be t2v or m2v')
        runtime = settings['runtime']
        if runtime['devices'] != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('the current StoryMem benchmark runtime requires exactly one GPU process')
        if not runtime['offload_model']:
            raise ValueError('StoryMem memory selection requires explicit model offload')
        for path in settings['weights'].values():
            if not Path(path).exists():
                raise FileNotFoundError(path)
        self.quality_model = None
        self.models = {}
        self.stage = None
        self.model_load_seconds = 0.0
        self._switch_stage(stage)

    def _load_model(self, stage: str):
        import wan
        from wan.configs import WAN_CONFIGS
        settings = self.settings
        if Path(settings['upstream']).resolve() not in Path(wan.__file__).resolve().parents:
            raise RuntimeError(f'expected StoryMem from {settings["upstream"]}; got {wan.__file__}')
        cfg = deepcopy(WAN_CONFIGS[f'{stage}-A14B'])
        if stage == 'm2v':
            for name in ('low_noise_lora', 'high_noise_lora'):
                cfg[name].weight = settings['weights'][name]
                cfg[name].r = cfg[name].lora_alpha = settings['lora_rank']
        factory = wan.WanT2V if stage == 't2v' else wan.WanM2V
        model = factory(config=cfg, checkpoint_dir=settings['weights'][stage], device_id=0, rank=0,
                        t5_fsdp=False, dit_fsdp=False, use_sp=False,
                        t5_cpu=settings['runtime']['t5_cpu'],
                        convert_model_dtype=settings['runtime']['convert_model_dtype'])
        model.text_encoder = FixedNegativeTextEncoder(model.text_encoder, settings['sampling']['negative_prompt'])
        return model

    @staticmethod
    def _move_vae(vae, device):
        # scale 中的倒数保持原值，只搬运，不重新计算。
        vae.model.to(device)
        vae.mean, vae.std = vae.mean.to(device), vae.std.to(device)
        vae.scale = [value.to(device) for value in vae.scale]
        vae.device = device

    def _switch_stage(self, stage: str):
        import torch
        if self.stage == stage:
            return
        if self.stage is not None:
            previous = self.model
            previous.low_noise_model.cpu()
            previous.high_noise_model.cpu()
            previous.text_encoder.model.cpu()
            self._move_vae(previous.vae, torch.device('cpu'))
        if stage not in self.models:
            started = time.monotonic()
            self.models[stage] = self._load_model(stage)
            self.model_load_seconds += time.monotonic() - started
        else:
            self._move_vae(self.models[stage].vae, self.models[stage].device)
        self.stage = stage
        self.model = self.models[stage]

    def switch_to_first(self):
        self._switch_stage('t2v')

    def switch_to_memory(self):
        self._switch_stage('m2v')

    def _save_keyframes(self, target: Path):
        import clip
        import extract_keyframes
        import torch
        from hpsv3 import HPSv3RewardInferencer
        weights = self.settings['weights']
        if extract_keyframes._CLIPCtx.model is None:
            model, _ = clip.load(weights['clip'], device='cpu', jit=False)
            model.eval().requires_grad_(False)
            extract_keyframes._CLIPCtx.model = model
            extract_keyframes._CLIPCtx.device = 'cpu'
            extract_keyframes._CLIPCtx.dtype = torch.float32
        if self.quality_model is None:
            self.quality_model = HPSv3RewardInferencer(config_path=weights['hps_config'],
                                                      checkpoint_path=weights['hps'], device='cuda')
        else:
            self.quality_model.model.to('cuda')
        # 仅注入显式本地资产并复用同一官方模型；阈值、去重与选帧函数保持上游原样。
        factory = extract_keyframes.HPSv3RewardInferencer
        extract_keyframes.HPSv3RewardInferencer = lambda device: self.quality_model
        try:
            extract_keyframes.save_keyframes(str(target))
        finally:
            extract_keyframes.HPSv3RewardInferencer = factory
            self.quality_model.model.to('cpu')
            torch.cuda.empty_cache()

    def render_shot(self, task: dict, segment: dict, previous_units: list[dict]) -> dict:
        from wan.utils.utils import save_video
        sample = self.settings['sampling']
        ordinal = segment['start_shot']
        if ordinal != len(previous_units) or any(
            unit['start_shot'] != index or unit['end_shot'] != index + 1
            for index, unit in enumerate(previous_units)
        ):
            raise ValueError('StoryMem memory requires an ordered completed-shot prefix')
        target = Path(task['video_path']).parent / 'segments' / f'{ordinal + 1:04d}.mp4'
        target.parent.mkdir(parents=True, exist_ok=True)
        memory = storymem_memory(previous_units, self.settings['memory'])
        fallback = MEMORY_FALLBACK if ordinal and not any(unit['keyframes'] for unit in previous_units) else None
        if fallback and not Path(memory[0]).exists():
            shutil.copyfile(target.parent / 'last_frame.jpg', memory[0])
        if target.exists() or any(target.parent.glob(f'{target.stem}_keyframe*.jpg')):
            raise FileExistsError(f'StoryMem output already exists: {target}')
        if any(not Path(path).is_file() or Path(path).stat().st_size == 0 for path in memory):
            raise FileNotFoundError('a completed StoryMem memory frame is missing or empty')
        seed = task['seed'] + ordinal * self.settings['seed_stride']
        start = time.monotonic()
        kwargs = dict(frame_num=segment['num_frames'], sampling_steps=sample['num_inference_steps'],
                      sample_solver=sample['solver'], guide_scale=sample['cfg_scale'],
                      n_prompt=sample['negative_prompt'], seed=seed,
                      offload_model=self.settings['runtime']['offload_model'])
        if ordinal == 0:
            video = self.model.generate(segment['prompts'][0], size=(sample['width'], sample['height']),
                                        shift=sample['t2v_shift'], **kwargs)
        else:
            video = self.model.generate(segment['prompts'][0], memory,
                                        first_frame_file=None, motion_frames_file=None,
                                        max_area=sample['width'] * sample['height'], shift=sample['m2v_shift'], **kwargs)
        if video.shape[1] != segment['num_frames']:
            raise ValueError(f"StoryMem output has {video.shape[1]} frames, declared {segment['num_frames']}")
        save_video(tensor=video[None], save_file=str(target), fps=self.fps, nrow=1,
                   normalize=True, value_range=(-1, 1))
        del video
        self._save_keyframes(target)
        return {**segment, 'seed': seed, 'output': str(target), 'memory_files': memory,
                **({'memory_fallback': fallback} if fallback else {}),
                'keyframes': sorted(str(path) for path in target.parent.glob(f'{target.stem}_keyframe*.jpg')),
                'seconds': time.monotonic() - start}


TI2V_CONFIG = {
    '_class_name': 'WanModel', '_diffusers_version': '0.33.0', 'dim': 3072, 'eps': 1e-06,
    'ffn_dim': 14336, 'freq_dim': 256, 'in_dim': 48, 'model_type': 'ti2v', 'num_heads': 24,
    'num_layers': 30, 'out_dim': 48, 'text_len': 512,
}


def ensure_ti2v_checkpoint(directory: Path) -> None:
    """补齐官方 from_pretrained 需要、而本地镜像没带上的 config 和分片索引。"""
    config_path = directory / 'config.json'
    if config_path.exists():
        found = json.loads(config_path.read_text())
        if found.get('model_type') != 'ti2v' or found.get('in_dim') != 48:
            raise ValueError(f'checkpoint config is not Wan2.2 TI2V-5B: {config_path}')
    else:
        config_path.write_text(json.dumps(TI2V_CONFIG, indent=2) + '\n')
    index_path = directory / 'diffusion_pytorch_model.safetensors.index.json'
    if index_path.exists():
        return
    from safetensors import safe_open
    shards = sorted(directory.glob('diffusion_pytorch_model-*-of-*.safetensors'))
    if len(shards) != 3:
        raise FileNotFoundError(f'{directory} does not contain the three TI2V weight shards')
    weight_map = {}
    seen_patch = False
    for shard in shards:
        with safe_open(shard, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f'duplicate tensor {key}')
                weight_map[key] = shard.name
                if key == 'patch_embedding.weight':
                    shape = handle.get_slice(key).get_shape()
                    if len(shape) < 2 or shape[1] != 48:
                        raise ValueError(f'patch embedding shape {shape} is not TI2V in_dim 48')
                    seen_patch = True
    if not seen_patch:
        raise ValueError('TI2V checkpoint has no patch_embedding.weight')
    index_path.write_text(json.dumps({'metadata': {'total_size': sum(shard.stat().st_size for shard in shards)},
                                      'weight_map': weight_map}, indent=2) + '\n')


def load_backend(settings: dict, fps: int):
    import torch
    family, weights, sample = settings['family'], settings['weights'], settings['sampling']
    if family == 'wan_official':
        import wan
        from wan.configs import WAN_CONFIGS
        checkpoint = Path(weights['checkpoint_dir'])
        ensure_ti2v_checkpoint(checkpoint)
        if Path(settings['upstream']).resolve() not in Path(wan.__file__).resolve().parents:
            raise RuntimeError(f'expected official Wan from {settings["upstream"]}; got {wan.__file__}')
        runtime = settings['runtime']
        model = wan.WanTI2V(config=WAN_CONFIGS['ti2v-5B'], checkpoint_dir=str(checkpoint),
                            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
                            t5_cpu=runtime['t5_cpu'], init_on_cpu=True,
                            convert_model_dtype=runtime['convert_model_dtype'])
        return model, None
    if family == 'wan21_official':
        import wan
        from wan.configs import WAN_CONFIGS
        if Path(settings['upstream']).resolve() not in Path(wan.__file__).resolve().parents:
            raise RuntimeError(f'expected official Wan2.1 from {settings["upstream"]}; got {wan.__file__}')
        model = wan.WanT2V(config=WAN_CONFIGS[settings['task']], checkpoint_dir=weights['checkpoint_dir'],
                           device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False,
                           t5_cpu=settings['runtime']['t5_cpu'])
        return model, None
    if family == 'storymem':
        return StoryMemRuntime(settings, fps, 't2v'), None
    if family in ('wan_single', 'wan_segment'):
        from mspc.inference.flow_unipc import require_sampler
        from mspc.inference.wan22 import Wan22InferenceRuntime
        if "sampler" not in sample:
            raise ValueError("sampling.sampler is required")
        require_sampler(sample["sampler"])
        model = {**settings['model'], 'diffsynth_root': settings['upstream']}
        inference = {**sample, 'target_height': sample['height'], 'target_width': sample['width'], 'fps': fps, 'device': 'cuda', 'model_paths': weights,
                     'weight_source': 'full_dit_checkpoint' if settings['checkpoint'] is not None else 'pretrained_model_paths'}
        runtime = Wan22InferenceRuntime(model_cfg=model, inference_cfg=inference,
                                       pipeline_cfg=settings['pipeline'], dit_ckpt=settings['checkpoint'])
        return runtime, None
    if family == 'multishotmaster':
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
        pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda', use_usp=False,
            model_configs=[ModelConfig(path=weights[k]) for k in ('dit', 't5', 'vae')],
            tokenizer_config=ModelConfig(path=weights['tokenizer']), redirect_common_files=False)
    else:
        from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig
        pipe = WanVideoHoloCinePipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda', use_usp=False,
            model_configs=[ModelConfig(path=weights[k], offload_device='cpu') for k in ('t5', 'dit_high', 'dit_low', 'vae')],
            tokenizer_config=ModelConfig(path=weights['tokenizer']), redirect_common_files=False)
        pipe.enable_vram_management()
        pipe.to('cuda')
    cache_pipeline_negative_prompt(pipe, family, sample['negative_prompt'])
    return pipe, None


def render_story(settings: dict, task: dict, board: dict, segments: list[dict], backend, count_tokens, fps: int,
                 initial_units: list[dict], *, on_storymem_progress=None) -> list[dict]:
    family = settings['family']
    runtime, _embed = backend
    directory = Path(task['video_path']).parent
    directory.mkdir(parents=True, exist_ok=True)
    records = list(initial_units)

    def record(unit):
        records.append(unit)
        write_json(directory / 'segments.json', records)

    if family == 'storymem':
        if not records:
            raise ValueError('StoryMem requires its completed first shot')
        for segment in segments[len(records):]:
            if on_storymem_progress is not None:
                on_storymem_progress('started', records)
            record(runtime.render_shot(task, segment, records))
            if on_storymem_progress is not None:
                on_storymem_progress('completed', records)
    elif family in ('wan_single', 'wan_segment'):
        from mspc.inference.segment_story import generate_story
        memory = settings['model']['memory_prefix']
        history = settings['history_policy'] != 'none'
        if history and memory is None:
            raise ValueError('reference history requires a renderer with the reference pathway')
        generate_story(runtime, annotation=board, shot_ids=[f'shot-{i+1}' for i in range(len(board['shots']))],
                       segments=segments,
                       output=directory / 'segments', token_counter=count_tokens,
                       frames_max=memory['frames_max'] if history else None,
                       seed=task['seed'], segment_seed_stride=settings['seed_stride'], on_segment=record,
                       history_policy=settings['history_policy'])
    elif family in OFFICIAL_WAN:
        if family == 'wan_official':
            from wan.utils.utils import save_video
        else:
            from wan.utils.utils import cache_video as save_video
        sample = settings['sampling']
        for ordinal, segment in enumerate(segments):
            if len(segment['prompts']) != 1:
                raise ValueError('official Wan takes one prompt per shot')
            target = directory / 'segments' / f'{ordinal + 1:04d}.mp4'
            target.parent.mkdir(parents=True, exist_ok=True)
            seed = task['seed'] + ordinal * settings['seed_stride']
            start = time.monotonic()
            video = runtime.generate(segment['prompts'][0], size=(sample['width'], sample['height']),
                                     frame_num=segment['num_frames'], shift=sample['sigma_shift'],
                                     sample_solver=sample['solver'], sampling_steps=sample['num_inference_steps'],
                                     guide_scale=sample['cfg_scale'], n_prompt=sample['negative_prompt'],
                                     seed=seed, offload_model=settings['runtime']['offload_model'])
            if video.shape[1] != segment['num_frames']:
                raise ValueError(f"official Wan output has {video.shape[1]} frames, declared {segment['num_frames']}")
            save_video(tensor=video[None], save_file=str(target), fps=segment['output_fps'], nrow=1,
                       normalize=True, value_range=(-1, 1))
            del video
            record({**segment, 'seed': seed, 'output': str(target), 'seconds': time.monotonic() - start,
                    'solver': sample['solver'], 'size': [sample['width'], sample['height']]})
    else:
        for ordinal, segment in enumerate(segments):
            target = directory / 'segments' / f'{ordinal + 1}.mp4'
            target.parent.mkdir(parents=True, exist_ok=True)
            seed = task['seed'] + ordinal * settings['seed_stride']
            start = time.monotonic()
            sample = settings['sampling']
            kwargs = {'negative_prompt': sample['negative_prompt'], 'seed': seed,
                      'height': sample['height'], 'width': sample['width'],
                      'num_inference_steps': sample['num_inference_steps'], 'cfg_scale': sample['cfg_scale'],
                      'sigma_shift': sample['sigma_shift'], **settings['pipeline']}
            if family == 'multishotmaster':
                from util import pad_shot_groups_to_4n_plus_1, rgb_to_latent_shot_groups_list, get_user_wanted_frames
                from diffsynth.utils.data import save_video
                padded, wanted = pad_shot_groups_to_4n_plus_1(segment['shot_groups'])
                latent = rgb_to_latent_shot_groups_list(padded)
                kwargs['negative_prompt'] = [sample['negative_prompt']] * len(wanted)
                video = runtime(prompt=segment['prompts'], num_frames=latent[-1][-1],
                                shot_groups=segment['shot_groups'], latent_shot_groups=latent, **kwargs)
                video = get_user_wanted_frames(video, padded, wanted)
            else:
                from diffsynth import save_video
                video = runtime(prompt=segment['prompt'], num_frames=segment['num_frames'],
                                shot_cut_frames=segment['shot_cut_frames'], **kwargs)
            if len(video) != segment['num_frames']:
                raise ValueError(f"native output has {len(video)} frames, declared {segment['num_frames']}")
            save_video(video, str(target), fps=fps, quality=sample['video_quality'])
            record({**segment, 'seed': seed, 'output': str(target), 'seconds': time.monotonic() - start})
    from mspc.core.video import concat_shot_videos
    concat_shot_videos([Path(unit['output']) for unit in records], output_path=Path(task['video_path']), ffmpeg=settings['ffmpeg'])
    return records


class RendererSession:
    """一个 worker 的冻结方法及模型生命周期；任务记录仍按故事独立落盘。"""

    def __init__(self, output: Path, config: dict, method: str, stories: dict):
        self.output, self.method, self.stories = output, method, stories
        self.settings = config['methods'][method]
        self.count_tokens = self.provenance = self.backend = self.fps = None

    def render_task(self, task: dict) -> None:
        if task['method'] != self.method:
            raise ValueError('one renderer session requires one frozen method')
        settings = self.settings
        target = Path(task['generation_record_path'])
        if target.exists():
            return  # completed和失败都不重抽；未知中断也保留，另一次运行需新目录。
        source = plan_path(self.output, settings['planner'], task)
        if not source.exists():
            raise FileNotFoundError(f'planning has not run: {source}')
        planned = json.loads(source.read_text())
        if planned['status'] != 'completed':
            write_json(target, {'status': 'planning_failed', 'plan': str(source), 'planner_status': planned['status']})
            return
        if self.count_tokens is None:
            self.count_tokens, self.provenance = select_runtime(settings)
        board = planned['storyboard']
        try:
            segments = adapt_storyboard(board, family=settings['family'], limits=settings['limits'], count_tokens=self.count_tokens)
        except (ValueError, TypeError, KeyError) as error:
            write_json(target, {'status': 'adaptation_failed', 'plan': str(source), 'error': f'{type(error).__name__}: {error}'})
            return
        write_json(target.parent / 'inputs.json', {'task': task, 'plan': str(source), 'settings': settings,
                                                  'storyboard': board, 'segments': segments, 'runtime': self.provenance})
        fps = (settings['sampling']['output_fps'] if settings['family'] == 'holocine'
               else self.stories[task['story_id']]['fps'])
        if self.fps is not None and fps != self.fps:
            raise ValueError('one renderer process requires one explicitly shared output FPS')
        load_seconds = 0.0
        if self.backend is None:
            started = time.monotonic()
            try:
                self.backend = load_backend(settings, fps)
            except Exception as error:
                write_json(target, {'status': 'runtime_failed', 'error': f'{type(error).__name__}: {error}'})
                raise
            load_seconds = time.monotonic() - started
            self.fps = fps
        runtime = self.backend[0]
        storymem = settings['family'] == 'storymem'
        previous_load_seconds = runtime.model_load_seconds if storymem else 0.0
        record = {'status': 'generation_started', 'model_load_seconds': load_seconds}
        write_json(target, record)
        start = time.monotonic()
        try:
            initial = []
            if storymem:
                runtime.switch_to_first()
                initial = [runtime.render_shot(task, segments[0], [])]
                write_json(target.parent / 'segments.json', initial)
                if len(segments) > 1:
                    runtime.switch_to_memory()
            with ExitStack() as routing:
                if settings['family'] in ('wan_single', 'wan_segment'):
                    from mspc.evaluation.renderer_diagnostics import communication_intervention
                    routing.enter_context(communication_intervention(runtime.pipe.dit, settings['communication']))
                units = render_story(settings, task, board, segments, self.backend, self.count_tokens, fps, initial)
            media = inspect_video(Path(task['video_path']), settings['ffprobe'])
            target_seconds = self.stories[task['story_id']]['target_seconds']
            record.update(status='completed', segments=units, media=media, target_seconds=target_seconds,
                          duration_error_seconds=media['duration_seconds'] - target_seconds,
                          generated_shots=len(board['shots']), generation_calls=len(units))
        except Exception as error:
            record.update(status='generation_failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            late_load_seconds = runtime.model_load_seconds - previous_load_seconds if storymem else 0.0
            record['model_load_seconds'] += late_load_seconds
            record['elapsed_seconds'] = time.monotonic() - start - late_load_seconds
            write_json(target, record)
        print(json.dumps({**task, 'status': record['status'], 'elapsed_seconds': record['elapsed_seconds']}), flush=True)


def render_tasks(output: Path, config: dict, method: str, tasks: list[dict], stories: dict) -> None:
    session = RendererSession(output, config, method, stories)
    for task in tasks:
        session.render_task(task)


def inspect_storymem_video(path: Path, ffprobe: str, settings: dict, fps: int, frames: int) -> dict:
    """逐帧解码计数，确保待拼接镜头和成片具有冻结的帧数、尺寸与帧率。"""
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'StoryMem clip missing or empty: {path}')
    result = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 1:
        raise ValueError(f'{path}: expected one video stream')
    media = streams[0]
    duration = float(media['duration'])
    if (int(media['nb_read_frames']) != frames or Fraction(media['avg_frame_rate']) != fps
            or media['width'] != settings['sampling']['width']
            or media['height'] != settings['sampling']['height']
            or not math.isfinite(duration) or abs(duration - frames / fps) > 0.001):
        raise ValueError(f'{path}: video does not match its frozen frame count, FPS, size and duration')
    return {**media, 'duration_seconds': duration}


def load_storymem_resume_source(output: Path, method: str, story_id: str, ffprobe: str) -> dict:
    """只读检查冻结任务和有完整记录的连续前缀，不从目录推断任何新镜或记忆。"""
    if method != 'official_storymem_14b':
        raise ValueError('completed-prefix resume is only defined for official StoryMem')
    manifest = load_manifest(output / 'inputs/requests.json')
    matches = [row for row in manifest['requests'] if row['method'] == method and row['story_id'] == story_id]
    if len(matches) != 1:
        raise ValueError('resume requires exactly one frozen request')
    task, = matches
    settings = json.loads((output / 'inputs/config.json').read_text())['methods'][method]
    if settings['family'] != 'storymem' or settings['limits']['max_shots'] != 1:
        raise ValueError('resume requires the frozen one-shot StoryMem configuration')
    directory = Path(task['video_path']).parent
    if Path(task['generation_record_path']) != directory / 'generation.json':
        raise ValueError('source generation and video paths do not share a story directory')
    if Path(task['video_path']).exists():
        raise ValueError('source already has a whole video')
    generation = json.loads(Path(task['generation_record_path']).read_text())
    status = generation.get('status')
    timeout_failure = (status == 'generation_failed' and
                       generation.get('process_error') == 'TimeoutError: story or cumulative leased-GPU limit reached')
    if ((status != 'generation_started' and not timeout_failure) or generation.get('error')):
        raise ValueError('source is completed or failed for a reason other than an interrupted timeout')
    source = json.loads((directory / 'inputs.json').read_text())
    plan = plan_path(output, settings['planner'], task)
    planned = json.loads(plan.read_text())
    if (source['task'] != task or source['settings'] != settings or source['plan'] != str(plan)
            or planned['status'] != 'completed' or source['storyboard'] != planned['storyboard']):
        raise ValueError('source inputs differ from the frozen request, configuration or plan')
    stories = json.loads((output / 'inputs/stories.json').read_text())['stories']
    story, = [row for row in stories if row['story_id'] == story_id]
    if story['fps'] != task['fps'] or story['target_seconds'] != task['target_seconds']:
        raise ValueError('source story timing differs from the frozen request')
    segments = source['segments']
    board = source['storyboard']
    if len(segments) != len(board['shots']) or len(segments) != task['expected_shot_count']:
        raise ValueError('source segments do not cover the frozen story')
    for index, segment in enumerate(segments):
        expected = segment_inputs(board, 'storymem', index, index + 1)
        counts = segment.get('token_counts')
        if ({key: value for key, value in segment.items() if key != 'token_counts'} != expected
                or not isinstance(counts, list) or len(counts) != 1
                or type(counts[0]) is not int or not 0 < counts[0] <= settings['limits']['max_tokens']
                or segment['num_frames'] > settings['limits']['max_frames']
                or segment['num_frames'] % 4 != 1):
            raise ValueError(f'original shot {index + 1} differs from frozen StoryMem inputs')
    units = json.loads((directory / 'segments.json').read_text())
    if not isinstance(units, list) or not 0 < len(units) < len(segments):
        raise ValueError('resume requires a nonempty, incomplete prefix')
    media = []
    for index, unit in enumerate(units):
        segment = segments[index]
        if any(unit.get(key) != value for key, value in segment.items()):
            raise ValueError(f'completed shot {index + 1} differs from the frozen input')
        path = directory / 'segments' / f'{index + 1:04d}.mp4'
        if unit.get('output') != str(path) or unit.get('seed') != task['seed'] + index * settings['seed_stride']:
            raise ValueError(f'completed shot {index + 1} has an unexpected path or seed')
        seconds = unit.get('seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('completed shot has invalid execution time')
        expected_memory = storymem_memory(units[:index], settings['memory'])
        if unit.get('memory_files') != expected_memory:
            raise ValueError(f'completed shot {index + 1} has inconsistent memory')
        keyframes = unit.get('keyframes')
        if not isinstance(keyframes, list) or keyframes != sorted(keyframes) or len(keyframes) != len(set(keyframes)):
            raise ValueError('completed shot keyframes must be a sorted unique list')
        for value in keyframes:
            frame = Path(value)
            if (frame.parent != path.parent or not re.fullmatch(rf'{index + 1:04d}_keyframe[0-9]+\.jpg', frame.name)
                    or frame.is_symlink() or not frame.is_file() or frame.stat().st_size == 0):
                raise ValueError(f'completed shot {index + 1} has an invalid keyframe')
        media.append(inspect_storymem_video(path, ffprobe, settings, story['fps'], segment['num_frames']))
    return {'task': task, 'settings': settings, 'story': story, 'source': source,
            'segments': segments, 'units': units, 'prefix_media': media}


def resume_storymem(output: Path, method: str, story_id: str, destination: Path,
                    run_root: Path, source_slurm_job: str) -> None:
    """恢复到独立目录；原始生成记录、已完成镜头与关键帧保持只读。"""
    expected = (run_root / 'resumed' / method / story_id).resolve()
    if destination.resolve() != expected:
        raise ValueError('resume destination must be the uniquely reserved story directory')
    queue_config = json.loads((run_root / 'inputs/queue.json').read_text())
    queue = json.loads(Path(queue_config['methods'][method]['queue']).read_text())
    matches = [row for row in queue if row['story_id'] == story_id and row['method'] == method]
    if len(matches) != 1 or Path(matches[0]['execution_output_root']).resolve() != output.resolve():
        raise ValueError('resume must match one frozen queue task')
    from mspc.comparison_methods.dispatch import validate_storymem_timeout
    timeout = validate_storymem_timeout(run_root, matches[0], source_slurm_job)
    # 父进程独占预留目录后会记录实际 child PID；只允许它启动的这个消费者写入。
    deadline = time.monotonic() + 5
    while True:
        process = json.loads((destination / 'process.json').read_text())
        if process.get('status') == 'running' and process.get('stage') == 'render' and process.get('pid') == os.getpid():
            break
        if time.monotonic() >= deadline:
            raise ValueError('resume renderer is not the process registered by the supervisor')
        time.sleep(0.05)
    if process.get('source_timeout') != timeout:
        raise ValueError('resume supervisor has different source timeout evidence')
    context = load_storymem_resume_source(output, method, story_id, 'ffprobe')
    if any(context['task'].get(key) != matches[0][key] for key in
           ('method', 'story_id', 'input_tier', 'seed', 'video_path', 'generation_record_path')):
        raise ValueError('resume queue differs from the frozen source request')
    task = {**context['task'], 'video_path': str(destination / 'video.mp4'),
            'generation_record_path': str(destination / 'generation.json')}
    source_directory = Path(context['task']['video_path']).parent.resolve()
    resolved = destination.resolve()
    if (resolved == source_directory or source_directory in resolved.parents
            or resolved in source_directory.parents or not destination.is_dir()
            or Path(task['generation_record_path']).exists() or Path(task['video_path']).exists()):
        raise ValueError('resume destination must be a distinct, newly reserved directory')
    count_tokens, provenance = select_runtime(context['settings'])
    if provenance != context['source']['runtime']:
        raise ValueError('StoryMem upstream runtime differs from the original run')
    # 冻结故事使用当时的 schema；已逐镜核对当前 caption 语义，此处只复核原 tokenizer 计数。
    segments = context['segments']
    if any([count_tokens(text) for text in segment['prompts']] != segment['token_counts'] for segment in segments):
        raise ValueError('current StoryMem tokenizer differs from the original run')
    prefix = context['units']
    write_json(destination / 'inputs.json', {'source_generation_record': context['task']['generation_record_path'],
                                             'source_inputs': context['source'], 'task': task,
                                             'prefix_media': context['prefix_media']})
    write_json(destination / 'segments.json', prefix)
    start = time.monotonic()
    record = {'status': 'generation_started', 'source_generation_record': context['task']['generation_record_path'],
              'completed_prefix_shots': len(prefix), 'new_generation_calls_started': 0, 'new_generation_calls': 0}
    write_json(Path(task['generation_record_path']), record)

    def record_progress(event, units):
        if event == 'started':
            started = {**record, 'new_generation_calls_started': record['new_generation_calls_started'] + 1}
            write_json(Path(task['generation_record_path']), started)
            record.update(started)
        else:
            # 已保存的 segments 是已完成调用的依据；之后再同步 generation。
            record['new_generation_calls'] = len(units) - len(prefix)
            write_json(Path(task['generation_record_path']), record)

    try:
        # 前缀已有首镜，直接加载原方法的 M2V；不调用 T2V、不重新选帧。
        backend = (StoryMemRuntime(context['settings'], context['story']['fps'], 'm2v'), None)
        units = render_story(context['settings'], task, context['source']['storyboard'], segments,
                             backend, count_tokens, context['story']['fps'], prefix,
                             on_storymem_progress=record_progress)
        media = inspect_storymem_video(Path(task['video_path']), 'ffprobe', context['settings'],
                                       context['story']['fps'], sum(unit['num_frames'] for unit in units))
        record.update(status='completed', segments=units, media=media,
                      target_seconds=context['story']['target_seconds'],
                      duration_error_seconds=media['duration_seconds'] - context['story']['target_seconds'],
                      generated_shots=len(units), generation_calls=len(units),
                      new_generation_calls=len(units) - len(prefix))
    except Exception as error:
        record.update(status='generation_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        write_json(Path(task['generation_record_path']), record)


def main() -> None:
    parser = argparse.ArgumentParser(description='Resume an interrupted StoryMem story from its completed prefix')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--story-id', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-slurm-job', required=True)
    args = parser.parse_args()
    resume_storymem(args.output, args.method, args.story_id, args.destination, args.run_root, args.source_slurm_job)


if __name__ == '__main__':
    main()
