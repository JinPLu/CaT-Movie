"""关键帧路线的逐故事执行：准备、出图、逐镜视频、拼接。命令行与 benchmark 生成入口相同，由对比方法队列调用。"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from mspc.comparison_methods.keyframe_video import load_config, prepare_keyframe_story, read_json, write_json

MODULES = {'storydiffusion_wan_a14b': 'mspc.comparison_methods.storydiffusion',
           'ic_lora_wan_a14b': 'mspc.comparison_methods.ic_lora'}


def story_config(template: dict, story_id: str, characters: dict) -> dict:
    """characters 是开镜输入的角色登记（标签 → 外貌）；IC-LoRA 按标签号显式命名。"""
    config = {**template, 'story_id': story_id}
    if config['image_method'] == 'ic_lora':
        names = {}
        for tag in characters:
            match = re.fullmatch(r'\[C(\d+)\]', tag)
            if match is None:
                raise ValueError(f'{story["story_id"]}: unexpected character tag {tag}')
            names[tag] = f'Character {match.group(1)}'
        config['ic_lora'] = {**config['ic_lora'], 'character_names': names}
    return config


def completed_images(config: dict, prepared: dict) -> bool:
    """复用已完成的关键帧阶段；未完成或缺图的旧产物须先人工核查。"""
    from PIL import Image

    directory = (Path(config['output_root']) / 'inference/raw/images'
                 / config['image_method'] / prepared['story_id'])
    if not directory.exists():
        return False
    record = directory / 'generation.json'
    if not record.is_file() or read_json(record)['status'] != 'completed':
        raise ValueError(f'incomplete image stage requires inspection: {directory}')
    expected = {f'{i+1:04d}.png' for i in range(len(prepared['shots']))}
    if {path.name for path in directory.glob('*.png')} != expected:
        raise ValueError(f'completed image stage has missing or unexpected panels: {directory}')
    if config['image_method'] == 'ic_lora':
        size = tuple(prepared['panel_size'])
    else:
        sample = config['storydiffusion']
        size = (sample['width'], sample['height'])
    for name in sorted(expected):
        with Image.open(directory / name) as image:
            if image.size != size:
                raise ValueError(f'completed image panel has a different size: {directory / name}')
            image.verify()
    return True


def run_stages(output: Path, method: str, config: dict, env: dict) -> None:
    """冻结逐故事配置后依次跑准备、出图（已完成则复用）、逐镜视频、拼接。"""
    story_id = config['story_id']
    path = output / 'inputs/configs' / f'{story_id}.json'
    if path.exists() and read_json(path) != config:
        raise FileExistsError(f'refusing to change the frozen configuration for {story_id}')
    write_json(path, config)

    def stage(*args: str) -> None:
        print(f'STAGE {args[0]} {story_id}', flush=True)
        subprocess.run([sys.executable, '-m', MODULES[method], *args, '--config', str(path)], check=True, env=env)

    stage('prepare')
    prepared = read_json(output / 'inputs/prepared' / f'{story_id}.json')
    shots = len(prepared['shots'])
    if completed_images(config, prepared):
        print(f'REUSE completed images {story_id}', flush=True)
    else:
        stage('images')
    stage('videos', '--shot-indices', *map(str, range(shots)))
    stage('assemble')


def render(output: Path, method: str, story_ids: list[str]) -> None:
    template = load_config(output / 'inputs/keyframe-config.json')
    if template['video_method'] != method or 'story_id' in template:
        raise ValueError('keyframe template must name this method and leave story_id to the queue')
    stories = {s['story_id']: s for s in read_json(template['given_plan_source'])['stories']}
    env = dict(os.environ, PYTHONPATH=':'.join([*template['python_path'], os.environ['PYTHONPATH']]))
    for story_id in story_ids:
        characters = prepare_keyframe_story(stories[story_id])['characters']
        run_stages(output, method, story_config(template, story_id, characters), env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['render'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', required=True, choices=sorted(MODULES))
    parser.add_argument('--story-ids', nargs='+', required=True)
    args = parser.parse_args()
    render(args.output, args.method, args.story_ids)


if __name__ == '__main__':
    main()
