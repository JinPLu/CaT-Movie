"""只读核验显式部署配置；不导入 torch、不初始化 CUDA、不加载模型。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent


def read_config(path: Path):
    text = os.path.expandvars(path.read_text())
    if '${' in text:
        raise ValueError(f'{path}: unresolved environment variable')
    return json.loads(text)


def asset(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {'path': path, 'exists': False}
    if p.is_file():
        return {'path': path, 'exists': True, 'bytes': p.stat().st_size}
    # 权重目录不能仅凭目录存在就通过；核索引声明的每个分片以及空文件。
    indexes = []
    for index in sorted(p.rglob('*.index.json')):
        data = json.loads(index.read_text())
        if 'weight_map' in data:
            shards = sorted(set(data['weight_map'].values()))
            indexes.append({'index': str(index), 'shards': len(shards),
                            'missing': [name for name in shards if not (index.parent / name).is_file()]})
    files = [f for f in p.rglob('*') if f.is_file() and '.cache' not in f.relative_to(p).parts]
    return {'path': path, 'exists': True, 'files': len(files),
            'empty_files': [str(f) for f in files if f.stat().st_size == 0], 'indexes': indexes}


def source(path: str, kind: str, expected: str | None) -> dict:
    p = Path(path)
    result = {'path': path, 'exists': p.is_dir(), 'expected_revision': expected}
    if not p.is_dir():
        return result
    if kind in ('git', 'project'):
        # 共享盘所有权可与当前UID不同；仅本次只读Git调用认可显式配置的目录。
        owner = p.parents[1] if kind == 'project' else p.resolve()
        prefix = ['git', '-c', f'safe.directory={owner}', '-C', str(p)]
        revision = subprocess.run([*prefix, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run([*prefix, 'status', '--porcelain', '--untracked-files=no', '--', '.'],
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        result.update(revision=revision, tracked_changes=changes)
    else:
        record = json.loads((p / kind).read_text())
        result.update(revision=record['commit'], source_record=record,
                      limitation='SOURCE.json是交付记录，未独立逐文件比对官方归档。')
    result['revision_matches'] = result['revision'] == expected if expected is not None else None
    return result


def environment(python: str, pythonpath: list[str]) -> dict:
    script = '''import importlib.metadata as m, json, sys
packages = {}
for name in ['torch', 'diffusers', 'transformers', 'peft', 'flash-attn', 'accelerate', 'sentencepiece',
             'hpsv3', 'clip', 'opencv-python', 'decord']:
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
print(json.dumps({'executable': sys.executable, 'version': sys.version, 'packages': packages}))
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': os.pathsep.join(pythonpath)}
    result = subprocess.run([python, '-c', script], env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def inspect() -> dict:
    manifest = read_config(ROOT / 'deployment.json')
    rows = []
    for profile in manifest['methods']:
        row = {'method': profile['method'], 'config': profile['config']}
        try:
            config = read_config(ROOT / profile['config'])
            settings = config[profile['section']] if profile['section'] else config
            row['source'] = source(settings['upstream'], profile['source_record'], profile['upstream_revision'])
            paths = settings['weights']
            if isinstance(paths, str):
                paths = {'weights': paths}
            paths = {**paths, **{key: settings[key] for key in profile['extra_asset_keys']}}
            row['assets'] = {key: [asset(p) for p in value] if isinstance(value, list) else asset(value)
                             for key, value in paths.items()}
            row['supporting_assets'] = {key: asset(path) for key, path in profile['supporting_assets'].items()}
            row['environment'] = environment(profile['python'], profile['pythonpath'])
            if profile['section']:
                i2v = config['wan_i2v']
                row['wan_i2v'] = {'source': source(i2v['upstream'], 'git', '052c68d2627a22a95f79dbb4d3376cc30c96f1a3'),
                                  'assets': asset(i2v['weights'])}
            row['inspection_completed'] = True
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            row.update(inspection_completed=False, error=f'{type(error).__name__}: {error}')
        rows.append(row)
    return {'observed_at': datetime.now(timezone.utc).isoformat(), 'scope': manifest['scope'], 'methods': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'methods_inspected': len(result['methods']),
                      'inspection_errors': [r['method'] for r in result['methods'] if not r['inspection_completed']]}))


if __name__ == '__main__':
    main()
