"""在已有原片副本上重放官方选帧，记录分数与实际写图，不生成新视频。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(os.path.expandvars(args.config.read_text()))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, config['upstream'])
    import cv2
    import clip
    import torch
    import extract_keyframes as official
    from hpsv3 import HPSv3RewardInferencer

    started = time.monotonic()
    target = output / '0001.mp4'
    shutil.copyfile(config['video'], target)
    scores, writes = [], []
    checkpoint = config['weights']
    model = HPSv3RewardInferencer(config_path=checkpoint['hps_config'],
                                checkpoint_path=checkpoint['hps'], device='cuda')
    original_reward = model.reward

    def reward(*args, **kwargs):
        result = original_reward(*args, **kwargs)
        row = dict(call=len(scores), prompts=kwargs.get('prompts'),
                   scores=result.detach().float().cpu().tolist() if isinstance(result, torch.Tensor)
                   else [[float(v) for v in item] for item in result])
        scores.append(row)
        with (output / 'quality-scores.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return result

    model.reward = reward
    # 同正式部署只指向本地资产；官方函数、阈值和判断均不改变。
    clip_model, _ = clip.load(checkpoint['clip'], device='cpu', jit=False)
    clip_model.eval().requires_grad_(False)
    official._CLIPCtx.model = clip_model
    official._CLIPCtx.device = 'cpu'
    official._CLIPCtx.dtype = torch.float32
    factory, imwrite = official.HPSv3RewardInferencer, cv2.imwrite
    official.HPSv3RewardInferencer = lambda device: model

    def record_write(path, image, *args, **kwargs):
        result = imwrite(path, image, *args, **kwargs)
        writes.append(dict(path=str(path), success=bool(result)))
        return result

    cv2.imwrite = record_write
    result = dict(started_at=datetime.now(timezone.utc).isoformat(), source_video=config['video'],
                  official_source=config['upstream'], threshold=official.HPSV3_QUALITY_THRESHOLD)
    try:
        official.save_keyframes(str(target))
        result.update(status='completed', keyframes=[str(p) for p in output.glob('*keyframe*.jpg')],
                      quality_calls=len(scores), scores=scores, writes=writes,
                      decoded_frames=len(official.read_video(str(target))[0]))
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', scores=scores, writes=writes)
        raise
    finally:
        official.HPSv3RewardInferencer, cv2.imwrite = factory, imwrite
        result.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scores','writes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
