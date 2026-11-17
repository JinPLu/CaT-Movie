#!/usr/bin/env python3
"""第 8 步空间层：按块、显式配置、独立生产者与可续跑阶段。

清单读 ``final/blocks.jsonl``，标注读 ``final/block-annotations.jsonl``（第 18 版合同、已校验）。可处理的块是登记表层合法的块：
``status == "ok"``，或 ``status == "invalid"`` 且整块错误为空；块里个别镜的记录不合法（``shot_errors``）时，这一镜照样取帧、做深度，
但不定位、不分割，原因记在 ``frames.json`` 的该镜 ``annotation_errors`` 里。每镜在 16fps 训练帧网格上每隔 ``frame_stride`` 帧取一帧；
产物里的帧号一律是镜内训练帧号（0、4、8…），``frames.json`` 记下每张图对应的源帧号。阶段顺序：
prepare → locate → sam → depth → assemble → audit（review 画叠图）→ cleanup（删掉用完的取帧）。
另有子命令 ``select``（不属于上面的阶段）：把可处理且取帧阶段已成功的块的键按标注文件的顺序写成键清单，给 ``--selection`` 用。
"""
from __future__ import annotations
import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mspc.data.storyboard import holding_targets
from mspc.data.spatial_client import LocalClient, normalized_corners, save
POSE_CAVEAT = '姿态几何未通过训练级视觉验收；保留全部133点与原始分数，不据此筛选。'

def read(p):
    return json.loads(Path(p).read_text())

def training_frames(render_frames, stride):
    """镜内训练帧号：第 0、s、2s… 帧。训练帧数是 4n+1，间隔 4 时首末帧都在内；帧数减 1 不能被间隔整除就报错，不补帧。"""
    if type(render_frames) is not int or type(stride) is not int or render_frames < 1 or stride < 1:
        raise ValueError('invalid training frame count or stride')
    if (render_frames-1) % stride:
        raise ValueError(f'{render_frames} training frames do not divide by stride {stride}')
    return list(range(0, render_frames, stride))

def sampled_frames(train_frames):
    """定位与叠图用的首、中、末三张：从已取出帧的训练帧号里挑，返回训练帧号。"""
    if not train_frames:
        raise ValueError('empty shot')
    return sorted({train_frames[0], train_frames[len(train_frames) // 2], train_frames[-1]})

def frame_file(block, fs, train_frame):
    """取出的图按取出顺序编号（SAM 按目录里第几张图数帧），训练帧号只写在 frames.json 与各产物里。"""
    return block/'frames'/fs['id']/f"{fs['train_frames'].index(train_frame):06d}.jpg"

def instance_prompt(entry):
    """SAM 官方实例框：归一化左上/右下角，专属 corner 标签 [2,3]。"""
    return normalized_corners(entry['bbox_2d']), [2, 3]


def validate_locations(obj, names, positions):
    entries = obj['entities']
    if sorted(e['name'] for e in entries) != sorted(names):
        raise ValueError('localization names differ from present registry')
    for entity in entries:
        states = entity['visibility']
        if len(states) != len(positions) or any(s not in ('clear', 'absent', 'unsure') for s in states):
            raise ValueError('invalid per-frame visibility')
        clear = [p for p, s in zip(positions, states, strict=True) if s == 'clear']
        if not clear:
            if any(entity[k] is not None for k in ('frame_index', 'bbox_2d')):
                raise ValueError('nonvisible entity must have null coordinates')
            continue
        if type(entity['frame_index']) is not int or entity['frame_index'] != clear[0]:
            raise ValueError('anchor must be earliest clear sampled frame')
        normalized_corners(entity['bbox_2d'])
    return obj

def in_training_frames(result, train_frames, anchor_frame):
    """SAM 按目录里第几张图给帧编号；落盘一律改成镜内训练帧号（锚点也是训练帧号）。"""
    result['tracks'] = {ident: {str(train_frames[int(index)]): record for index, record in track.items()}
                        for ident, track in result['tracks'].items()}
    result['anchor_frame'] = anchor_frame
    return result

def save_review_jpeg(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for quality in (88,80,72,64,56,48,40,32,24):
        image.save(path, 'JPEG', quality=quality, optimize=True)
        if path.stat().st_size <= 600000:
            return
    raise ValueError('review image exceeds 600 KB: '+str(path))

def mask_panel(source, tracks, frame, palette, font, *, labels=True):
    import numpy as np
    from PIL import Image, ImageDraw
    from pycocotools import mask as mu
    array = np.asarray(source,dtype=np.float32).copy()
    text = []
    for track in tracks:
        record = track['frames'].get(str(frame))
        if record is None:
            continue
        mask = mu.decode(record['rle']).astype(bool)
        color = palette[track['name']]
        array[mask] = 0.58*array[mask]+0.42*np.array(color)
        edge = mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
        array[edge] = color
        y,x = np.nonzero(mask)
        text.append((int(np.median(x)),int(np.median(y)),track['name'].replace('Character ','C').replace('Object ','O'),color))
    result = Image.fromarray(array.astype(np.uint8))
    if labels:
        draw = ImageDraw.Draw(result)
        for x,y,name,color in text:
            draw.text((x,y),name,font=font,fill=color,stroke_width=2,stroke_fill=(0,0,0))
    return result

def sam_session(predictor, directory, prompt, boxes=None, *, hotstart,
                anchor=0, direction='forward'):
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    predictor.model.hotstart_delay, predictor.model.hotstart_unmatch_thresh = hotstart
    sid = predictor.handle_request({'type': 'start_session', 'resource_path': str(directory),
                                     'offload_video_to_cpu': True})['session_id']
    tracks = {}
    try:
        expected = len(list(directory.glob('*.jpg')))
        if boxes is not None:
            # SAM3.1 660a5e9e 的 _build_sam2_output 在帧缓存不存在时
            # 直接返回 {}，丢弃已算出的实例提示传播结果。框角会话没有 VG
            # 预先填缓存，因此在提示前建立空帧槽；不注入掩码或文本检测。
            state=predictor._get_session(sid)['state']
            assert not state['cached_frame_outputs']
            state['cached_frame_outputs'].update({i:{} for i in range(state['num_frames'])})
        request = {'type': 'add_prompt', 'session_id': sid, 'frame_index': anchor}
        if boxes is None:
            request['text'] = prompt
            predictor.handle_request(request)
        else:
            for p in boxes:
                coordinates, labels = instance_prompt(p)
                predictor.handle_request({**request,'points':coordinates,
                    'point_labels':labels,'obj_id':p['id'],'rel_coordinates':True})
        # 上游 660a5e9e 把边界锚点的第二次传播当作缓存读取。
        # 末帧先向前只计算锚帧，因此先反向，再读取已缓存的锚帧。
        directions = ['backward', 'forward'] if direction == 'both' and anchor == expected-1 and expected > 1 else [direction]
        from itertools import chain
        outputs = chain.from_iterable(predictor.handle_stream_request({
            'type':'propagate_in_video', 'session_id':sid,
            'propagation_direction':part, 'start_frame_index':anchor}) for part in directions)
        seen = []
        for item in outputs:
            index, out = int(item['frame_index']), item['outputs']
            seen.append(index)
            for ident, prob, mask in zip(out['out_obj_ids'], out['out_probs'], out['out_binary_masks'], strict=True):
                mask = np.asarray(mask, dtype=np.uint8)
                if not mask.any():
                    continue
                rle = mask_utils.encode(np.asfortranarray(mask))
                area = int(mask_utils.area(rle))
                bbox = mask_utils.toBbox(rle).tolist()
                rle['counts'] = rle['counts'].decode('ascii')
                tracks.setdefault(str(int(ident)), {})[str(index)] = {'rle': rle, 'bbox_xywh': bbox,
                    'area': area, 'score': float(prob)}
        if sorted(set(seen)) != list(range(expected)):
            raise ValueError('incomplete SAM frame traversal')
        torch.cuda.synchronize()
        return {'tracks': tracks, 'seconds': time.monotonic()-start,
            'anchor_frame':anchor, 'propagation_direction':direction, 'propagation_order':directions,
            'frame_visits':len(seen),
            'hotstart':list(hotstart),
            'instance_cache_initialization':'empty_per_frame_slots_before_prompt' if boxes is not None else None,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'frames': expected}
    finally:
        predictor.handle_request({'type': 'close_session', 'session_id': sid})



def load_config(path):
    config = read(path)
    required = {
        'schema_version': int, 'output_root': str, 'status_root': str,
        'annotations': str, 'manifest': str,
        'coordinate_system': str, 'sample_frames': str,
        'image_size': list, 'train_fps': int, 'frame_stride': int, 'ffmpeg': str, 'jpeg_quality': int,
        'status_interval_seconds': int, 'disk_report_every_blocks': int, 'font': str,
        'sam': {'root': str, 'revision': str, 'weights': str, 'bpe': str,
                'version': str, 'max_objects': int, 'multiplex_count': int,
                'use_fa3': bool, 'use_rope_real': bool, 'compile': bool,
                'warm_up': bool, 'box_encoding': str,
                'hotstart': {'box':list, 'group':list}},
        'depth': {'root': str, 'revision': str, 'weights': str,
                  'model': str, 'resolution': int, 'mode': str, 'writers': int,
                  'inflight_blocks': int, 'max_pending_shots': int,
                  'video_codec': str, 'video_preset': str, 'video_qp': int,
                  'pixel_format': str, 'levels': int},
        'pose': {'weights': str, 'model_input_size': list, 'keypoints': int},
        'locate': {'model': str, 'endpoints': list, 'timeout_seconds': int, 'workers': int,
                   'enable_thinking': bool, 'reasoning_effort': str, 'temperature': float,
                   'top_p': float, 'top_k': int, 'min_p': float, 'presence_penalty': float,
                   'repetition_penalty': float, 'max_tokens': int}}
    def check(obj, spec, prefix=''):
        for key, typ in spec.items():
            value = obj[key]
            if isinstance(typ, dict):
                check(value, typ, prefix+key+'.')
            elif type(value) is not typ:
                raise TypeError(prefix+key+' has wrong type')
            elif typ is str and not value:
                raise ValueError(prefix+key+' is empty')
    required['execution'] = {'cpu_threads': int, 'stage_attempts': int}
    check(config, required)
    fixed = {'schema_version': 2, 'coordinate_system': 'normalized_0_1000',
             'sample_frames': 'first_middle_last', 'image_size': [832, 480], 'train_fps': 16}
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError('unsupported '+key)
    if config['frame_stride'] < 1:
        raise ValueError('invalid frame_stride')
    for session, values in config['sam']['hotstart'].items():
        if len(values) != 2 or any(type(v) is not int or v < 0 for v in values):
            raise ValueError('invalid hotstart for '+session)
    if config['sam']['box_encoding'] != 'official_instance_corner_labels_2_3':
        raise ValueError('unsupported box encoding')
    if config['depth']['resolution'] != 512 or config['depth']['mode'] != 'balanced':
        raise ValueError('VGGT-Omega deployment requires balanced 512')
    if config['depth']['video_qp'] != 0:
        raise ValueError('depth video must be lossless (video_qp 0)')
    if not 1 <= config['depth']['levels'] <= 254 or config['depth']['writers'] < 1:
        raise ValueError('invalid depth levels or writers')
    if any(config['depth'][k] < 1 for k in ('inflight_blocks', 'max_pending_shots')):
        raise ValueError('invalid depth pipeline capacity')
    if any(value < 1 for value in config['execution'].values()):
        raise ValueError('invalid execution capacity')
    if config['pose']['keypoints'] != 133:
        raise ValueError('RTMW requires 133 keypoints')
    for key in ('status_interval_seconds', 'disk_report_every_blocks'):
        if config[key] <= 0:
            raise ValueError('invalid '+key)
    from urllib.parse import urlsplit
    if not config['locate']['endpoints']:
        raise ValueError('empty local endpoints')
    for address in config['locate']['endpoints']:
        endpoint = urlsplit(address)
        if endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('local locator requires loopback HTTP endpoint')
    if config['locate']['workers'] < 1:
        raise ValueError('invalid locate workers')
    return config


def processable(record):
    """块能进空间层：登记表层合法（整块错误为空）——``ok``，或只是个别镜的记录不合法的 ``invalid``。"""
    return record['status'] in ('ok', 'invalid') and not record['block_errors']


def shot_problems(record, shot_count):
    """块里记录不合法的镜：{镜号（从 1 起）: 错误列表}。JSON 里镜号是字符串键。"""
    problems = {int(number): list(errors) for number, errors in record['shot_errors'].items()}
    if any(not 1 <= number <= shot_count or not errors for number, errors in problems.items()):
        raise ValueError('shot_errors do not match the shot count')
    return problems


def select_blocks(path, shard, count, *, keys=None, limit=0):
    """流式读 ``block-annotations.jsonl``，按文件顺序取可处理的块；再按 ``keys`` 过滤，截前 ``limit`` 块，最后 ``[shard::count]``。"""
    if count < 1 or not 0 <= shard < count or limit < 0:
        raise ValueError('invalid shard or limit')
    seen, matched, position, rows = set(), set(), 0, []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record['key']
            if key in seen:
                raise ValueError('duplicate block keys: '+key)
            seen.add(key)
            if not processable(record) or (keys is not None and key not in keys):
                continue
            matched.add(key)
            if (not limit or position < limit) and position % count == shard:
                rows.append({name: record[name] for name in
                             ('key', 'source', 'status', 'shot_count', 'shot_errors', 'annotation')})
            position += 1
    if keys is not None and keys - matched:
        raise ValueError(f'selected blocks are not processable or not in the annotations: {sorted(keys - matched)[:5]}')
    return rows


def read_blocks(path, keys):
    """流式读 ``blocks.jsonl``，只留要处理的块；缺一个就报错。"""
    found = {}
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row['key'] in keys:
                    found[row['key']] = row
    if set(found) != keys:
        raise ValueError(f'block list is missing selected blocks: {sorted(keys - set(found))[:5]}')
    return found


def stage_complete(block, stage):
    """阶段记录成功且产物在位。清理过（``stages/cleanup.json`` 成功）的块，取帧的图已按设计删掉，不再核对。"""
    path = block / 'stages' / (stage+'.json')
    if not path.exists():
        return False
    record = read(path)
    if record['state'] != 'succeeded':
        return False
    cleaned = stage != 'cleanup' and stage_complete(block, 'cleanup')
    return all(cleaned and p.startswith('frames/') or
               (block/p).is_file() and (block/p).stat().st_size > 0 for p in record['products'])


def store_depth(config, folder, fs, result, measured):
    """一镜的深度与置信度落盘：两个无损灰度视频（写完读回比对）和最后写的 ``metadata.json``。

    ``result`` 是模型原始输出：depth、conf [N,H,W]，intrinsic [N,3,3]，extrinsic [N,3,4]；``measured`` 是前向的耗时与显存峰值。
    """
    import numpy as np
    from mspc.data.depth_video import (encode_log_depth, encode_linear, write_verified,
                                       max_log_depth_error, max_linear_error)
    spec, levels = config['depth'], config['depth']['levels']
    tick = time.monotonic()
    depth_codes, near, far = encode_log_depth(result['depth'], levels)
    conf_codes, low, high = encode_linear(result['conf'], levels)
    storage = dict(frame_rate=f"{config['train_fps']}/{config['frame_stride']}", codec=spec['video_codec'],
                   preset=spec['video_preset'], qp=spec['video_qp'], pixel_format=spec['pixel_format'])
    write_verified(config['ffmpeg'], depth_codes, folder/'depth.mp4', **storage)
    write_verified(config['ffmpeg'], conf_codes, folder/'confidence.mp4', **storage)
    log_error, relative_error = max_log_depth_error(near, far, levels)
    count, height, width = depth_codes.shape
    save(folder/'metadata.json', {
        'model': spec['model'], 'mode': spec['mode'], 'resolution': spec['resolution'],
        'frames': count, 'frame_stride': config['frame_stride'], 'train_frames': fs['train_frames'],
        'native_size': [width, height], 'near': near, 'far': far,
        'confidence_min': low, 'confidence_max': high,
        'intrinsic': np.asarray(result['intrinsic']).tolist(), 'extrinsic': np.asarray(result['extrinsic']).tolist(),
        'storage': {'depth_video': 'depth.mp4', 'confidence_video': 'confidence.mp4', 'levels': levels,
                    'codec': spec['video_codec'], 'preset': spec['video_preset'], 'qp': spec['video_qp'],
                    'pixel_format': spec['pixel_format'], 'frame_rate': storage['frame_rate'],
                    'lossless_checked': 'decoded video equals the quantized array element by element'},
        'depth_encode': f'v = 1 + round({levels} * (ln d - ln near) / (ln far - ln near)); v = 1 everywhere when near == far; 0 = invalid',
        'depth_decode': f'd = exp(ln near + (v - 1) / {levels} * (ln far - ln near))',
        'confidence_encode': f'v = 1 + round({levels} * (c - min) / (max - min)); v = 1 everywhere when min == max; 0 = invalid',
        'confidence_decode': f'c = min + (v - 1) / {levels} * (max - min)',
        'max_quantization_error': {'ln_depth': log_error, 'relative_depth': relative_error,
                                   'confidence': max_linear_error(low, high, levels)},
        'unit': 'relative_scene_scale_not_verified_metric', 'cross_shot_comparable': False,
        'forward_seconds': measured['seconds'], 'store_seconds': time.monotonic()-tick,
        'peak_allocated_mib': measured['peak_allocated_mib'], 'peak_reserved_mib': measured['peak_reserved_mib']})


class Progress:
    def __init__(self, path, stage, shard, count, total, config, work):
        self.path, self.config, self.work = path, config, work
        self.tick = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.data = dict(stage=stage, shard=[shard, count], state='running', total=total,
                         completed=0, skipped=0, failed=0, output_bytes=0, current_block=None,
                         pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
        self.disk()
        self.write()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def disk(self):
        used = shutil.disk_usage(self.work)
        self.data['disk'] = dict(used_bytes=used.used, free_bytes=used.free,
                                measured_at=datetime.now(timezone.utc).isoformat())

    def add(self, **counts):
        with self.lock:
            for name, value in counts.items():
                self.data[name] += value

    def write(self):
        with self.lock:
            save(self.path, {**self.data, 'seconds': time.monotonic()-self.tick,
                            'updated_at': datetime.now(timezone.utc).isoformat()})

    def loop(self):
        while not self.stop.wait(self.config['status_interval_seconds']):
            self.write()

    def close(self, state):
        self.stop.set()
        self.thread.join()
        self.data['state'] = state
        self.disk()
        self.write()


class Pipeline:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.work = args.work or Path(config['output_root'])
        self.work.mkdir(parents=True, exist_ok=True)
        self.stage = 'sam-'+args.session if args.step == 'sam' else args.step
        keys = None
        if args.selection is not None:
            keys = {line.strip() for line in args.selection.read_text().splitlines() if line.strip()}
        self.rows = select_blocks(config['annotations'], *args.shard, keys=keys, limit=args.limit)
        self.manifest = {}
        if args.step == 'prepare':
            self.manifest = read_blocks(config['manifest'], {r['key'] for r in self.rows})
        self.models = {}
        self.model_load_seconds = 0.0

    def model(self, kind):
        if kind in self.models:
            return self.models[kind]
        import subprocess
        tick = time.monotonic()
        spec = self.config[kind]
        if kind in ('sam', 'depth'):
            checkout = Path(spec['root'])
            revision_file = checkout/'UPSTREAM_COMMIT'
            revision = revision_file.read_text().splitlines()[0].strip() if revision_file.exists() else subprocess.check_output(
                ['git', '-c', 'safe.directory='+str(checkout), '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            if revision != spec['revision']:
                raise ValueError(kind+' upstream revision differs: '+revision)
            sys.path.insert(0, str(checkout))
        if kind == 'sam':
            from sam3.model_builder import build_sam3_predictor
            model = build_sam3_predictor(checkpoint_path=spec['weights'], bpe_path=spec['bpe'],
                version=spec['version'], max_num_objects=spec['max_objects'], multiplex_count=spec['multiplex_count'],
                use_fa3=spec['use_fa3'], use_rope_real=spec['use_rope_real'], compile=spec['compile'], warm_up=spec['warm_up'])
        elif kind == 'depth':
            import torch
            from vggt_omega.models import VGGTOmega
            model = VGGTOmega().to('cuda').eval()
            model.load_state_dict(torch.load(spec['weights'], map_location='cpu'))
        else:
            if os.environ.get('LD_LIBRARY_PATH'):
                raise ValueError('RTMW requires LD_LIBRARY_PATH=')
            import torch  # 先加载当前环境的 CUDA 库。
            import onnxruntime as ort
            from rtmlib import RTMPose
            ort.preload_dlls()
            model = RTMPose(spec['weights'], model_input_size=tuple(spec['model_input_size']), backend='onnxruntime', device='cuda')
            if model.session.get_inputs()[0].shape[-2:] != list(reversed(spec['model_input_size'])):
                raise ValueError('RTMW ONNX input dimensions differ from pose.model_input_size')
            if 'CUDAExecutionProvider' not in model.session.get_providers():
                raise ValueError('RTMW CUDA provider unavailable')
        self.model_load_seconds += time.monotonic()-tick
        self.models[kind] = model
        return model

    def run(self):
        k, n = self.args.shard
        status_root = self.work/'status' if self.args.work else Path(self.config['status_root'])
        lock_path = self.work/'logs'/f'{self.stage}-s{k}of{n}.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            progress = Progress(status_root/f'spatial-{self.stage}-s{k}of{n}.json', self.stage, k, n,
                                len(self.rows), self.config, self.work)
            state = 'failed'
            try:
                if self.args.step == 'locate':
                    # 定位是在等服务答复，块与块互不依赖：同时在途若干个块，把服务的并发喂满。
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=self.config['locate']['workers']) as pool:
                        list(pool.map(lambda row: self.process_block(row, progress, status_root), self.rows))
                elif self.args.step == 'depth':
                    from concurrent.futures import ThreadPoolExecutor
                    spec = self.config['depth']
                    self.depth_gpu_lock = threading.Lock()
                    self.depth_slots = threading.BoundedSemaphore(spec['max_pending_shots'])
                    # 一个模型、一个前向流；两个块交替供给，让下块前向覆盖上块末尾的编码等待。
                    with ThreadPoolExecutor(max_workers=spec['writers']) as writers:
                        self.depth_writers = writers
                        with ThreadPoolExecutor(max_workers=spec['inflight_blocks']) as blocks:
                            list(blocks.map(lambda row: self.process_block(row, progress, status_root), self.rows))
                else:
                    for row in self.rows:
                        self.process_block(row, progress, status_root)
                state = 'completed'
            finally:
                progress.close(state)
        return 0 if state == 'completed' and not progress.data['failed'] else 1

    def process_block(self, row, progress, status_root):
        """一个块过当前阶段：已完成的跳过；成功或失败都落 ``stages/<阶段>.json``，失败不影响别的块。

        阶段方法返回产物路径列表；要在阶段记录里多写字段的（清理记删了多少字节）返回 ``(路径列表, 字段字典)``。
        """
        k, n = self.args.shard
        key = row['key']; block = self.work/key
        block.mkdir(parents=True, exist_ok=True)
        progress.data['current_block'] = key
        if stage_complete(block, self.stage):
            progress.add(skipped=1, completed=1)
            return
        tick = time.monotonic()
        previous_load = self.model_load_seconds
        try:
            if getattr(self.args, 'wait_for_prepare', False):
                while not stage_complete(block, 'prepare'):
                    upstream = block/'stages/prepare.json'
                    if upstream.exists() and read(upstream)['state'] == 'failed':
                        raise ValueError('prepare failed: '+read(upstream)['reason'])
                    status = read(status_root/f'spatial-prepare-s{k}of{n}.json')
                    if status['state'] != 'running':
                        raise ValueError('prepare stopped without this block')
                    updated = datetime.fromisoformat(status['updated_at'])
                    if (datetime.now(timezone.utc)-updated).total_seconds() > 120:
                        raise ValueError('prepare heartbeat stale')
                    time.sleep(self.config['status_interval_seconds'])
            # 取帧阶段从标注清单取标注并留一份副本；之后的阶段都读这份副本，同一个块前后用的标注一致。
            annotation = row['annotation'] if self.args.step == 'prepare' else read(block/'annotation.json')
            problems = shot_problems(row, len(annotation['shots']))
            for shot in annotation['shots']:
                if shot['shot'] in problems:
                    continue
                missing = set(holding_targets(shot)) - {entry['name'] for entry in shot['present']}
                if missing:
                    raise ValueError(f"shot {shot['shot']} holding targets not present: {sorted(missing)}")
            outcome = getattr(self, self.args.step)(row, annotation)
            products, notes = outcome if isinstance(outcome, tuple) else (outcome, {})
            files = sorted({str(p.relative_to(block)) for p in products})
            size = sum((block/p).stat().st_size for p in files)
            result = dict(state='succeeded', stage=self.stage, key=key, products=files,
                          seconds=time.monotonic()-tick, output_bytes=size,
                          model_load_seconds=self.model_load_seconds-previous_load,
                          completed_at=datetime.now(timezone.utc).isoformat())
            result.update(notes)
            save(block/'stages'/(self.stage+'.json'), result)
            progress.add(completed=1, output_bytes=size)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as error:
            result = dict(state='failed', stage=self.stage, key=key, seconds=time.monotonic()-tick,
                          error_type=type(error).__name__, reason=str(error), traceback=traceback.format_exc(),
                          failed_at=datetime.now(timezone.utc).isoformat())
            save(block/'stages'/(self.stage+'.json'), result)
            progress.add(failed=1)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if (progress.data['completed']+progress.data['failed']) % self.config['disk_report_every_blocks'] == 0:
            progress.disk()
        progress.write()

    def prepare(self, row, annotation):
        from PIL import Image
        from mspc.data.frames import center_crop, iter_scene_frames, shot_grid
        key = row['key']; block = self.work/key
        scene = self.manifest[key]['scene']
        if len(annotation['shots']) != len(scene['shots']):
            raise ValueError('annotation shot count differs from manifest')
        problems = shot_problems(row, len(scene['shots']))
        stride = self.config['frame_stride']
        targets, shots = {}, []
        for i, shot in enumerate(scene['shots']):
            train = training_frames(shot['render_frames'], stride)
            grid = shot_grid(shot['start_frame'], shot['end_frame'], shot['render_frames'])
            source = [grid[t] for t in train]
            sid = f's{i+1:02d}'; directory = block/'frames'/sid
            directory.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(source):
                targets.setdefault(frame, []).append(directory/f'{j:06d}.jpg')
            shots.append(dict(shot=i+1, id=sid, render_frames=shot['render_frames'], train_frames=train,
                              source_frames=source, frames=len(train), annotation_errors=problems.get(i+1, [])))
        for number, array in iter_scene_frames(scene, sorted(targets), ffmpeg=self.config['ffmpeg'],
                                               error_path=self.work/'logs'/f'{key}-decode.stderr', video_decode_num_threads=1):
            image = center_crop(Image.fromarray(array).crop(scene['crop_box']))
            if list(image.size) != self.config['image_size']:
                raise ValueError('frame geometry differs')
            for path in targets[number]:
                image.save(path, quality=self.config['jpeg_quality'])
        files = [p for paths in targets.values() for p in paths]
        if any(not p.exists() for p in files):
            raise ValueError('incomplete extraction')
        save(block/'frames.json', dict(key=key, source=row['source'], frame_stride=stride, train_fps=self.config['train_fps'],
                                       annotation_status=row['status'], shots=shots, frames=sum(s['frames'] for s in shots)))
        save(block/'annotation.json', annotation)
        save(block/'source.json', self.manifest[key])
        return files+[block/p for p in ('frames.json', 'annotation.json', 'source.json')]
    def locate(self, row, annotation):
        block = self.work/row['key']; frames = read(block/'frames.json')
        registry = {e['name']: e for e in annotation['characters']+annotation['objects']}
        from concurrent.futures import ThreadPoolExecutor
        # 记录不合法的镜不定位。
        pairs = [(shot, fs) for shot, fs in zip(annotation['shots'], frames['shots'], strict=True) if not fs['annotation_errors']]
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=len(pairs)) as pool:  # 一块的各镜同时发；同时在途几个块由 locate.workers 定
            futures = [pool.submit(self.locate_shot, block, registry, shot, fs) for shot, fs in pairs]
            return [future.result() for future in futures]

    def locate_shot(self, block, registry, shot, fs):
        target = block/'locations'/(fs['id']+'.json')
        names = [e['name'] for e in shot['present'] if registry[e['name']].get('kind') != 'group']
        positions = sampled_frames(fs['train_frames'])
        if target.exists():
            validate_locations(read(target), names, positions)
            return target
        if not names:
            save(target, {'entities':[], 'sampled_frames':positions,
                          'coordinate_system':self.config['coordinate_system']})
            return target
        entities = [{**registry[e['name']], **e} for e in shot['present'] if e['name'] in names]
        prompt = (
            f'These chronological images are from ONE shot, actual zero-based frame indices {positions}. '
            'Locate EVERY supplied named entity using appearance and current clothing. For each image give '
            'visibility: clear, absent, or unsure. The visibility field is an array of exactly '
            f'{len(positions)} labels, one per image in the same order as frame indices {positions}; '
            'do not return a single overall label. clear means identity and visible extent are distinguishable. '
            'Choose the EARLIEST sampled frame marked clear; frame_index must match that array entry. '
            'Return bbox_2d [x1,y1,x2,y2] tightly enclosing '
            'the WHOLE visible instance including visible limbs, not only head or clothing. Do not hallucinate '
            'occluded extent. Use a normalized image canvas: x=0 at the left edge and x=1000 at the right edge; '
            'y=0 at the top edge and y=1000 at the bottom edge. Estimate bbox_2d directly on '
            'this normalized canvas. Do not estimate native pixel dimensions or convert from '
            'pixel coordinates. A position halfway across the image is x=500; halfway down is '
            'y=500. Check the box against the named visible instance in the selected image. '
            'If none is clear, frame_index and bbox_2d must be null. Return JSON entities, one per name, '
            'with name, visibility, frame_index, bbox_2d, reason. Entities: '+json.dumps(entities))
        properties = {'name': {'type':'string','enum':names},
                      'visibility': {'type':'array', 'minItems':len(positions), 'maxItems':len(positions),
                                     'items':{'type':'string','enum':['clear','absent','unsure']}},
                      'frame_index': {'type':['integer','null'], 'enum':positions+[None]},
                      'bbox_2d': {'type':['array','null'], 'minItems':4, 'maxItems':4,
                                  'items':{'type':'number','minimum':0,'maximum':1000}}, 'reason': {'type':'string'}}
        schema = {'type':'object','additionalProperties':False,'required':['entities'], 'properties':{
            'entities': {'type':'array', 'minItems':len(names), 'maxItems':len(names),
                         'items': {'type':'object','additionalProperties':False,
                                                'required':list(properties),'properties':properties}}}}
        content = []
        for pos in positions:
            path = frame_file(block, fs, pos)
            content.extend([{'type':'text','text':f'Frame index {pos}.'}, {'type':'image_url','image_url':{
                'url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()}}])
        content.append({'type':'text','text':prompt})
        spec = self.config['locate']
        body = {key:spec[key] for key in ('model','temperature','top_p','top_k','min_p',
                'presence_penalty','repetition_penalty','max_tokens')}
        # vLLM 0.18 的顶层枚举不收 xhigh；直接交给官方模型模板。
        body.update(chat_template_kwargs={'enable_thinking':spec['enable_thinking'],
                    'reasoning_effort':spec['reasoning_effort'], 'preserve_thinking':True},
                    response_format={'type':'json_schema','json_schema':{
                        'name':'spatial_locations','strict':True,'schema':schema}},
                    messages=[{'role':'user','content':content}])
        tag = block.name+'-'+fs['id']+'-locations'
        response = self.work/'responses'/(tag+'.json')
        # locations 是通过校验的成功产物；外置 responses 仅留诊断，块重建后不能据此复用旧绑定。
        answer = self.client.call(tag, body)
        try:
            obj = json.loads(answer)
            validate_locations(obj, names, positions)
        except ValueError:
            # 不合约定的答案挪到一边留档：重跑时让模型重答，而不是再读回同一条坏答案。
            (self.work/'responses-invalid').mkdir(exist_ok=True)
            response.replace(self.work/'responses-invalid'/response.name)
            raise
        save(target, {'entities':obj['entities'], 'sampled_frames':positions,
                      'coordinate_system':self.config['coordinate_system']})
        return target

    def sam(self, row, annotation):
        import torch
        block = self.work/row['key']; frames = read(block/'frames.json')
        registry = {e['name']: e for e in annotation['characters']+annotation['objects']}
        sessions = ['box', 'group'] if self.args.session == 'all' else [self.args.session]
        files = {session: [] for session in sessions}
        seconds = {session: 0.0 for session in sessions}
        for shot, fs in zip(annotation['shots'], frames['shots'], strict=True):
            if fs['annotation_errors']:
                continue  # 记录不合法的镜不分割
            folder = block/'sam_raw'/fs['id']; directory = block/'frames'/fs['id']
            train = fs['train_frames']
            pending = []
            for session in sessions:
                jobs = []
                if session == 'box':
                    entries = read(block/'locations'/(fs['id']+'.json'))['entities']
                    entries = [e for e in entries if registry[e['name']].get('kind') != 'group']
                    validate_locations({'entities': entries}, [e['name'] for e in shot['present']
                        if registry[e['name']].get('kind') != 'group'], sampled_frames(train))
                    visible = [e for e in entries if e['frame_index'] is not None]
                    for anchor in sorted({e['frame_index'] for e in visible}):  # 锚点是训练帧号
                        anchored = [e for e in visible if e['frame_index'] == anchor]
                        for offset in range(0, len(anchored), self.config['sam']['max_objects']):
                            batch = anchored[offset:offset+self.config['sam']['max_objects']]
                            boxes = [dict(id=i+1, bbox_2d=e['bbox_2d']) for i, e in enumerate(batch)]
                            jobs.append((f'box-{anchor:06d}-{offset:03d}', None, boxes, anchor, batch))
                elif session == 'group':
                    for entry in shot['present']:
                        entity = registry[entry['name']]
                        if entity.get('kind') == 'group':
                            jobs.append(('group-'+entity['name'].replace(' ','_'), entity['appearance'], None, 0, [entity]))
                for label, prompt, boxes, anchor, entities in jobs:
                    target = folder/(label+'.json'); files[session].append(target)
                    if not target.exists():
                        pending.append((session, label, prompt, boxes, anchor, entities, target))
            if not pending:
                continue
            model = self.model('sam')
            for session, label, prompt, boxes, anchor, entities, target in pending:
                tick = time.monotonic()
                hotstart = tuple(self.config['sam']['hotstart'][session])
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    result = sam_session(model, directory, prompt, boxes, hotstart=hotstart,
                                         anchor=train.index(anchor), direction='both' if boxes is not None else 'forward')
                result = in_training_frames(result, train, anchor)
                result.update(session=session, prompt=prompt, entities=entities)
                if prompt is not None:
                    from sam3.model.tokenizer_ve import SimpleTokenizer
                    tokenizer = SimpleTokenizer(bpe_path=self.config['sam']['bpe'], context_length=32)
                    tokens = tokenizer.encode(prompt)
                    result.update(prompt_tokens_with_special=len(tokens)+2, prompt_truncated=len(tokens)+2>32,
                                  effective_prompt=tokenizer.decode(tokens[:30]))
                save(target, result)
                seconds[session] += time.monotonic()-tick
                print(json.dumps({'session':row['key']+'/'+fs['id']+'/'+label, 'seconds':result['seconds']}), flush=True)
        for session in sessions:
            # 无 group / 无清楚可见实例也是完成，而不是上游缺失。
            index = block/'sam_raw'/(session+'-index.json')
            save(index, {'session':session, 'files':[str(p.relative_to(block)) for p in files[session]]})
            files[session].append(index)
            if self.args.session == 'all':
                save(block/'stages'/('sam-'+session+'.json'), dict(
                    state='succeeded', stage='sam-'+session, key=row['key'],
                    products=[str(p.relative_to(block)) for p in files[session]],
                    seconds=seconds[session], timing_scope='new_session_calls_including_close_and_write',
                    model_load_seconds=0.0, output_bytes=sum(p.stat().st_size for p in files[session]),
                    completed_at=datetime.now(timezone.utc).isoformat()))
        return [p for group in files.values() for p in group]

    def depth_infer(self, block, fs):
        """一镜的 GPU 前向；调用方串行进入，返回 CPU 数组供后台写盘。"""
        import numpy as np
        import torch
        model = self.model('depth')
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
        paths = [block/'frames'/fs['id']/f'{i:06d}.jpg' for i in range(fs['frames'])]
        torch.cuda.reset_peak_memory_stats(); tick = time.monotonic()
        images = load_and_preprocess_images([str(p) for p in paths], mode=self.config['depth']['mode'],
                                           image_resolution=self.config['depth']['resolution']).to('cuda')
        with torch.inference_mode():
            predictions = model(images)
            extrinsic, intrinsic = encoding_to_camera(predictions['pose_enc'], predictions['images'].shape[-2:])
        dep = predictions['depth'][0].float().cpu().numpy()
        if dep.shape[-1] == 1:
            dep = dep[..., 0]
        result = {'depth':dep, 'conf':predictions['depth_conf'][0].float().cpu().numpy(),
                  'intrinsic':intrinsic[0].float().cpu().numpy(), 'extrinsic':extrinsic[0].float().cpu().numpy()}
        torch.cuda.synchronize()
        measured = {'seconds':time.monotonic()-tick, 'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        if dep.shape[0] != len(paths) or not all(np.isfinite(v).all() for v in result.values()) or np.any(dep <= 0):
            raise ValueError('invalid native depth/camera/confidence')
        return result, measured

    def depth(self, row, annotation):
        block = self.work/row['key']; frames = read(block/'frames.json'); files = []
        pending = []
        load_seconds = 0.0
        try:
            with self.depth_gpu_lock:
                previous_load = self.model_load_seconds
                for fs in frames['shots']:
                    folder = block/'depth'/fs['id']; folder.mkdir(parents=True, exist_ok=True)
                    expected = [folder/'depth.mp4', folder/'confidence.mp4', folder/'metadata.json']
                    files.extend(expected)
                    if all(p.exists() and p.stat().st_size > 0 for p in expected):
                        continue
                    # 获得空位才算下一镜，限制所有块合计的 CPU 数组数，长块也不能无限排队。
                    self.depth_slots.acquire()
                    try:
                        result, measured = self.depth_infer(block, fs)
                        (folder/'metadata.json').unlink(missing_ok=True)
                        write = self.depth_writers.submit(store_depth, self.config, folder, fs, result, measured)
                    except BaseException:
                        self.depth_slots.release()
                        raise
                    write.add_done_callback(lambda _: self.depth_slots.release())
                    pending.append(write)
                    del result
                load_seconds = self.model_load_seconds-previous_load
        finally:
            # 离开前向锁后等写盘，下一个块已经能用 GPU。所有写线程收齐后才报告本块成功/失败。
            error = None
            for write in pending:
                try:
                    write.result()
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
        return files, {'model_load_seconds': load_seconds}

    def assemble(self, row, annotation):
        import numpy as np
        from PIL import Image
        from pycocotools import mask as mu
        from mspc.data.depth_video import decode_log_depth, read_gray_video
        block = self.work/row['key']; frames = read(block/'frames.json')
        registry = {e['name']:e for e in annotation['characters']+annotation['objects']}
        for stage in ('sam-box', 'sam-group', 'depth'):
            if not stage_complete(block, stage):
                raise ValueError('incomplete producer: '+stage)
        masks = {'schema_version':2, 'width':832, 'height':480, 'rle_format':'COCO compressed, Fortran order',
                 'frame_numbers':'training frame within the shot', 'shots':{}}
        instances = {'registry':registry, 'cross_shot_identity':'annotation names only', 'shots':{}}
        medians, error = {}, {}
        for shot, fs in zip(annotation['shots'], frames['shots'], strict=True):
            sid = fs['id']; folder = block/'sam_raw'/sid
            metadata = read(block/'depth'/sid/'metadata.json')
            error[sid] = metadata['max_quantization_error']['relative_depth']
            tracks, bindings = {}, {}
            if not fs['annotation_errors']:
                locations = {e['name']:e for e in read(block/'locations'/(sid+'.json'))['entities']}
                named = {}
                for path in sorted(folder.glob('box-*.json')):
                    raw = read(path)
                    for i, entry in enumerate(raw['entities']):
                        ident = str(i+1)
                        named[entry['name']] = (path, ident, raw['tracks'].get(ident))
                for entity in shot['present']:
                    name = entity['name']; kind = registry[name].get('kind','object'); tids = []
                    if kind == 'group':
                        path = folder/('group-'+name.replace(' ','_')+'.json'); raw = read(path)
                        selected = raw['tracks']; status = 'group_phrase_authority'; anchor = None
                        prefix = 'group-'+name.replace(' ','_')
                    else:
                        anchor = locations[name]['frame_index']; prefix = 'box-'+name.replace(' ','_')
                        if anchor is None:
                            selected = {}; status = 'not_clearly_visible_in_samples'
                        else:
                            path, ident, track = named[name]
                            selected = {ident:track} if track is not None else {}
                            status = 'assigned' if selected else 'no_mask'
                    for ident, track in selected.items():
                        tid = prefix+':'+ident; tids.append(tid)
                        tracks[tid] = dict(name=name, kind=kind, frames=track,
                                           origin=str(path.relative_to(self.work)), raw_track=ident)
                    visible = set().union(*(set(tracks[t]['frames']) for t in tids))
                    bindings[name] = dict(tracks=tids, status=status, kind=kind, candidate_ids=list(selected),
                        frames_with_mask=len(visible), frame_fraction=len(visible)/fs['frames'],
                        last_frame_mask=str(fs['train_frames'][-1]) in visible, anchor_frame=anchor)
            masks['shots'][sid] = dict(tracks=tracks, train_frames=fs['train_frames'], source_frames=fs['source_frames'],
                                       annotation_errors=fs['annotation_errors'])
            instances['shots'][sid] = bindings
            medians[sid] = {tid:{'name':v['name'], 'median_depth':{}} for tid,v in tracks.items()}
            if not tracks:
                continue
            width, height = metadata['native_size']
            codes = read_gray_video(self.config['ffmpeg'], block/'depth'/sid/metadata['storage']['depth_video'],
                                    (metadata['frames'], height, width))
            if len(codes) != fs['frames'] or metadata['train_frames'] != fs['train_frames']:
                raise ValueError('depth frames differ from extracted frames')
            for frame, code_map in zip(fs['train_frames'], codes, strict=True):
                wanted = [(tid, track) for tid, track in tracks.items() if str(frame) in track['frames']]
                if not wanted:
                    continue
                aligned = np.asarray(Image.fromarray(code_map).resize(tuple(self.config['image_size']), Image.Resampling.NEAREST))
                for tid, track in wanted:
                    mask = mu.decode(track['frames'][str(frame)]['rle']).astype(bool)
                    depth = decode_log_depth(aligned[mask], metadata['near'], metadata['far'], metadata['storage']['levels'])
                    medians[sid][tid]['median_depth'][str(frame)] = float(np.median(depth))
        products = {'masks.json':masks, 'instances.json':instances,
                    'instance_depth.json':{'unit':'relative_per_shot',
                                           'sampling':'depth.mp4 decoded and dequantized; code map nearest-resized to the mask grid; median over the mask',
                                           'precision':'8-bit log-depth quantization; relative error of each decoded depth is at most max_relative_error[shot]',
                                           'max_relative_error':error, 'shots':medians}}
        for name, data in products.items():
            save(block/name, data)
        return [block/name for name in products]
    def pose(self, row, annotation):
        import numpy as np
        from PIL import Image
        block = self.work/row['key']
        if not stage_complete(block, 'assemble'):
            raise ValueError('assemble is incomplete')
        masks = read(block/'masks.json')
        frames = read(block/'frames.json')
        model = self.model('pose')
        result = {'keypoint_order':'COCO-WholeBody 133', 'coordinates':'832x480 pixels',
                  'frame_numbers':'training frame within the shot',
                  'score':'SimCC raw peak, not probability', 'input_color':'RGB per exported SDK',
                  'acceptance':POSE_CAVEAT, 'shots':{}, 'timings':{}}
        for fs in frames['shots']:
            sid = fs['id']; shot = masks['shots'][sid]
            tick = time.monotonic(); out = {}
            people = {t:v for t,v in shot['tracks'].items() if
                v['kind'] in ('person','group')}
            for frame in fs['train_frames']:
                active = [t for t,v in people.items() if str(frame) in v['frames']]
                if not active:
                    continue
                img = np.asarray(Image.open(frame_file(block, fs, frame)).convert('RGB'))
                boxes = []
                for tid in active:
                    x,y,w,h = people[tid]['frames'][str(frame)]['bbox_xywh']; boxes.append([x,y,x+w,y+h])
                points, scores = model(img, bboxes=np.asarray(boxes, dtype=np.float32))
                if points.shape != (len(active),133,2) or scores.shape != (len(active),133):
                    raise ValueError('wrong RTMW output shape')
                if not np.isfinite(points).all() or not np.isfinite(scores).all():
                    raise ValueError('nonfinite RTMW output')
                for j,tid in enumerate(active):
                    out.setdefault(tid, {'name':people[tid]['name'], 'frames':{}})['frames'][str(frame)] = {
                        'keypoints':points[j].tolist(), 'scores':scores[j].tolist()}
            result['shots'][sid] = out; result['timings'][sid] = time.monotonic()-tick
        target = block/'pose.json'; save(target, result); return [target]

    def audit(self, row, annotation):
        import numpy as np
        from pycocotools import mask as mu
        from mspc.data.depth_video import read_gray_video
        block = self.work/row['key']
        for stage in ('prepare','sam-box','sam-group','depth','assemble'):
            if not stage_complete(block, stage):
                raise ValueError('incomplete '+stage)
        frames = read(block/'frames.json'); masks = read(block/'masks.json')['shots']
        instances = read(block/'instances.json')['shots']; medians = read(block/'instance_depth.json')['shots']
        rles = 0
        for fs, shot in zip(frames['shots'], annotation['shots'], strict=True):
            sid = fs['id']; tracks = masks[sid]['tracks']
            if fs['annotation_errors']:
                if tracks or instances[sid] or medians[sid]:
                    raise ValueError('a shot with annotation errors has instance products')
            elif set(instances[sid]) != {e['name'] for e in shot['present']}:
                raise ValueError('binding names differ')
            for binding in instances[sid].values():
                if any(t not in tracks for t in binding['tracks']):
                    raise ValueError('binding points at missing track')
            for tid, track in tracks.items():
                if set(medians[sid][tid]['median_depth']) != set(track['frames']):
                    raise ValueError('median frame set differs')
                for frame, rec in track['frames'].items():
                    if int(frame) not in fs['train_frames']:
                        raise ValueError('mask frame is not an extracted training frame')
                    decoded = mu.decode(rec['rle'])
                    if decoded.shape != (480,832) or int(decoded.sum()) != rec['area']:
                        raise ValueError('RLE area/shape differs')
                    if not np.allclose(mu.toBbox(rec['rle']), rec['bbox_xywh']):
                        raise ValueError('RLE box differs')
                    if not np.isfinite(medians[sid][tid]['median_depth'][frame]):
                        raise ValueError('nonfinite median')
                    rles += 1
            folder = block/'depth'/sid; metadata = read(folder/'metadata.json')
            if metadata['train_frames'] != fs['train_frames'] or metadata['frames'] != fs['frames']:
                raise ValueError('depth frames differ from extracted frames')
            width, height = metadata['native_size']
            storage = metadata['storage']
            if not 0 < metadata['near'] <= metadata['far'] or not metadata['confidence_min'] <= metadata['confidence_max']:
                raise ValueError('invalid depth or confidence range')
            for name in ('intrinsic', 'extrinsic'):
                values = np.asarray(metadata[name])
                if len(values) != fs['frames'] or not np.isfinite(values).all():
                    raise ValueError('invalid depth camera output')
            for name in ('depth_video', 'confidence_video'):
                codes = read_gray_video(self.config['ffmpeg'], folder/storage[name], (fs['frames'], height, width))
                if codes.min() < 1 or codes.max() > storage['levels']+1:
                    raise ValueError('quantized code outside 1..levels+1: '+name)
        path = block/'audit.json'
        save(path, {'state':'passed','rles':rles,'shots':len(frames['shots']),
                    'meaning':'结构与数值检查；不代表掩码身份正确'})
        return [path]

    def review(self, row, annotation):
        """登记实例锚点：同名跨镜同色，人物暖色、物体冷色。"""
        from PIL import Image, ImageDraw, ImageFont
        block = self.work/row['key']; frames = read(block/'frames.json'); products = []
        if not stage_complete(block, 'assemble'):
            raise ValueError('assemble is incomplete')
        masks = read(block/'masks.json')['shots']
        instances = read(block/'instances.json')
        font = ImageFont.truetype(self.config['font'], 18)
        warm = [(230,70,60),(245,150,40),(225,60,150),(190,90,30),(250,200,50),(200,40,90)]
        cool = [(40,140,230),(40,190,170),(110,90,230),(60,200,90),(20,110,140),(150,170,240)]
        palette = {}
        for entries, colors in ((annotation['characters'], warm), (annotation['objects'], cool)):
            palette.update({entry['name']:colors[i % len(colors)] for i, entry in enumerate(entries)})
        for fs in frames['shots']:
            if fs['annotation_errors']:
                continue
            bindings = instances['shots'][fs['id']]
            tracks = [masks[fs['id']]['tracks'][tid]
                      for binding in bindings.values() for tid in binding['tracks']]
            positions = sampled_frames(fs['train_frames'])
            width = 832*len(positions)
            # 外貌完整换行，包含未分出掩码的登记名字。
            measure = ImageDraw.Draw(Image.new('RGB', (1,1)))
            legend = []
            for name, binding in bindings.items():
                entry = instances['registry'][name]
                line = name + ': ' + entry['appearance']
                if not binding['tracks']:
                    line += ' [' + binding['status'] + ']'
                part = ''
                for char in line:
                    if measure.textlength(part+char, font=font) > width-44:
                        legend.append((name, part)); part = ''
                    part += char
                legend.append((name, part))
            sheet = Image.new('RGB',(width,520+26*len(legend)),(25,25,25))
            draw = ImageDraw.Draw(sheet)
            for i, frame in enumerate(positions):
                source = Image.open(frame_file(block, fs, frame)).convert('RGB')
                active = sorted(tracks, key=lambda t:-t['frames'].get(str(frame), {'area':0})['area'])
                sheet.paste(mask_panel(source, active, frame, palette, font),(832*i,30))
                draw.text((832*i+8,5),f"{row['key']} / {fs['id']} / frame {frame}",font=font,fill='white')
            for i, (name, line) in enumerate(legend):
                y = 518+26*i
                draw.rectangle((8,y+4,24,y+20),fill=palette[name])
                draw.text((32,y),line,font=font,fill='white')
            path = block/'review'/(fs['id']+'-anchors.jpg')
            save_review_jpeg(sheet,path); products.append(path)
        return products

    def cleanup(self, row, annotation):
        """装配（跑了审计则审计也）成功之后删掉用完的取帧；源片可以重取。记下删了多少字节。"""
        block = self.work/row['key']
        if not stage_complete(block, 'assemble'):
            raise ValueError('assemble is incomplete')
        if (block/'stages/audit.json').exists() and not stage_complete(block, 'audit'):
            raise ValueError('audit did not succeed')
        frames = block/'frames'
        files = [p for p in frames.rglob('*') if p.is_file()] if frames.exists() else []
        deleted = sum(p.stat().st_size for p in files)
        if frames.exists():
            shutil.rmtree(frames)
        return [], {'deleted_bytes': deleted, 'deleted_files': len(files)}


def select_prepared(annotations, work, out):
    """可处理且取帧阶段已成功（``<work>/<键>/stages/prepare.json`` 的 state 是 succeeded）的块，键按标注文件的顺序一行一个写到 ``out``（原子写）。"""
    keys, seen, counts = [], set(), dict(annotation_records=0, processable=0)
    with Path(annotations).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record['key']
            if key in seen:
                raise ValueError('duplicate block keys: '+key)
            seen.add(key)
            counts['annotation_records'] += 1
            if not processable(record):
                continue
            counts['processable'] += 1
            stage = Path(work)/key/'stages'/'prepare.json'
            if stage.exists() and read(stage)['state'] == 'succeeded':
                keys.append(key)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix+'.partial')
    tmp.write_text(''.join(key+'\n' for key in keys))
    tmp.replace(out)
    return dict(counts, selected=len(keys), out=str(out))


def select_main(argv):
    parser = argparse.ArgumentParser(prog='spatial_layer.py select', description=select_prepared.__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='键清单文件，一行一个键')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    print(json.dumps(select_prepared(config['annotations'], config['output_root'], args.out), ensure_ascii=False))
    return 0


def pending_main(argv):
    """只数本次选集、分片里尚未完成的块；与阶段执行共用选择与完成判定。"""
    parser = argparse.ArgumentParser(description=pending_main.__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--shard', nargs=2, type=int, required=True, metavar=('K', 'N'))
    parser.add_argument('--stage', choices=['locate'], required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    keys = {line.strip() for line in args.selection.read_text().splitlines() if line.strip()}
    rows = select_blocks(config['annotations'], *args.shard, keys=keys)
    print(sum(not stage_complete(Path(config['output_root'])/row['key'], args.stage) for row in rows))
    return 0


def main():
    if sys.argv[1:2] == ['pending']:
        return pending_main(sys.argv[2:])
    if sys.argv[1:2] == ['select']:
        return select_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['prepare','locate','sam','depth','assemble','pose','audit','review','cleanup'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--shard', nargs=2, type=int, required=True, metavar=('K','N'))
    parser.add_argument('--session', choices=['box','group','all'])
    parser.add_argument('--work', type=Path, help='显式的独立验证输出根；状态也写入此根')
    parser.add_argument('--wait-for-prepare', action='store_true', help='定位等待同分片取帧；上游失败或心跳过期则记失败')
    parser.add_argument('--selection', type=Path, help='只跑这些块：键清单文件，一行一个键；不给就跑全部可处理的块')
    parser.add_argument('--limit', type=int, default=0, help='先截可处理块清单的前 N 块，再分片；0 为全部')
    args = parser.parse_args(); config = load_config(args.config)
    if (args.step == 'sam') != (args.session is not None):
        parser.error('only sam requires --session')
    if args.wait_for_prepare and args.step != 'locate':
        parser.error('--wait-for-prepare is only valid for locate')
    pipeline = Pipeline(args, config)
    # 输出根锁仅保护配置登记；不同阶段、不同块允许并行。
    with (pipeline.work/'config.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        snapshot = pipeline.work/'config.json'
        # 定位的并发块数只决定同时在途几个请求，不改任何请求的内容与产物，调它不要求换输出根。
        # 同理：深度的写盘线程数只决定怎么并行，按块落盘的产物不变。
        produced = lambda value: {**{k: v for k, v in value.items() if k != 'execution'},
                                  'locate': {k: v for k, v in value['locate'].items() if k != 'workers'},
                                  'depth': {k: v for k, v in value['depth'].items() if k not in ('writers', 'inflight_blocks', 'max_pending_shots')}}
        if snapshot.exists() and produced(read(snapshot)) != produced(config):
            raise ValueError('config changed: use a new output root')
        save(snapshot, config)
    if args.step == 'locate':
        pipeline.client = LocalClient(pipeline.work, config, args.shard[0])
    return pipeline.run()


if __name__ == '__main__':
    raise SystemExit(main())
