"""块流程打分用的两个后端：DOVER 命名头与 LAION 美学分（官方 simple_inference 的线性头）。"""
from __future__ import annotations

from pathlib import Path

from mspc.data.quality_scorer.adapters.dover import DoverBackend, DoverSettings


def dover_backend(cfg,device):
    d=cfg['dover']
    return DoverBackend(DoverSettings(source_root=Path(d['source_root']),weight_path=Path(d['weight_path']),
        eval_yaml=Path(d['eval_yaml']),seed=cfg['seed'],device=device,declared_sampling=d['sampling']))


def laion_import(cfg):
    import sys
    path=cfg['laion']['clip_source_root']
    if path not in sys.path:sys.path.insert(0,path)
    import clip
    return clip


class LaionBackend:
    def __init__(self,cfg,device):
        import torch
        from torch import nn
        clip=laion_import(cfg)
        self.torch=torch;self.device=device
        self.model,self.preprocess=clip.load(cfg['laion']['clip_weight_path'],device=device,jit=False)
        # 官方 simple_inference.py 的 MLP；Dropout 在 eval 下恒等，无激活层。
        self.head=nn.Sequential(nn.Linear(768,1024),nn.Dropout(.2),nn.Linear(1024,128),nn.Dropout(.2),
            nn.Linear(128,64),nn.Dropout(.1),nn.Linear(64,16),nn.Linear(16,1)).to(device).eval()
        weights=torch.load(cfg['laion']['head_weight_path'],map_location='cpu',weights_only=True)
        self.head.load_state_dict({k.removeprefix('layers.'):v for k,v in weights.items()})
        self.model.eval()

    def score(self,tensors):
        import numpy as np
        with self.torch.inference_mode():
            features=self.model.encode_image(self.torch.stack(tensors).to(self.device))
            # 与官方一致：CLIP CUDA fp16 → numpy 原 dtype L2 归一化 → float32 线性头。
            a=features.cpu().numpy();norm=np.atleast_1d(np.linalg.norm(a,2,axis=-1));norm[norm==0]=1
            emb=a/np.expand_dims(norm,-1)
            return self.head(self.torch.from_numpy(emb).to(self.device).float()).flatten().cpu().tolist()
