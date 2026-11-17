#!/usr/bin/env python3
"""训练参考放大裁剪表（:mod:`mspc.data.reference_crops`）：按训练配置的同一抽签，读 SAM 给每张参考写框；仅 CPU。

裁剪框包住参考帧里全部登记人物，与读者范围（``attention_policy``）、参考行写法（``row_policy``）和窗内读法无关，
所以抽签相同的各组训练（定稿、参考不带身份行、不路由、窗内全互读）共用一份表，用定稿配置生产即可。
与参考身份标签同一构造：去掉待产的标签、裁剪表、文字准入与负载上限后建数据集；目标缓存尚未发布时用
``--metadata-vae-geometry`` 只构造抽签元数据。``--sample-indices`` 给 JSON 列表（例如先跑约 200 行真实样本），
不给就写全表。摘要写到输出同名 ``.summary.json``，含「抽中要裁但人物框太大包不住、不裁」的比例。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mspc.config import load_experiment_config
from mspc.core.wan22_contract import MEMORY_ROW_POLICY_FIXED_ROLE_V1
from mspc.data.reference_crops import build_reference_crops
from mspc.training.block_dataset import BlockDataset
from mspc.training.training_args import build_training_dataset_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spatial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--metadata-vae-geometry", type=int, nargs=2, metavar=("CHANNELS", "SPATIAL_STRIDE"))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    values = build_training_dataset_config(config)
    values.update(reference_entities=None, reference_crops=None, text_admission_path=None, workload_cap=None)
    values["memory_prefix_options"] = {**values["memory_prefix_options"], "row_policy": MEMORY_ROW_POLICY_FIXED_ROLE_V1}
    # 只有训练表：验证集不用裁剪（``build_validation_dataset``）。
    with Path(values["sampling_manifest_path"]).open() as handle:
        values["sample_budget"] = sum(1 for _ in handle)
    if args.metadata_vae_geometry is not None:
        channels, stride = args.metadata_vae_geometry
        values.update(target_storage="video", target_index_path=None, target_roots=None,
                      target_latent_geometry={"channels": channels, "spatial_stride": stride}, target_mask_weights=None)
    dataset = BlockDataset(**values)
    indices = (json.loads(args.sample_indices.read_text()) if args.sample_indices is not None
               else list(range(len(dataset))))
    print(json.dumps(build_reference_crops(dataset, spatial_root=args.spatial_root, output_path=args.output,
                                           sample_indices=indices, workers=args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
