#!/usr/bin/env python3
"""从训练成员冻一份渲染器目标窗口表，或汇总冻结表的实际曝光。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from mspc.config import load_experiment_config
from mspc.data.scene_training_manifest import media_from_task
from mspc.data.renderer_training_members import SAMPLING_SCHEMES, summarize_exposure, write_drawn_sampling


def main():
    if sys.argv[1:2] == ["exposure"]:
        parser = argparse.ArgumentParser(description="汇总冻结采样实际曝光")
        parser.add_argument("--members", type=Path, required=True)
        parser.add_argument("--sampling", type=Path, required=True)
        parser.add_argument("--config", type=Path, required=True, help="媒体合同从训练配置的task读取")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--limit", type=int)
        args = parser.parse_args(sys.argv[2:])
        media = media_from_task(load_experiment_config(args.config).resolved["task"])
        result = summarize_exposure(members=args.members, sampling=args.sampling, media=media, limit=args.limit)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    elif sys.argv[1:2] == ["draw"]:
        parser = argparse.ArgumentParser(description="按镜数分层冻结完整源窗，或重放旧时长档抽样")
        parser.add_argument("--config", type=Path, required=True, help="媒体合同从训练配置的task读取")
        parser.add_argument("--members", type=Path, required=True)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--sampling-scheme", choices=SAMPLING_SCHEMES, required=True)
        parser.add_argument("--partition-max-rgb-frames", type=int,
                            help="镜数分层在源16fps网格上的帧预算；超预算完整单镜独占窗，不裁镜")
        parser.add_argument("--max-source-shot-seconds", type=float,
                            help="镜数分层必填：任何目标镜源时长超过此值，不进入候选窗并中断连续段")
        parser.add_argument("--shot-count-weights", type=float, nargs=4,
                            help="镜数分层必填：1、2、3、4及以上镜四层的权重；层内全部合法完整窗均匀抽")
        parser.add_argument("--frame-buckets", type=int, nargs="+",
                            help="各时长档总帧数上界，升序，例如480档使用80 160 228；不自动改变窗口预算")
        parser.add_argument("--bucket-weights", type=float, nargs="+", help="旧时长档重放：每档一个权重")
        parser.add_argument("--max-shots", type=int, help="只给逐镜对照取 1；联合窗口不给（镜数不封顶）")
        parser.add_argument("--long-single-shot", nargs=2, metavar=("SHARE", "MIN_SOURCE_FRAMES"),
                            help="镜数分层可选：抽到单镜层时按 SHARE 只取长单镜窗（源16fps帧数不少于 MIN_SOURCE_FRAMES，"
                                 "且块标注该镜 activity 为 action_or_state_change 或 changes 非空；成员须带 annotation）；"
                                 "四层权重不变")
        parser.add_argument("--sample-budget", type=int, required=True)
        parser.add_argument("--forced-samples", type=int, help="旧时长档重放：显式衣着冲突定向份额")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--limit", type=int, help="只写前若干行（同一全量表的前缀），用于首次真实切片")
        args = parser.parse_args(sys.argv[2:])
        media = media_from_task(load_experiment_config(args.config).resolved["task"])
        result = write_drawn_sampling(members_path=args.members, media=media, seed=args.seed,
                                      sampling_scheme=args.sampling_scheme,
                                      partition_max_rgb_frames=args.partition_max_rgb_frames,
                                      max_source_shot_seconds=args.max_source_shot_seconds, shot_count_weights=args.shot_count_weights,
                                      frame_buckets=args.frame_buckets,
                                      bucket_weights=args.bucket_weights, max_shots=args.max_shots,
                                      sample_budget=args.sample_budget, forced_samples=args.forced_samples,
                                      output=args.output, limit=args.limit,
                                      long_single_shot=None if args.long_single_shot is None else
                                      (float(args.long_single_shot[0]), int(args.long_single_shot[1])))
    elif sys.argv[1:2] == ["filter-text"]:
        parser = argparse.ArgumentParser(description="按真实固定抽签文字生产512-token准入，不重排原data_id")
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--split", choices=("train", "validation"), required=True)
        parser.add_argument("--world-size", type=int, required=True)
        parser.add_argument("--workers", type=int, required=True)
        parser.add_argument("--latent-channels", type=int, required=True)
        parser.add_argument("--vae-spatial-stride", type=int, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--rows-output", type=Path, required=True)
        parser.add_argument("--limit", type=int, help="首次真实切片；不完整产物不能被训练消费者读取")
        args = parser.parse_args(sys.argv[2:])
        from mspc.training.text_admission import write_text_admission
        result = write_text_admission(load_experiment_config(args.config), split=args.split,
            world_size=args.world_size, workers=args.workers,
            latent_channels=args.latent_channels, vae_spatial_stride=args.vae_spatial_stride,
            output=args.output, rows_output=args.rows_output, limit=args.limit)
        result = {key: result[key] for key in ("complete", "split", "sample_budget", "checked_samples",
            "token_limit", "eligible_count", "excluded_count", "compiled_prompt_count", "max_prompt_tokens",
            "workers", "initialization_seconds", "production_seconds")}
    else:
        raise SystemExit("usage: prepare_renderer_training.py {draw|exposure|filter-text} ...")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
