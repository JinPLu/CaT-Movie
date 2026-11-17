"""布局基线的一对渲染器。文字组 strength=0，结构组 strength=ln2；其余空间接口相同。

层范围和扩散时间范围是接口起点，不是已测过的最优注入位置。覆盖率下限是开训门槛：
抽样目标镜里带布局的比例低于该值就拒绝启动，避免空偏置被当成布局监督。
"""
from pathlib import Path
import math


_layers = list(range(30))
_timesteps = [0, 1000]

_name = "text_whole_frame_layout"
_temporal_position_mode = "global_time_msm_fixed_shot_phase_v1"
_self_attention_policy = "full"
_memory_prefix = {
    "frames_max": 10, "dropout_p": 0.2,
    "noise": {"train_sigma_max": 0.0, "inference_sigma": 0.0},
    "reference_offset": {"train": {"dt": [-5, -5], "dh": [0, 0], "dw": [0, 0]},
                         "inference": [-5, 0, 0]},
    "attention_policy": "bidirectional_v1", "row_policy": "fixed_role_v1", "placement": "ordered_slots",
    "attention_window": {"layer_indices": list(range(30)), "timestep_range": [0, 1000]},
}
_reference_row_dropout_p = 0.0

_shared = "${MSPC_SHARED_ROOT}"
_root = _shared
# 代码路径取本配置文件所在的检出（启动时即冻结代码根），不经数据根：冻结根里只放代码、不建软链。
_checkout = str(Path(__file__).resolve().parents[2])
_weights = f"{_root}/OoVMetric/models_ckpt/wan22_5b_native"
_run_root = f"{_root}/Storyboard/outputs/experiments/renderer-text-whole-frame-20261027"
# 数据资产与本次输出分开，改输出目录不会改到训练成员。
# 成员带布局；采样表用修订监督那份整帧随机参考，不用实体裁图采样。
_members = f"{_root}/Storyboard/outputs/experiments/renderer-layout-baseline-20261027/inputs"
_sampling = f"{_root}/Storyboard/outputs/experiments/renderer-revised-supervision-20261022/inputs"
_sam = f"{_root}/Storyboard/outputs/experiments/renderer-layout-baseline-20261027/sam-weights"
_pythonpath = [f"{_checkout}/src", _checkout, f"{_checkout}/third_party/wan22_runtime"]
EXPERIMENT_CONFIG = {
    "schema_version": 1,
    "kind": "train",
    "trainer": "sft",
    "identity": {
        "model_key": f"wan22_5b_{_name}",
        "dataset_key": "block_annotation_v18_quality",
        "run_name": _name,
        "experiment_key": "layout_baseline",
    },
    "task": {"paradigm": "text_to_multishot_video", "height": 480, "width": 832,
             "frame_policy": "fps_aware_target_fps", "max_rgb_frames": 308,
             "target_fps": 16, "vae_temporal_stride": 4},
    "runtime": {
        "python": f"{_shared}/env/wan22_fun/bin/python",
        "distributed_timeout_seconds": 3600,
        "deepspeed": {"zero_optimization": {"stage": 2, "contiguous_gradients": False,
                                                          "reduce_bucket_size": 64_000_000,
                                                          "allgather_bucket_size": 64_000_000,
                                                          "overlap_comm": False}, "gradient_clipping": 1.0,
                      "wall_clock_breakdown": True,
                      "fp16": {"enabled": False}, "bf16": {"enabled": False},
                      "torch_autocast": {"enabled": True, "dtype": "bfloat16", "lower_precision_safe_modules": []}},
        "hf_cache": f"{_shared}/OoVMetric/models_ckpt",
        "offline_hf": True,
    },
    "environment": {
        "tokenizers_parallelism": "false", "python_unbuffered": "1", "wandb_mode": "disabled",
    },
    "nccl": {
        "debug": "WARN", "async_error_handling": "1", "device_max_connections": "32",
        "blocking_wait": "0", "buffsize": "8388608", "nthreads": "512",
    },
    "launch": {
        "train_entrypoint": f"{_checkout}/src/mspc/training/runner.py",
        "infer_entrypoint": f"{_checkout}/src/mspc/inference/wan22.py",
        "train_pythonpath": list(_pythonpath), "infer_pythonpath": list(_pythonpath),
        "cuda_visible_devices": None,
    },
    # 原生 Wan2.2-TI2V-5B：30层、dim3072、24头、FFN14336，
    # in/out48通道、patch(1,2,2)、text_dim4096；实测DiT共4,999,787,712参数。
    # 结构随所选权重，
    # 定义在 third_party/wan22_runtime/diffsynth/configs/model_configs.py。
    # 可调机制直接在本文件声明；历史配方由各运行的固定提交保存。
    "model": {
        "type": "wan22_diffsynth",
        "diffsynth_root": f"{_checkout}/third_party/wan22_runtime",
        "temporal_position_mode": _temporal_position_mode,
        "self_attention_policy": _self_attention_policy,
        "paths": {
            "dit": [f"{_weights}/diffusion_pytorch_model-{index:05d}-of-00003.safetensors"
                    for index in (1, 2, 3)],
            "t5": f"{_weights}/models_t5_umt5-xxl-enc-bf16.pth",
            "vae": f"{_weights}/Wan2.2_VAE.pth",
            "tokenizer": f"{_weights}/google/umt5-xxl",
        },
        "memory_prefix": _memory_prefix,
        "spatial_layout": {
            "text_attention_backend": "sdpa",
            "strength": 0,
            "layer_indices": _layers,
            "timestep_range": _timesteps,
            "mode": "text_cross_attention_bias_v1",
        },
    },
    "dataset": {
        "cache_provenance_path": f"{_root}/Storyboard/outputs/experiments/renderer-quality-20261014/inputs/cache-provenance.json",
        "segment_capacity": {"max_frames": 228},
        "reference_row_dropout_p": _reference_row_dropout_p,
        "target_mask_weights": {"index_path": f"{_sam}/index.jsonl", "root": _sam, "weighting": "area_coverage"},
        "reference_policy": "random_frame",
        "box_dropout_p": 0.2,
        "minimum_layout_shot_fraction": 0.5,
        "type": "block_latent_cache",
        "members_path": f"{_members}/train.jsonl",
        "text_admission_path": None,
        "sampling_manifest_path": f"{_sampling}/train-sampling.jsonl",
        "target_index_path": f"{_root}/Storyboard/outputs/experiments/renderer-multishot-quality-20261014/inputs/target-index.jsonl",
        "target_roots": {"first_batch_window": f"{_root}/Storyboard/outputs/experiments/renderer-multishot-quality-20261014/inputs/targets/first_batch_window",
                         "shot": f"{_root}/Storyboard/outputs/experiments/renderer-multishot-quality-20261014/inputs/targets/shot"},

    },
    "training": {
        "shot_forward_mode": "joint", "seed": 42, "task": "sft", "mixed_precision": "no",
        "scheduler_train_timesteps": 1000, "flow_shift": 5.0, "num_epochs": 1, "save_steps": 500,
        "max_train_steps": 3000, "step_unit": "optimizer_update",
        "validation": {"members_path": f"{_members}/val.jsonl", "sample_budget": 256,
                       "text_admission_path": None,
                       "sampling_manifest_path": f"{_sampling}/val-sampling.jsonl",
                       "every_steps": 250, "draws": 2},
        "gradient_accumulation_steps": 2, "use_gradient_checkpointing": True,
        "activation_checkpointing": "recompute_all",
        "find_unused_parameters": False, "dit_parameter_freeze": {"recipe": "full_dit_v1"},
        "loss_tracking": {"project": "Storyboard", "workspace": "JinPLu", "experiment_name": _name,
                          "mode": "offline", "api_key_file": None},
    },
    "pipeline_inputs": {"tiled": False, "tile_size": [30, 52], "tile_stride": [15, 26]},
    "optimization": {"learning_rate": 1e-5, "weight_decay": 0.01,
                     "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 300},
    "dataloader": {"per_device_batch_size": 1, "split_batches": False, "drop_last": True,
                   "dataset_num_workers": 4 if _memory_prefix is not None else 0, "prefetch_factor": 1,
                   "video_decode_num_threads": 1,
                   "persistent_workers": False, "pin_memory": False},
    # 每更新保留CPU及DeepSpeed原生计时；前10步与每10步同步细分GPU阶段。
    "timing": {"log_steps": 10, "log_every": 10, "sync_cuda": True},
    "output": {"root": f"{_run_root}/training"},
    # 显式分镜面板的检查点诊断从这里读取生成参数；CSV自动smoke不用于块训练。
    "inference": {
        "enabled": False, "device": "cuda", "target_height": 480, "target_width": 832,
        "fps": 16, "seed": 42,
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "num_inference_steps": 50, "sigma_shift": 5.0, "cfg_scale": 5.0, "video_quality": 8,
        "sampler": "flow_unipc_official",
        "test_csv_path": None, "smoke_case_id": None, "output_dir": None,
        "weight_source": "training_output_checkpoint", "dit_checkpoint": None,
    },
    "checkpoint": f"{_root}/Storyboard/outputs/experiments/renderer-revised-supervision-20261022/training/revised_supervision_reference__20261023T104856_952788Z/checkpoints/step-3000.safetensors",
}
