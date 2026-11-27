"""显式服务配置入口；GPU选择只由GPU scheduler租约环境给定。"""
import argparse
import os
from pathlib import Path

import yaml

from mspc.config.shared_root import expand_shared_root_token
from mspc.inference.planner_serving import save_service_provenance, service_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    # 服务配置里的环境与模型路径写 ${MSPC_SHARED_ROOT}，主服务器、训练集群各自展开（训练集群没有 /media/datasets）。
    config = {key: expand_shared_root_token(value, f"{args.config}:{key}") if isinstance(value, str) else
              [expand_shared_root_token(v, f"{args.config}:{key}") if isinstance(v, str) else v for v in value]
              if isinstance(value, list) else value for key, value in raw.items()}
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("GPU scheduler CUDA device selectors required")
    os.environ["LD_LIBRARY_PATH"] = config["ld_library_path"]
    os.environ["OMP_NUM_THREADS"] = str(config["omp_num_threads"])
    save_service_provenance(args.config, config, args.provenance_dir)
    command = service_command(config)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
