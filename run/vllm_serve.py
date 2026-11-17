"""显式服务配置入口；GPU选择只由GPU scheduler租约环境给定。"""
import argparse
import os
from pathlib import Path

import yaml

from mspc.inference.planner_serving import save_service_provenance, service_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("GPU scheduler CUDA device selectors required")
    os.environ["LD_LIBRARY_PATH"] = config["ld_library_path"]
    os.environ["OMP_NUM_THREADS"] = str(config["omp_num_threads"])
    save_service_provenance(args.config, config, args.provenance_dir)
    command = service_command(config)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
