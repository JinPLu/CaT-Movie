#!/usr/bin/env python3
"""从真实块、校验标注和逐镜取样台账导出按片分开的 CPU 消费者成员。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "wan22_runtime"))
from mspc.data.block_members import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("blocks", "annotations", "ledger-shots", "output-dir", "tokenizer-path"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--min-laion-aesthetic", type=float, required=True)
    parser.add_argument("--min-mean-luma", type=float, required=True)
    parser.add_argument("--min-dover-technical", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
