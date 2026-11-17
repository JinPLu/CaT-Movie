#!/usr/bin/env python3
"""取样台账：按块一行、按镜一行，写 ``final/sampling-ledger/{blocks,shots}.jsonl`` 与 ``summary.json``。

只读各关已落盘的产物（块清单、块标注、评测排除、装块被拒、目标潜变量索引、参考帧库清单、空间层），
路径全在配置 ``configs/data/sampling_ledger_v1.json`` 里，相对数据根；数据根默认取 ``pipeline_config`` 的 ``root``，
``--root`` 可换（用来在切片上跑）。目标潜变量索引、参考帧库清单、空间层根写 ``null`` 表示这一关还没有产物，对应的栏就是 ``null``。

    python tools/data/build_sampling_ledger.py --config configs/data/sampling_ledger_v1.json [--root DIR]
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mspc.data import sampling_ledger  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="换数据根；不给就用 pipeline_config 的 root")
    args = parser.parse_args(argv)
    config = sampling_ledger.check_config(json.loads(args.config.read_text()))
    root = args.root or Path(json.loads((REPO / config["pipeline_config"]).read_text())["root"])
    contract = json.loads((REPO / config["annotation_contract"]).read_text())
    summary = sampling_ledger.build(sampling_ledger.resolve_inputs(config["inputs"], root), root / config["output_dir"],
                                    contract=contract)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
