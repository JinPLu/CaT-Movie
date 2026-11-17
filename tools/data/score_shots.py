"""前置处理第二遍「取」的入口：给合格镜打分（DOVER 两头、LAION 美学分）并按镜导代理片。

一个进程一张卡（CUDA_VISIBLE_DEVICES 只给一张）；同一张卡可以起两个进程跑不同分片。读第一遍 collect 出的合格镜清单，可续跑。
"""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard", type=int, nargs=2, metavar=("INDEX", "COUNT"), required=True)
    p.add_argument("--limit", type=int, required=True, help="只跑这么多条源片（按镜数排序后等距取），0 为全部")
    a = p.parse_args()
    if not os.environ["CUDA_VISIBLE_DEVICES"].isdigit():
        raise ValueError("scoring needs exactly one visible GPU")
    from mspc.data.shot_scoring import run_shard

    print(json.dumps(run_shard(json.loads(a.config.read_text()), tuple(a.shard), a.limit), ensure_ascii=False))
