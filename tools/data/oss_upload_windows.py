"""把窗或块的代理片传到 OSS（私有存储空间），供官方批量接口用带签名的地址读取。

密钥不进参数、环境变量、日志或磁盘：从标准输入读两行（AccessKey ID、AccessKey Secret）。可续传：已传过的对象记在
``final/oss-uploaded.txt``（块是 ``final/oss-uploaded-blocks.txt``），重跑会跳过。对象名是 ``<prefix>/<键>.mp4``。只用标准库。

    <两行密钥> | python tools/data/oss_upload_windows.py --unit window --config CFG --bucket NAME --endpoint oss-cn-beijing.aliyuncs.com \
        --prefix windows-v1 --workers 64
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
import hashlib
import hmac
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request


def put(task):
    key, path, bucket, endpoint, access_key, secret = task
    data = Path(path).read_bytes()
    for attempt in range(4):
        date = formatdate(usegmt=True)
        text = f"PUT\n\nvideo/mp4\n{date}\n/{bucket}/{key}"
        signature = base64.b64encode(hmac.new(secret.encode(), text.encode(), hashlib.sha1).digest()).decode()
        request = urllib.request.Request(f"https://{bucket}.{endpoint}/{key}", data=data, method="PUT",
                                         headers={"Date": date, "Authorization": f"OSS {access_key}:{signature}", "Content-Type": "video/mp4"})
        try:
            urllib.request.urlopen(request, timeout=300)
            return key, len(data), None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            failure = f"{type(error).__name__}: {getattr(error, 'code', '')}"
            time.sleep(2 ** attempt)
    return key, 0, failure


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unit", choices=("window", "block"), required=True)
    p.add_argument("--keys-file", type=Path, help="只传这些键，一行一个")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--workers", type=int, required=True)
    a = p.parse_args()
    access_key, secret = sys.stdin.readline().strip(), sys.stdin.readline().strip()
    if not access_key or not secret:
        raise SystemExit("two lines expected on stdin: AccessKey ID, AccessKey Secret")
    root = Path(json.loads(a.config.read_text())["root"])
    ledger = root / "final" / ("oss-uploaded.txt" if a.unit == "window" else "oss-uploaded-blocks.txt")
    wanted = set(a.keys_file.read_text().split()) if a.keys_file is not None else None
    name = "oss-upload" if a.unit == "window" else "oss-upload-blocks"
    done = set(ledger.read_text().split()) if ledger.exists() else set()
    tasks = []
    with (root / "final" / f"{a.unit}s.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if wanted is not None and row["key"] not in wanted:
                continue
            key = f"{a.prefix}/{row['key']}.mp4"
            if key not in done:
                tasks.append((key, row["proxy"]["video"], a.bucket, a.endpoint, access_key, secret))
    started, lock, counts = time.monotonic(), threading.Lock(), {"ok": 0, "failed": 0, "bytes": 0}
    total = len(tasks) + len(done)

    def status(state):
        seconds = time.monotonic() - started
        value = {"shard": [0, 1], "state": state, "objects_done": len(done) + counts["ok"], "objects_total": total, "failed": counts["failed"],
                 "megabytes": counts["bytes"] / 1e6, "seconds": seconds, "megabytes_per_second": counts["bytes"] / 1e6 / seconds if seconds else 0.0}
        temporary = root / "status" / (name + ".partial")
        temporary.write_text(json.dumps(value))
        temporary.replace(root / "status" / (name + "-s0of1.json"))
        return value

    written = started
    with ThreadPoolExecutor(max_workers=a.workers) as pool, ledger.open("a") as handle:
        for key, size, failure in pool.map(put, tasks):
            if failure:
                counts["failed"] += 1
            else:
                counts["ok"] += 1
                counts["bytes"] += size
                handle.write(key + "\n")
            if time.monotonic() - written >= 10:
                written = time.monotonic()
                handle.flush()
                status("running")
    print(json.dumps(status("completed"), ensure_ascii=False))
