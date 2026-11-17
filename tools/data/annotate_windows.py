"""窗或块标注的官方批量入口：一单位一请求，视频用 OSS 签名地址。

合同在 ``configs/data/window_annotation_v15.json``（提示词、模型、是否开思考）。一个长驻进程做完三件事：把窗清单切成
若干请求文件并提交批量任务；轮询并把进度写进 ``status/annotate-sKofN.json``（终端进度工具会显示）；任务完成后把结果
与错误文件取回 ``annotations/``。可续跑：已提交的任务记在 ``final/annotation-batches.json``，重跑只接着轮询和取回。
验收（``validate_annotations.py``）之后，``--round N`` 把不合法与没返回的窗重发一轮，台账与结果文件带 ``rN`` 标记。
第17版块使用独立 block- 路径；整块重发只处理登记表层不合法与没答复。

密钥不进参数、环境变量、日志或磁盘：从标准输入读三行（OSS AccessKey ID、OSS AccessKey Secret、DashScope API Key）。

    <三行密钥> | python tools/data/annotate_windows.py --unit window --config CFG --contract configs/data/window_annotation_v15.json \
        --bucket NAME --endpoint oss-cn-beijing.aliyuncs.com --prefix windows-v2 --lines-per-file 5000 --limit 0 --round 0
"""
import argparse
from contextlib import contextmanager
import fcntl
import base64
import hashlib
import hmac
import json
import math
import re
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SIGNED_DAYS = 7  # 签名地址的有效期，要盖住批量任务的完成时限
COMPLETION_WINDOW = "48h"
POLL_SECONDS = 120
BLOCK_SCHEMA_VERSIONS = (17, 18)  # 第 18 版只改了提示词的写法，输出结构与第 17 版相同


class Api:
    def __init__(self, base, key, secrets):
        self.base, self.key, self.secrets = base, key, secrets

    def clean(self, text):
        text = re.sub(r'https?://[^\s"<>]*(?:OSSAccessKeyId|Signature)=[^\s"<>]*', '<signed-url-redacted>', text)
        for secret in self.secrets:
            text = text.replace(secret, "<redacted>")
        return text

    def call(self, method, path, data=None, headers=None, raw=False):
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Authorization": "Bearer " + self.key, **(headers or {})})
        for attempt in range(6):
            try:
                body = urllib.request.urlopen(request, timeout=1800).read()
                return body if raw else json.loads(body)
            except urllib.error.HTTPError as error:
                detail = self.clean(error.read().decode(errors="replace"))[:600]
                if error.code < 500 and error.code != 429:
                    raise SystemExit(f"{method} {path}: HTTP {error.code} {detail}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                detail = type(error).__name__
            if method == "POST" and path == "/batches":
                raise SystemExit(f"batch submission outcome unknown ({detail}); do not resubmit")
            time.sleep(min(300, 5 * 2 ** attempt))  # 官方建议指数退避
        raise SystemExit(f"{method} {path}: gave up after retries ({detail})")

    def upload(self, name, payload):
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: application/jsonl\r\n\r\n").encode()
        body += payload + f"\r\n--{boundary}--\r\n".encode()
        return self.call("POST", "/files", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def signed_url(bucket, endpoint, key, access_key, secret):
    expires = int(time.time()) + SIGNED_DAYS * 86400
    signature = base64.b64encode(hmac.new(secret.encode(), f"GET\n\n\n{expires}\n/{bucket}/{key}".encode(), hashlib.sha1).digest()).decode()
    return (f"https://{bucket}.{endpoint}/{key}?OSSAccessKeyId={urllib.parse.quote(access_key)}&Expires={expires}"
            f"&Signature={urllib.parse.quote(signature, safe='')}")


def request_line(row, contract, url):
    """一窗一条请求：一条 user 消息，视频在前、说明在后，各镜时间附在末尾；不设 system、不传采样参数、不设输出上限。"""
    call = contract["call"]
    times = "\n".join(f"Shot {item['shot'] + 1}: {item['start']:.1f}–{item['end']:.1f} s" for item in row["proxy"]["shot_timestamps"])
    body = {"model": call["model"], "enable_thinking": call["enable_thinking"], "response_format": call["response_format"],
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": url}, "fps": call["video_fps"]},
                {"type": "text", "text": contract["prompt"] + "\n" + times}]}]}
    return json.dumps({"custom_id": row["key"], "method": "POST", "url": "/v1/chat/completions", "body": body}, ensure_ascii=False)


def annotation_paths(root, unit, round_number=0):
    if unit not in ("window", "block"):
        raise ValueError(f"unknown annotation unit: {unit}")
    prefix = "block-" if unit == "block" else ""
    suffix = f"-r{round_number}" if round_number else ""
    return {"manifest": root/"final"/f"{unit}s.jsonl",
            "ledger": root/"final"/f"{prefix}annotation-batches{suffix}.json",
            "results": root/f"{prefix}annotations",
            "annotations": root/"final"/f"{prefix}annotations.jsonl",
            "summary": root/"final"/f"{prefix}annotation-summary.json",
            "status": root/"status"/f"annotate{'-blocks' if unit == 'block' else ''}{suffix}"}


def selected_rows(a, root):
    paths = annotation_paths(root, a.unit, a.round)
    wanted = set(a.keys_file.read_text().split()) if a.keys_file is not None else None
    done = set()
    if a.round:
        for record in map(json.loads, paths["annotations"].open()):
            if a.unit == "window":
                keep = record["status"] == "ok" or record.get("error_code") == "data_inspection_failed"
            else:
                # 局部错交给修镜；整块重发只处理登记表层无效或没有答复。
                keep = record["status"] == "ok" or (record["status"] == "invalid" and not record["block_errors"])
            if keep:
                done.add(record["key"])
    selected, seen = [], set()
    for row in map(json.loads, paths["manifest"].open()):
        if wanted is not None and row["key"] not in wanted:
            continue
        seen.add(row["key"])
        if row["key"] not in done:
            selected.append(row)
    if wanted is not None and wanted - seen:
        raise ValueError(f"keys missing from manifest: {sorted(wanted-seen)[:8]}")
    return selected[:a.limit] if a.limit else selected


def price(usage, batch=False):
    if not usage:
        return None
    inp, out = usage["prompt_tokens"], usage["completion_tokens"]
    if inp > 256000:
        raise ValueError("input beyond reserved price tier")
    return (inp*.2+out*.8)*(1 if inp <= 32000 else 3)*(0.5 if batch else 1)/1e6


def batch_reservation(row, contract, output_tokens):
    """一条请求的预留：输入按提示词与视频时长的上界，输出按 ``output_tokens``（回答加思考）。

    ``output_tokens`` 由命令行给出。取模型上限（131072+262144）时任何一条都不会超出预留，但同时在途的条数很少；
    取实测最大值的几倍时，守住硬顶靠的是一个文件里几千条的合计不超过合计预留，超了 ``settle`` 会报错停下。
    """
    seconds = row['proxy']['shot_timestamps'][-1]['end']
    upper = len(contract['prompt'].encode())+2000+math.ceil(seconds+4)*390
    if upper > 256000:
        raise ValueError('input beyond reserved price tier')
    amount = (upper*.2+output_tokens*.8)*(1 if upper <= 32000 else 3)*.5/1e6
    return math.ceil(amount*1e6)/1e6


class SpendLedger:
    """实时与批量共享的费用账；锁保护发前预留，未知用量不释放。"""
    def __init__(self, path, cap):
        self.path, self.cap = Path(path), cap
        if cap <= 0:
            raise ValueError("budget must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(self.path.read_text()) if self.path.exists() else {}
            yield ledger
            save(self.path, ledger)

    def snapshot(self):
        with self.locked() as ledger:
            return dict(ledger)

    def reserve(self, key, amount, **metadata):
        with self.locked() as ledger:
            if key in ledger:
                return False
            if sum(r["charged_or_reserved_cny"] for r in ledger.values()) + amount > self.cap + 1e-9:
                return False
            ledger[key] = {"state": "started", "charged_or_reserved_cny": amount, "started_at": time.time(), **metadata}
            return True

    def settle(self, key, actual, **metadata):
        with self.locked() as ledger:
            row = ledger[key]
            if actual is not None and actual > row["charged_or_reserved_cny"] + 1e-9:
                raise ValueError("actual cost exceeded reservation")
            row.update(state="finished", actual_cny=metadata.get("measured_cny", actual), **metadata)
            if actual is not None:
                row["charged_or_reserved_cny"] = actual


def prepare_batches(a, root, contract):
    selected = selected_rows(a, root)
    parts = [selected[i:i+a.lines_per_file] for i in range(0, len(selected), a.lines_per_file)]
    return {"contract": contract["contract"], "model": contract["call"]["model"], "prefix": a.prefix,
            "unit": a.unit, "selected_keys": [r["key"] for r in selected],
            "batches": [{"index": i, "keys": [r["key"] for r in part], "requests": len(part), "fetched": False} for i, part in enumerate(parts)]}, {r["key"]: r for r in selected}


def submit_part(a, contract, api, access_key, secret, item, catalog):
    lines = [request_line(catalog[key], contract, signed_url(a.bucket, a.endpoint, f"{a.prefix}/{key}.mp4", access_key, secret)) for key in item["keys"]]
    if any(len(line.encode()) > 6_000_000 for line in lines):
        raise ValueError("request line exceeds batch API limit")
    payload = ("\n".join(lines)+"\n").encode()
    if len(payload) > 500_000_000:
        raise ValueError("request file exceeds batch API limit")
    uploaded = api.upload(f"{a.unit}s-{item['index']:02d}.jsonl", payload)
    batch = api.call("POST", "/batches", json.dumps({"input_file_id": uploaded["id"], "endpoint": "/v1/chat/completions",
                                                     "completion_window": COMPLETION_WINDOW}).encode(), {"Content-Type": "application/json"})
    item.update(batch_id=batch["id"], input_file_id=uploaded["id"], megabytes=len(payload)/1e6, submitted_at=time.time())
    print(json.dumps({"submitted": item["index"], "requests": item["requests"], "batch": batch["id"], "status": batch.get("status")}), flush=True)


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    temporary.replace(path)


def run(a, root, contract, access_key, secret, api_key):
    api = Api(contract["call"]["base_url"], api_key, (access_key, secret, api_key))
    paths = annotation_paths(root, a.unit, a.round)
    paths["results"].mkdir(exist_ok=True)
    (root/"status").mkdir(exist_ok=True)
    ledger = paths["ledger"]
    budget = SpendLedger(a.spend_ledger, a.budget_cny) if a.spend_ledger else None
    if ledger.exists():
        state = json.loads(ledger.read_text())
        if state["contract"] != contract["contract"] or state["prefix"] != a.prefix:
            raise ValueError("existing batch ledger belongs to another contract or prefix")
        catalog = {r["key"]: r for r in map(json.loads, paths["manifest"].open())}
    else:
        state, catalog = prepare_batches(a, root, contract)
        save(ledger, state)
    tag = f"r{a.round}-" if a.round else ""
    count = len(state["batches"])
    started = time.monotonic()
    while True:
        waiting, submitted = 0, 0
        deadline = a.time_limit_seconds is not None and time.monotonic()-started >= a.time_limit_seconds
        for item in state["batches"]:
            spend_key = f"batch-{a.unit}-r{a.round}-{item['index']:03d}"
            if item["fetched"]:
                continue
            if "batch_id" not in item:
                if deadline:
                    continue
                if item.get("submitting"):
                    raise RuntimeError("submission outcome unknown; inspect remote batch list before resuming")
                item["reservations"] = {key:batch_reservation(catalog[key],contract,a.reserve_output_tokens) for key in item["keys"]}
                if budget and not budget.reserve(spend_key, sum(item["reservations"].values()), requests=item["requests"], mode="batch"):
                    continue
                item["submitting"] = True
                save(ledger, state)
                submit_part(a, contract, api, access_key, secret, item, catalog)
                item.pop("submitting")
                save(ledger, state)
                submitted += 1
            batch = api.call("GET", "/batches/"+item["batch_id"])
            counts, status = batch.get("request_counts") or {}, batch.get("status")
            item["last_status"] = status
            item["last_observed_at"] = time.time()
            ended = status in ("completed", "failed", "expired", "cancelled")
            if ended:
                actual, accounted, unbilled = 0.0, set(), set()
                for kind in ("output", "error"):
                    file_id = batch.get(f"{kind}_file_id")
                    if not file_id:
                        continue
                    data = api.call("GET", f"/files/{file_id}/content", raw=True)
                    # 响应错误偶尔引用请求URL；任何签名地址只留在内存。
                    clean = api.clean(data.decode())
                    (paths["results"]/f"{tag}batch-{item['index']:02d}.{kind}.jsonl").write_text(clean)
                    for row in map(json.loads, clean.splitlines()):
                        spent = ((row.get("response") or {}).get("body") or {}).get("usage")
                        if spent:
                            actual += price(spent, batch=True)
                            accounted.add(row["custom_id"])
                        elif kind == "error" or row.get("error"):
                            # 官方批量接口只对成功执行的请求计费，行级错误不产生费用；两个结果文件里都没有的请求仍留着预留。
                            unbilled.add(row["custom_id"])
                if budget:
                    budget.settle(spend_key, actual+sum(v for k,v in item["reservations"].items() if k not in accounted|unbilled), measured_cny=actual, usage_accounted=len(accounted), failed_unbilled=len(unbilled-accounted))
                item.update(fetched=True, final_status=status, finished_at=time.time(), measured_cny=actual)
            else:
                waiting += 1
            save(ledger, state)
            save(paths["status"].with_name(paths["status"].name+f"-s{item['index']}of{count}.json"),
                 {"shard": [item["index"], count], "state": "completed" if ended else "running", "batch_status": status,
                  "batch_id": item["batch_id"], "requests_done": counts.get("completed", 0)+counts.get("failed", 0),
                  "requests_total": item["requests"], "failed": counts.get("failed", 0),
                  "seconds": (item.get("finished_at") or time.time())-item["submitted_at"]})
        unfinished = sum(not i["fetched"] for i in state["batches"])
        other_inflight = budget and any(r['state'] == 'started' for r in budget.snapshot().values())
        if not unfinished or deadline or (not waiting and not submitted and not other_inflight):
            break
        time.sleep(POLL_SECONDS)
    print(json.dumps({"batches": state["batches"], "pending": unfinished}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unit", choices=("window", "block"), required=True)
    p.add_argument("--keys-file", type=Path, help="只标这些键，一行一个")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--lines-per-file", type=int, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--spend-ledger", type=Path, help="与实时试标共享的费用台账")
    p.add_argument("--budget-cny", type=float, help="含在途预留的硬顶")
    p.add_argument("--reserve-output-tokens", type=int, help="每条请求按多少输出 token 预留费用；与 --spend-ledger 一起给")
    p.add_argument("--time-limit-seconds", type=float, help="到时停止提交与轮询，远端任务不取消")
    a = p.parse_args(argv)
    if a.round < 0 or a.limit < 0 or not 1 <= a.lines_per_file <= 50000:
        p.error("invalid round, limit or lines-per-file")
    if len({bool(a.spend_ledger), a.budget_cny is not None, a.reserve_output_tokens is not None}) != 1:
        p.error("spend-ledger, budget-cny and reserve-output-tokens are required together")
    access_key, secret, api_key = (sys.stdin.readline().strip() for _ in range(3))
    if not (access_key and secret and api_key):
        raise SystemExit("three credential lines expected on stdin")
    root = Path(json.loads(a.config.read_text())["root"])
    contract = json.loads(a.contract.read_text())
    if contract["schema_version"] not in ((15,) if a.unit == "window" else BLOCK_SCHEMA_VERSIONS):
        raise ValueError("annotation unit and contract disagree")
    run(a, root, contract, access_key, secret, api_key)


if __name__ == "__main__":
    main()
