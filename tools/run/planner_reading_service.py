#!/usr/bin/env python3
"""规划器读取的本机服务：在规划器环境里加载 Qwen3.5-9B 底座，按 ``PlannerTokenReader`` 认出参考图里可见的登记人物，或出
点控制加特征注入的特征；调用方经 ``PlannerReadingClient`` 调它。故事出片不再调它（11-14 起参考身份行由分镜写），留给分析工具。

    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=<卡> PYTHONPATH=src:. <规划器环境 python> tools/run/planner_reading_service.py \\
        --model <Qwen3.5-9B> --feature-pooling name --feature-layer 19 --max-new-tokens 768 --cut-margin 4 --port 8711 \\
        --devices 0,0,0

``--devices`` 每项一个副本：各是一个独立进程、各载一份模型（项是可见卡内的序号，同一张卡可重复）。所有副本从同一条请求队列
取活，谁空谁取，所以一个地址后面同时有几条请求在算；每条请求仍是批量 1、贪心，与单副本时算法相同。全部副本载完才开端口，
任一副本进程退出则整个服务退出。

路由：``POST /info`` 回口径与层；``POST /ground_references`` 收 {annotation, references, images_path}，回 {status, record}；
``POST /plan_features`` 收 {annotation, span, images_path（各参考帧与段前末帧的 .npy）, references, last_shot,
target_prompts, starts, features_path}，教师强制预填，特征写进 features_path（npz 的 ``features``，与 starts 同序）。
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import multiprocessing
import multiprocessing.connection
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

ROUTES = ("ground_references", "plan_features")


def answer(reader, route: str, payload: dict) -> dict:
    """一条请求在一个副本上的回复；``ground_references`` 的失败随原答回传、由渲染进程落盘。"""
    import numpy as np

    if route == "ground_references":
        images = list(np.load(payload["images_path"]))
        try:
            record = reader.ground_references(payload["annotation"], references=payload["references"], images=images)
        except Exception as error:
            return {"status": "failed", "error": f"{type(error).__name__}: {error}",
                    "record": getattr(error, "record", {})}
        return {"status": "ok", "record": record}
    if route == "plan_features":
        images = list(np.load(payload["images_path"]))
        starts = [(i, n, p) for i, n, p in payload["starts"]]
        features = reader.plan_features(payload["annotation"], span=payload["span"], images=images,
                                        references=payload["references"], last_shot=payload["last_shot"],
                                        target_prompts=payload["target_prompts"], starts=starts)
        np.savez(payload["features_path"], features=np.asarray(features, dtype=np.float32))
        return {"status": "ok", "record": {}}
    raise ValueError(f"unknown route {route!r}")


def replica(device: str, settings: dict, parent: int, requests, replies) -> None:
    """一个副本进程：载一份模型，从共享队列逐条取请求。没有结构化失败的路由出错时回 ``error``，由前端回 500。"""
    import ctypes
    import signal

    # PR_SET_PDEATHSIG：前端进程按 pid 停掉或退出时，内核随之杀掉副本，不留占卡的孤儿。
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    if os.getppid() != parent:
        return
    import torch
    from mspc.inference.planner_token_reading import PlannerTokenReader

    torch.cuda.set_device(int(device))
    reader = PlannerTokenReader(model_path=settings["model"], device=f"cuda:{device}",
                                feature_pooling=settings["feature_pooling"], feature_layer=settings["feature_layer"],
                                max_new_tokens=settings["max_new_tokens"], cut_margin=settings["cut_margin"])
    replies.put(("ready", device, reader.runtime_info))
    while True:
        ticket, route, payload = requests.get()
        try:
            replies.put((ticket, "reply", answer(reader, route, payload)))
        except Exception as error:
            replies.put((ticket, "error", f"{type(error).__name__}: {error}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-pooling", required=True)
    parser.add_argument("--feature-layer", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--cut-margin", type=int, required=True, help="与标签构造器的 --cut-margin 同值")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--devices", required=True,
                        help="逗号分隔，每项一个副本进程（可见卡内的序号，同一张卡可重复），例如 0,0,0")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if any(not device.isdigit() for device in devices):
        raise SystemExit(f"--devices must be comma-separated card indices, got {args.devices!r}")
    settings = {"model": str(args.model), "feature_pooling": args.feature_pooling, "feature_layer": args.feature_layer,
                "max_new_tokens": args.max_new_tokens, "cut_margin": args.cut_margin}

    context = multiprocessing.get_context("spawn")
    requests, replies = context.Queue(), context.Queue()
    processes = [context.Process(target=replica, args=(device, settings, os.getpid(), requests, replies))
                 for device in devices]
    for process in processes:
        process.start()

    def watch() -> None:
        multiprocessing.connection.wait([process.sentinel for process in processes])
        dead = [device for device, process in zip(devices, processes) if not process.is_alive()]
        print(json.dumps({"event": "replica_exited", "devices": dead}), flush=True)
        for process in processes:
            process.kill()
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
    runtime = None
    for _ in processes:
        _, _, runtime = replies.get()
    tickets = itertools.count()
    pending: dict[int, list] = {}
    pending_lock = threading.Lock()

    def collect() -> None:
        while True:
            ticket, kind, value = replies.get()
            with pending_lock:
                slot = pending.pop(ticket)
            slot[1:] = [kind, value]
            slot[0].set()

    threading.Thread(target=collect, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, value, code=200):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            route = self.path.removeprefix("/")
            if route == "info":
                return self._reply({"feature_pooling": args.feature_pooling, "feature_layer": args.feature_layer,
                                    "cut_margin": args.cut_margin,
                                    "model": str(args.model), "max_new_tokens": args.max_new_tokens,
                                    "devices": devices, "runtime": runtime})
            if route not in ROUTES:
                return self.send_error(404)
            slot = [threading.Event(), None, None]
            ticket = next(tickets)
            with pending_lock:
                pending[ticket] = slot
            requests.put((ticket, route, payload))
            slot[0].wait()
            if slot[1] == "error":
                return self._reply({"status": "error", "error": slot[2]}, code=500)
            return self._reply(slot[2])

        def log_message(self, *values):
            pass

    print(json.dumps({"event": "serving", "port": args.port, "devices": devices}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
