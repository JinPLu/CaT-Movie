"""规划器服务命令与启动前来源快照；不把包版本等同于加载验收。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def service_command(config: Mapping) -> list[str]:
    command = [config["vllm_executable"], "serve", config["model_path"]]
    for field in ("host", "port", "tensor_parallel_size", "max_model_len", "gpu_memory_utilization", "dtype"):
        command.extend(["--" + field.replace("_", "-"), str(config[field])])
    if config["trust_remote_code"]:
        command.append("--trust-remote-code")
    command.extend(config["extra_cli_args"])
    return command


def save_service_provenance(config_path: Path, config: Mapping, output: Path) -> dict:
    """保存有效命令、目标环境包元数据和原始模板；只读模型文件，不加载模型。"""
    executable = Path(config["vllm_executable"])
    python = executable.parent / "python"
    # 用服务自己的解释器读取包元数据，不能拿launcher环境的版本代替。
    probe = """import importlib.metadata as m,json,sys
names=('vllm','transformers','torch','torchvision','tokenizers','pillow')
print(json.dumps({'python':sys.executable,'python_version':sys.version,
 'packages':{name:{'version':m.version(name),'location':str(m.distribution(name).locate_file(''))} for name in names}}))
"""
    runtime = json.loads(subprocess.check_output([str(python), "-c", probe], text=True))
    model = Path(config["model_path"])
    if not model.is_dir():
        raise FileNotFoundError(model)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "service.yaml")
    snapshots = output / "model_files"
    snapshots.mkdir()
    filenames = ("config.json", "generation_config.json", "preprocessor_config.json",
                 "processor_config.json", "tokenizer_config.json", "chat_template.jinja",
                 "video_preprocessor_config.json", "model.safetensors.index.json")
    files = []
    for name in filenames:
        source = model / name
        metadata = model / ".cache" / "huggingface" / "download" / (name + ".metadata")
        revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
        record = {"file": name, "present": source.is_file(), "download_revision": revision}
        if source.is_file():
            shutil.copy2(source, snapshots / name)
            record["size_bytes"] = source.stat().st_size
        files.append(record)
    result = {"captured_at": datetime.now(timezone.utc).isoformat(),
              "status": "prelaunch_metadata_only", "model_path": str(model),
              "command": service_command(config), "runtime": runtime,
              "vllm_script_shebang": executable.read_text().splitlines()[0],
              "environment": {key: os.environ.get(key) for key in
                  ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")},
              "model_files": files,
              "revision_scope": "download sidecars only; absent stays null, local edits and weight identity not verified"}
    (output / "provenance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
