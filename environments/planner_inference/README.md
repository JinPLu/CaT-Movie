# Planner serving environment (vLLM)

The environment behind `run/vllm_serve.py`, separate from the renderer environment. `uv.lock` pins Linux, Python
3.11–3.12, vLLM 0.19.0, Transformers 5.8.0 and torch 2.10.0+cu128; the Transformers upper bound that vLLM declares is
overridden in `override-dependencies`.

## Install

Build it where the serving config's `vllm_executable` points, for example the planner of the paper
(`configs/serving/vllm_planner_qwen38_27b.yaml`):

```bash
UV_PROJECT_ENVIRONMENT=$MSPC_SHARED_ROOT/env/planner_inference_cu128 uv sync --project environments/planner_inference --locked
```

## Serve

```bash
PYTHONPATH=src python run/vllm_serve.py \
  --config configs/serving/vllm_planner_qwen38_27b.yaml \
  --provenance-dir /absolute/path/to/run/service
```

`run/vllm_serve.py` writes `provenance.json` into `--provenance-dir`: the effective command, the GPU selection, the
model, processor and tokenizer configs, the chat template and the Hugging Face revisions it can read. Benchmark
generation (`mspc.benchmark.generation plan`, `configs/infer/benchmark.json`) reads that record through
`BENCHMARK_PUBLIC_SERVICE_RECORD` and the client key from `BENCHMARK_PLANNER_API_KEY`. Raw planner responses are written
before parsing; malformed, truncated or interrupted answers are kept and not retried.
