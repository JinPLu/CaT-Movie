#!/usr/bin/env bash
# Embarrassingly-parallel latent/context precompute for Wan2.2 P4 multi-shot
# training. One process per local GPU, no torchrun / NCCL: each process owns the
# rows where (row_index % NUM_SHARDS == SHARD_INDEX). Merged manifest.json must
# be absent. Existing valid latents/{case_id}.safetensors files are skipped after
# a fail-loud header check; empty/truncated files are unlinked and recomputed;
# corrupt keys abort. A complete shard JSON is idempotent on re-run.
# Decode workers, pin/prefetch, and VAE encode batch size come from
# config.precompute. experiment.dataloader.per_device_batch_size stays 1
# for training and is not the VAE batch.
# Resource scale comes from the platform-injected topology, exactly like
# run/train.sh.
set -euo pipefail

usage() {
  echo "Usage: bash run/precompute.sh CONFIG.json [--output-dir DIR] [--merge-only]"
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_PATH="$1"
shift
OUTPUT_DIR=""
MERGE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a path}"
      shift 2
      ;;
    --merge-only)
      MERGE_ONLY=1
      shift
      ;;
    *)
      echo "[precompute] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
EARLY_RANK="${RANK:-0}"
if [[ "$MERGE_ONLY" == "1" && "$EARLY_RANK" != "0" ]]; then
  echo "[precompute] merge-only no-op on rank=$EARLY_RANK"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ ! -f "$CONFIG_PATH" ]]; then
  if [[ -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
  else
    echo "[precompute] config not found: $CONFIG_PATH" >&2
    exit 2
  fi
fi
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

# Resolve which interpreter runs this launch.  Implementation and the
# reason it is stdlib-only live in run/lib/config_bootstrap.sh, shared with
# the other launcher so both entry points resolve the same interpreter.
BOOTSTRAP_LIB="$REPO_ROOT/run/lib/config_bootstrap.sh"
if [[ ! -r "$BOOTSTRAP_LIB" ]]; then
  echo "[precompute] missing launcher bootstrap library: $BOOTSTRAP_LIB" >&2
  exit 2
fi
# shellcheck source=lib/config_bootstrap.sh
source "$BOOTSTRAP_LIB"
PYTHON_BIN="$(mspc_resolve_runtime_python precompute "$CONFIG_PATH")"

# Authoritative validation under the configured interpreter.  It re-prints the
# resolved runtime.python so a bootstrap that read a different value than the
# real parser fails loud instead of silently launching under the wrong env.
CONFIG_ENV="$(PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import shlex
import sys

from mspc.config import load_experiment_config
from mspc.runs.launch import require_precompute_launch_dataset

captured = StringIO()
with redirect_stdout(captured), redirect_stderr(captured):
    loaded = load_experiment_config(sys.argv[1])
try:
    require_precompute_launch_dataset(loaded)
except ValueError as exc:
    print(f"[precompute] {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
config = loaded.resolved
python_bin = config["runtime"]["python"]
dataset_key = config["identity"]["dataset_key"]
train_pythonpath = config["launch"]["train_pythonpath"]
if not isinstance(python_bin, str) or not python_bin:
    raise SystemExit("[precompute] config value is missing: runtime.python")
if not train_pythonpath:
    raise SystemExit("[precompute] config value is missing: launch.train_pythonpath")
print(f"RESOLVED_PYTHON={shlex.quote(python_bin)}")
print(f"DATASET_KEY={shlex.quote(dataset_key)}")
print(f"TRAIN_PYTHONPATH={shlex.quote(':'.join(train_pythonpath))}")
PY
)"
eval "$CONFIG_ENV"

if [[ "$RESOLVED_PYTHON" != "$PYTHON_BIN" ]]; then
  echo "[precompute] bootstrap resolved runtime.python=$PYTHON_BIN but the config parser resolved $RESOLVED_PYTHON" >&2
  exit 2
fi

# Which interpreter runs the encode.  Identical routing to run/train.sh, from
# the same table and the same role: the interpreter is a property of the
# hardware, not of the experiment, and precompute loads the same torch and the
# same Wan VAE that training does.  Leaving this out is what made a config
# naming env/wan22_fun die on A100 with "the NVIDIA driver on your system is
# too old" while the registry already recorded wan22_fun_sm80_cu128 for sm80.
# Every call below is scoped to the leased devices (CUDA_VISIBLE_DEVICES) and
# none of them can end a launch the config authorized.
GPU_ARCH_LIB="$REPO_ROOT/run/lib/gpu_arch.sh"
if [[ -r "$GPU_ARCH_LIB" ]]; then
  # shellcheck source=lib/gpu_arch.sh
  source "$GPU_ARCH_LIB"
  # Detection runs inside a command substitution so a lib that aborts rather
  # than returns on failure cannot take the precompute launch down with it.
  if DETECTED_ARCH="$(mspc_detect_gpu_arch >/dev/null && printf '%s' "${MSPC_GPU_ARCH:-}")" \
    && [[ -n "$DETECTED_ARCH" ]]; then
    export MSPC_GPU_ARCH="$DETECTED_ARCH"
    echo "[precompute] gpu arch=$MSPC_GPU_ARCH runtime.python=$PYTHON_BIN"
    if REGISTRY_PYTHON="$(mspc_env_python_for_role wan22_training)" && [[ -n "$REGISTRY_PYTHON" ]]; then
      if [[ "$REGISTRY_PYTHON" != "$PYTHON_BIN" ]]; then
        if [[ ! -x "$REGISTRY_PYTHON" ]]; then
          echo "[precompute] env registry names $REGISTRY_PYTHON for wan22_training on $MSPC_GPU_ARCH but it is not executable" >&2
          exit 2
        fi
        echo "[precompute] env registry routes wan22_training on $MSPC_GPU_ARCH to $REGISTRY_PYTHON; config named $PYTHON_BIN" >&2
        PYTHON_BIN="$REGISTRY_PYTHON"
      fi
    else
      echo "[precompute] warn: env registry names no wan22_training interpreter for $MSPC_GPU_ARCH; following the config runtime.python=$PYTHON_BIN" >&2
    fi
  else
    echo "[precompute] warn: GPU architecture detection failed; skipping the env-registry cross-check and following the config runtime.python=$PYTHON_BIN" >&2
  fi
else
  echo "[precompute] warn: $GPU_ARCH_LIB is missing; skipping the env-registry cross-check and following the config runtime.python=$PYTHON_BIN" >&2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$REPO_ROOT/outputs/datasets/latent_cache/$DATASET_KEY"
fi

if [[ "$MERGE_ONLY" == "1" ]]; then
  echo "[precompute] merge-only config=$CONFIG_PATH output_dir=$OUTPUT_DIR rank=$EARLY_RANK"
  env PYTHONPATH="$TRAIN_PYTHONPATH" \
    "$PYTHON_BIN" -m mspc data-merge-latent-cache \
    --output-dir "$OUTPUT_DIR"
  exit $?
fi

: "${NPROC_PER_NODE:?[precompute] platform must inject NPROC_PER_NODE}"
: "${WORLD_SIZE:?[precompute] platform must inject WORLD_SIZE}"
: "${RANK:?[precompute] platform must inject RANK}"
NUM_SHARDS=$((WORLD_SIZE * NPROC_PER_NODE))
RUN_BARRIER_DIR="$OUTPUT_DIR/.nodes"
ready_marker="$RUN_BARRIER_DIR/ready"
if (( RANK == 0 )); then
  if [[ -f "$OUTPUT_DIR/manifest.json" ]]; then
    echo "[precompute] output already merged: $OUTPUT_DIR" >&2
    exit 2
  fi
  mkdir -p "$RUN_BARRIER_DIR"
  : > "$ready_marker"
else
  while [[ ! -f "$ready_marker" ]]; do
    sleep 2
  done
fi

echo "[precompute] config=$CONFIG_PATH output_dir=$OUTPUT_DIR rank=$RANK world_size=$WORLD_SIZE nproc_per_node=$NPROC_PER_NODE"

# The lease is a device *table*, not a count.  Worker `local_rank` must get the
# local_rank-th entry of CUDA_VISIBLE_DEVICES; writing `$local_rank` there would
# overwrite the lease with physical ordinals 0..N-1 and walk onto somebody
# else's cards whenever the broker hands out anything but 0,1,2,...  Indexing is
# also what torchrun does for run/train.sh, which is why that launcher passes the
# whole list through untouched -- one convention, two entry points.
: "${CUDA_VISIBLE_DEVICES:?[precompute] no leased devices; set CUDA_VISIBLE_DEVICES to the lease ordinals (GPU scheduler returns them as cuda_visible_devices)}"
IFS=',' read -r -a RAW_LEASED_DEVICES <<<"$CUDA_VISIBLE_DEVICES"
LEASED_DEVICES=()
for raw_device in "${RAW_LEASED_DEVICES[@]}"; do
  device="${raw_device//[[:space:]]/}"
  if [[ -n "$device" ]]; then
    LEASED_DEVICES+=("$device")
  fi
done
if (( ${#LEASED_DEVICES[@]} < NPROC_PER_NODE )); then
  echo "[precompute] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES names ${#LEASED_DEVICES[@]} devices but NPROC_PER_NODE=$NPROC_PER_NODE workers were requested" >&2
  exit 2
fi

pids=()
for (( local_rank=0; local_rank<NPROC_PER_NODE; local_rank++ )); do
  shard_index=$((RANK * NPROC_PER_NODE + local_rank))
  env PYTHONPATH="$TRAIN_PYTHONPATH" CUDA_VISIBLE_DEVICES="${LEASED_DEVICES[local_rank]}" \
    "$PYTHON_BIN" -m mspc data-precompute-latents \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --shard-index "$shard_index" \
    --num-shards "$NUM_SHARDS" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[precompute] one or more shards failed on rank $RANK" >&2
  exit 1
fi

touch "$RUN_BARRIER_DIR/node-$RANK.done"
if (( RANK == 0 )); then
  while true; do
    missing_nodes=()
    for (( expected_node=0; expected_node<WORLD_SIZE; expected_node++ )); do
      if [[ ! -f "$RUN_BARRIER_DIR/node-$expected_node.done" ]]; then
        missing_nodes+=("$expected_node")
      fi
    done
    if (( ${#missing_nodes[@]} == 0 )); then
      break
    fi
    sleep 10
  done
  echo "[precompute] all $WORLD_SIZE nodes completed"
fi

echo "[precompute] After ALL nodes finish, merge once:"
echo "[precompute]   bash run/precompute.sh $CONFIG_PATH --merge-only"
