#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash run/train.sh CONFIG.{py,json} [CHECKPOINT] [--resume-state STATE_DIRECTORY] [--resume-migration MIGRATION_JSON] [--cache-stage STAGE_JSON] [--execution-mode <plan|run>]"
  echo ""
  echo "  CHECKPOINT       热启动：拿另一条臂的权重初始化一条新臂。步数从 0，优化器重置。"
  echo "  --resume-state   从同一配置/拓扑的完整训练状态接续，与 CHECKPOINT 互斥。"
  echo "                   仅 training.state_checkpoint 明确开启的运行产生此状态。"
  echo "  --resume-migration 严格绑定旧SHA/config/state的新生产入口迁移合同。"
  echo "  --cache-stage    外部缓存阶段计划，不改变完整采样表或训练配置。"
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_PATH="$1"
shift
CHECKPOINT_PATH=""
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  CHECKPOINT_PATH="$1"
  shift
fi
EXECUTION_MODE="run"
RESUME_STATE=""
RESUME_MIGRATION=""
CACHE_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execution-mode)
      EXECUTION_MODE="${2:?--execution-mode requires plan or run}"
      shift 2
      ;;
    --resume-state)
      RESUME_STATE="${2:?--resume-state requires a training-state directory}"
      shift 2
      ;;
    --resume-migration)
      RESUME_MIGRATION="${2:?--resume-migration requires an explicit migration JSON}"
      shift 2
      ;;
    --cache-stage)
      CACHE_STAGE="${2:?--cache-stage requires an operational stage JSON}"
      shift 2
      ;;
    *)
      echo "[train] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
if [[ "$EXECUTION_MODE" != "plan" && "$EXECUTION_MODE" != "run" ]]; then
  echo "[train] --execution-mode must be plan or run" >&2
  exit 2
fi
if [[ -n "$RESUME_STATE" ]]; then
  [[ -z "$CHECKPOINT_PATH" ]] || { echo "[train] checkpoint and resume-state are mutually exclusive" >&2; exit 2; }
  [[ -d "$RESUME_STATE" ]] || { echo "[train] resume-state directory not found: $RESUME_STATE" >&2; exit 2; }
  RESUME_STATE="$(cd "$RESUME_STATE" && pwd)"
fi

if [[ -n "$RESUME_MIGRATION" ]]; then
  [[ -n "$RESUME_STATE" ]] || { echo "[train] resume-migration requires resume-state" >&2; exit 2; }
  [[ -f "$RESUME_MIGRATION" ]] || { echo "[train] migration file not found: $RESUME_MIGRATION" >&2; exit 2; }
  RESUME_MIGRATION="$(cd "$(dirname "$RESUME_MIGRATION")" && pwd)/$(basename "$RESUME_MIGRATION")"
fi

if [[ -n "$CACHE_STAGE" ]]; then
  [[ -f "$CACHE_STAGE" ]] || { echo "[train] cache-stage file not found: $CACHE_STAGE" >&2; exit 2; }
  CACHE_STAGE="$(cd "$(dirname "$CACHE_STAGE")" && pwd)/$(basename "$CACHE_STAGE")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ ! -f "$CONFIG_PATH" ]]; then
  if [[ -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
  else
    echo "[train] config not found: $CONFIG_PATH" >&2
    exit 2
  fi
fi
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
if [[ -n "$CHECKPOINT_PATH" ]]; then
  if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    if [[ -f "$REPO_ROOT/$CHECKPOINT_PATH" ]]; then
      CHECKPOINT_PATH="$REPO_ROOT/$CHECKPOINT_PATH"
    else
      echo "[train] checkpoint not found: $CHECKPOINT_PATH" >&2
      exit 2
    fi
  fi
  CHECKPOINT_PATH="$(cd "$(dirname "$CHECKPOINT_PATH")" && pwd)/$(basename "$CHECKPOINT_PATH")"
fi

# Resolve which interpreter runs this launch.  Implementation and the
# reason it is stdlib-only live in run/lib/config_bootstrap.sh, shared with
# the other launcher so both entry points resolve the same interpreter.
BOOTSTRAP_LIB="$REPO_ROOT/run/lib/config_bootstrap.sh"
if [[ ! -r "$BOOTSTRAP_LIB" ]]; then
  echo "[train] missing launcher bootstrap library: $BOOTSTRAP_LIB" >&2
  exit 2
fi
# shellcheck source=lib/config_bootstrap.sh
source "$BOOTSTRAP_LIB"
PYTHON_BIN="$(mspc_resolve_runtime_python train "$CONFIG_PATH")"

# Authoritative validation under the configured interpreter.  It re-prints the
# resolved runtime.python so a bootstrap that read a different value than the
# real parser fails loud instead of silently launching under the wrong env.
RESOLVED_PYTHON="$(PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$CONFIG_PATH" <<'VALIDATE'
import sys

from mspc.config import load_experiment_config
from mspc.runs.launch import require_train_launch_dataset

config = load_experiment_config(sys.argv[1])
require_train_launch_dataset(config)
print(config.resolved["runtime"]["python"])
VALIDATE
)"

if [[ "$RESOLVED_PYTHON" != "$PYTHON_BIN" ]]; then
  echo "[train] bootstrap resolved runtime.python=$PYTHON_BIN but the config parser resolved $RESOLVED_PYTHON" >&2
  exit 2
fi
# Topology.  A platform that injects the rendezvous (the cluster scheduler) stays
# authoritative and is used verbatim.  A direct-SSH single-node launch injects
# nothing, so derive it here instead of making every caller hand-write five
# variables.  A partially injected environment is ambiguous and fails loud
# rather than mixing platform values with derived ones.
TOPOLOGY_VARS=(NPROC_PER_NODE WORLD_SIZE RANK MASTER_ADDR MASTER_PORT)
TOPOLOGY_SET=()
TOPOLOGY_UNSET=()
for name in "${TOPOLOGY_VARS[@]}"; do
  if [[ -n "${!name:-}" ]]; then TOPOLOGY_SET+=("$name"); else TOPOLOGY_UNSET+=("$name"); fi
done

if (( ${#TOPOLOGY_SET[@]} == ${#TOPOLOGY_VARS[@]} )); then
  TOPOLOGY_SOURCE="platform"
elif (( ${#TOPOLOGY_SET[@]} == 0 )); then
  TOPOLOGY_SOURCE="single_node_derived"
  # Never guess the device set: an unset CUDA_VISIBLE_DEVICES would mean "every
  # GPU on the box", which on a shared host reaches outside the lease.
  : "${CUDA_VISIBLE_DEVICES:?[train] no platform topology injected; set CUDA_VISIBLE_DEVICES to the lease ordinals so the launcher can derive NPROC_PER_NODE}"
  NPROC_PER_NODE="$(awk -F, '{n=0; for (i=1; i<=NF; i++) if ($i != "") n++; print n}' <<<"$CUDA_VISIBLE_DEVICES")"
  if [[ ! "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 )); then
    echo "[train] could not derive NPROC_PER_NODE from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  WORLD_SIZE=1
  RANK=0
  MASTER_ADDR=127.0.0.1
  # Pure bash so this does not depend on which python3 is first on PATH.
  # A listening socket answers the connect; a free port refuses it.
  MASTER_PORT=""
  for _ in $(seq 1 32); do
    candidate=$(( 20000 + (RANDOM % 20000) ))
    if (exec 9<>"/dev/tcp/127.0.0.1/$candidate") 2>/dev/null; then
      exec 9<&- 2>/dev/null || true
      exec 9>&- 2>/dev/null || true
      continue
    fi
    MASTER_PORT="$candidate"
    break
  done
  if [[ -z "$MASTER_PORT" ]]; then
    echo "[train] could not find a free MASTER_PORT in 20000-39999; set MASTER_PORT explicitly" >&2
    exit 2
  fi
  export NPROC_PER_NODE WORLD_SIZE RANK MASTER_ADDR MASTER_PORT
else
  echo "[train] partial topology injection: set=[${TOPOLOGY_SET[*]}] unset=[${TOPOLOGY_UNSET[*]}]" >&2
  echo "[train] provide all five or none; refusing to mix injected and derived values" >&2
  exit 2
fi

echo "[train] topology source=$TOPOLOGY_SOURCE nproc_per_node=$NPROC_PER_NODE nnodes=$WORLD_SIZE node_rank=$RANK master=$MASTER_ADDR:$MASTER_PORT"

# 显存分配器用可扩展段：只改 CUDA 显存怎么切块，不改任何计算。2026-09-14 训练集群 2 卡
# 多卡门（Slurm 1176755）在第 1 步 ZeRO-2 优化器更新里申请 9.31 GiB 失败，而当时有
# 13.20 GiB「已预留未分配」——是碎片而不是总量不够；2026-09-10 主服务器 2 卡门带着这一项通过。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Which interpreter runs the training.  The env registry decides whenever it
# has an entry for this role on the detected architecture, because the
# interpreter is a property of the hardware, not of the experiment: the same
# config is meant to run on several GPU architectures, and a compiled torch/flash-attn
# stack cannot be both.  The config's runtime.python bootstraps the parser
# (which needs an interpreter before it can read anything) and stays the
# fallback for a role×arch pair the registry does not cover.
#
# This used to be advisory, and on 2026-08-26 that cost a run: the launcher
# printed "registry suggests wan22_fun_sm80_cu128, config specifies wan22_fun;
# following the config" and then loaded a cu130 build on a driver that tops out
# at 12.9, so all eight ranks died after a full model load.  Knowing the right
# answer and using the wrong one is worse than not knowing.
#
# Detection runs after the topology block so CUDA_VISIBLE_DEVICES is already
# resolved: it looks at the leased devices only, never at the whole box.  When
# detection itself fails — a platform-injected launch may not expose the driver
# query tool at all — the config still wins, so nothing here can block a launch
# the config authorized.
GPU_ARCH_LIB="$REPO_ROOT/run/lib/gpu_arch.sh"
if [[ -r "$GPU_ARCH_LIB" ]]; then
  # shellcheck source=lib/gpu_arch.sh
  source "$GPU_ARCH_LIB"
  # Detection runs inside a command substitution so a lib that aborts rather
  # than returns on failure cannot take the training launch down with it.
  if DETECTED_ARCH="$(mspc_detect_gpu_arch >/dev/null && printf '%s' "${MSPC_GPU_ARCH:-}")" \
    && [[ -n "$DETECTED_ARCH" ]]; then
    export MSPC_GPU_ARCH="$DETECTED_ARCH"
    echo "[train] gpu arch=$MSPC_GPU_ARCH runtime.python=$PYTHON_BIN"
    if REGISTRY_PYTHON="$(mspc_env_python_for_role wan22_training)" && [[ -n "$REGISTRY_PYTHON" ]]; then
      if [[ "$REGISTRY_PYTHON" != "$PYTHON_BIN" ]]; then
        if [[ ! -x "$REGISTRY_PYTHON" ]]; then
          echo "[train] env registry names $REGISTRY_PYTHON for wan22_training on $MSPC_GPU_ARCH but it is not executable" >&2
          exit 2
        fi
        echo "[train] env registry routes wan22_training on $MSPC_GPU_ARCH to $REGISTRY_PYTHON; config named $PYTHON_BIN" >&2
        PYTHON_BIN="$REGISTRY_PYTHON"
      fi
    else
      echo "[train] warn: env registry names no wan22_training interpreter for $MSPC_GPU_ARCH; following the config runtime.python=$PYTHON_BIN" >&2
    fi
    # Optional registry hook.  Present only if run/lib/gpu_arch.sh exposes the
    # verification status; without it the run is unannotated rather than
    # blocked.  Called in a subshell for the same reason detection is: every
    # call in this block has to be unable to end a launch the config authorized,
    # and a lib that aborts instead of returning would otherwise do exactly that.
    if declare -F mspc_env_role_verified_on_arch >/dev/null \
      && ! ( mspc_env_role_verified_on_arch wan22_training ); then
      echo "[train] warn: wan22_training is not recorded as verified on $MSPC_GPU_ARCH; this launch is the first evidence for that pair" >&2
    fi
  else
    echo "[train] warn: GPU architecture detection failed; skipping the env-registry cross-check and following the config runtime.python=$PYTHON_BIN" >&2
  fi
else
  echo "[train] warn: $GPU_ARCH_LIB is missing; skipping the env-registry cross-check and following the config runtime.python=$PYTHON_BIN" >&2
fi

CMD=(
  env "PYTHONPATH=$REPO_ROOT/src"
  "$PYTHON_BIN" -m mspc train
  --config "$CONFIG_PATH"
  --execution-mode "$EXECUTION_MODE"
  --gpus-per-node "$NPROC_PER_NODE"
  --nnodes "$WORLD_SIZE"
  --node-rank "$RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
)
if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi
if [[ -n "$RESUME_STATE" ]]; then
  CMD+=(--resume-state "$RESUME_STATE")
fi
if [[ -n "$RESUME_MIGRATION" ]]; then
  CMD+=(--resume-migration "$RESUME_MIGRATION")
fi
if [[ -n "$CACHE_STAGE" ]]; then
  CMD+=(--cache-stage "$CACHE_STAGE")
fi

exec "${CMD[@]}"
