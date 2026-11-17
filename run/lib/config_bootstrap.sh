# Shared launcher bootstrap: resolve `runtime.python` from a config with the
# standard library only.
#
# Reading `runtime.python` through mspc.config would need that interpreter's
# dependencies already installed -- the chicken-and-egg that made the launchers
# fail on any host whose bare `python3` lacks them.  So this resolver is
# stdlib-only, and the authoritative validation happens afterwards under the
# interpreter it names (each caller does that itself, because the two launchers
# validate different things).
#
# run/train.sh and run/precompute.sh must resolve the interpreter identically:
# the interpreter is a property of the hardware and the config, not of the entry
# point, and a precompute that reads a different value than the training run it
# feeds would produce a cache under the wrong torch/VAE build.  One convention,
# one implementation.  Only the log tag differs, so it is an argument.
#
# Usage:  PYTHON_BIN="$(mspc_resolve_runtime_python train "$CONFIG_PATH")"
# Prints the resolved interpreter path on stdout; returns 2 on any failure with
# the reason on stderr.

mspc_resolve_runtime_python() {
  local tag="$1"
  local config_path="$2"
  local bootstrap_python="${MSPC_BOOTSTRAP_PYTHON:-python3}"
  local resolved
  resolved="$("$bootstrap_python" - "$tag" "$config_path" <<'BOOTSTRAP'
import json
import os
import runpy
import sys

TAG = sys.argv[1]


def fail(message):
    sys.stderr.write("[" + TAG + "] " + message + "\n")
    raise SystemExit(2)


def expand(value):
    token = "${MSPC_SHARED_ROOT}"
    if token not in value:
        return value
    shared_root = os.environ.get("MSPC_SHARED_ROOT")
    if not shared_root:
        fail("config runtime.python contains %s but MSPC_SHARED_ROOT is unset" % token)
    return value.replace(token, shared_root)


path = os.path.abspath(sys.argv[2])
try:
    if path.endswith(".py"):
        namespace = runpy.run_path(path, run_name="__mspc_experiment_config__")
        document = namespace["EXPERIMENT_CONFIG"] if "EXPERIMENT_CONFIG" in namespace else namespace
    else:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
except (OSError, ValueError) as error:
    fail("could not read config %s: %s" % (path, error))
runtime = document.get("runtime")
if not isinstance(runtime, dict) or not runtime.get("python"):
    fail("config must declare runtime.python: %s" % path)
print(expand(runtime["python"]))

BOOTSTRAP
)" || return $?

  if [[ -z "$resolved" ]]; then
    echo "[$tag] could not resolve runtime.python from $config_path" >&2
    return 2
  fi
  if [[ ! -x "$resolved" ]]; then
    echo "[$tag] configured runtime python is not executable: $resolved" >&2
    return 2
  fi
  printf '%s\n' "$resolved"
}
