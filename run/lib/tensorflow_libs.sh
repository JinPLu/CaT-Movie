#!/usr/bin/env bash
# LD_LIBRARY_PATH seeding for the TensorFlow half of the public-metrics stack.
#
# Sourced, never executed:
#
#   source run/lib/tensorflow_libs.sh
#   mspc_seed_tensorflow_libs "$PYTHON_BIN" "[label]"   # exports LD_LIBRARY_PATH
#
# Why this exists.  `transnet_transition_deviation` -- one of the five frozen
# public metrics -- runs TransNetV2 on TensorFlow, and TensorFlow finds CUDA
# through the *scoring env's own* `nvidia/*/lib` directories rather than
# through the driver alone.  Without them on LD_LIBRARY_PATH,
# `tf.config.list_physical_devices("GPU")` comes back empty and
# `_bind_tensorflow_device` raises
#
#     TransNetV2 requested cuda but TensorFlow sees 0 GPU(s)
#
# The expensive part is *when* it raises.  The TransNet engine is built lazily,
# inside the per-case scoring loop, so a run that gets this wrong decodes --
# and, when generation and scoring share one launch, generates -- every video
# first and dies at the scoring step: an hour and a half of sampling thrown
# away for a missing path.
#
# Derive from the interpreter path, never from an env *name*.  Which
# public-metrics env this is depends on the leased card's architecture
# (e.g. sm_80, sm_120, see run/lib/gpu_arch.sh), so a hard-wired
# name is correct on exactly one of the two.
#
# This function was three copies before it was one: two launchers each carried
# the derivation, while a third carried none and relied on whichever caller
# happened to have seeded first -- so invoking it directly, or running
# tools/run/run_renderer_qualification_public_metrics.py by hand, hit the late
# failure with nothing to catch it.  A Python-side preflight now
# refuses such a run in seconds (see OfficialMetricBackend.preflight); this
# file is what makes the refusal unnecessary in the first place.
#
# Not a fallback: it fails loudly rather than continuing on CPU, because a
# silent CPU TransNetV2 would still produce numbers and they would land in the
# frozen five.

# Exports LD_LIBRARY_PATH with the interpreter's nvidia lib dirs prepended.
# $1 = interpreter path (must end in /bin/python)
# $2 = optional log label, defaults to "tf-libs"
# Returns non-zero and explains the specific problem when it cannot.
mspc_seed_tensorflow_libs() {
  local python_bin="${1:-}"
  local label="${2:-tf-libs}"

  if [[ -z "$python_bin" ]]; then
    echo "[$label] mspc_seed_tensorflow_libs needs the interpreter path as \$1" >&2
    return 2
  fi

  local env_dir="${python_bin%/bin/python}"
  if [[ "$env_dir" == "$python_bin" ]]; then
    echo "[$label] expected the public-metrics interpreter to end in /bin/python, got $python_bin" >&2
    return 2
  fi

  # Ask the interpreter itself.  A glob of $env_dir/lib/python*/site-packages
  # misses chained .pth mounts (renderer_public_metrics hangs nvidia wheels
  # from env/bernini_r1), so this helper used to be a permanent false negative
  # and evaluate.sh refused every run in seconds.  site.getsitepackages() plus
  # sys.path is the interpreter's own view, including those .pth entries.
  #
  # This only unblocks the false negative.  It does not override a DT_RPATH
  # pinned by torch; TransNet now runs in a TF-only subprocess that never
  # imports torch.
  #
  # Iterating the glob on each reported dir, rather than collecting first,
  # stays correct under the callers' `set -u`: no match leaves the pattern
  # literal, the -d test rejects it, and nothing is added.
  local site_output
  if ! site_output="$("$python_bin" -c 'import site, sys
paths = []
for candidate in (*site.getsitepackages(), *sys.path):
    if candidate and candidate not in paths:
        paths.append(candidate)
print("\n".join(paths))')"; then
    echo "[$label] $python_bin could not report site-packages via site.getsitepackages(); cannot seed TensorFlow CUDA libs" >&2
    return 2
  fi

  local dirs=()
  local site_dir candidate
  while IFS= read -r site_dir; do
    [[ -z "$site_dir" ]] && continue
    for candidate in "$site_dir"/nvidia/*/lib; do
      [[ -d "$candidate" ]] && dirs+=("$candidate")
    done
  done <<< "$site_output"

  if (( ${#dirs[@]} == 0 )); then
    echo "[$label] no nvidia/*/lib dirs in site-packages reported by $python_bin; TransNetV2 would see 0 GPUs and the run would fail only after decoding every video" >&2
    return 2
  fi

  local nvlibs
  printf -v nvlibs '%s:' "${dirs[@]}"
  nvlibs="${nvlibs%:}"

  export LD_LIBRARY_PATH="${nvlibs}:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
  echo "[$label] LD_LIBRARY_PATH seeded with ${#dirs[@]} nvidia lib dir(s) from $env_dir" >&2
  return 0
}
