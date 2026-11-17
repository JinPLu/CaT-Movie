from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "run" / "precompute.sh"


def test_merge_only_is_a_nonzero_rank_noop(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), str(tmp_path / "missing.json"), "--merge-only"],
        check=False,
        env={**os.environ, "RANK": "1"},
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "merge-only no-op on rank=1" in completed.stdout


def test_precompute_uses_one_process_per_injected_local_rank() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert ': "${NPROC_PER_NODE:?[precompute] platform must inject NPROC_PER_NODE}"' in script
    assert ': "${WORLD_SIZE:?[precompute] platform must inject WORLD_SIZE}"' in script
    assert ': "${RANK:?[precompute] platform must inject RANK}"' in script
    assert "for (( local_rank=0; local_rank<NPROC_PER_NODE; local_rank++ ))" in script
    assert "shard_index=$((RANK * NPROC_PER_NODE + local_rank))" in script
    # The lease is a device table, not a count: worker `local_rank` takes the
    # local_rank-th leased ordinal.  Writing `$local_rank` here would overwrite
    # the lease and run on physical cards 0..N-1.
    assert 'IFS=\',\' read -r -a RAW_LEASED_DEVICES <<<"$CUDA_VISIBLE_DEVICES"' in script
    assert 'CUDA_VISIBLE_DEVICES="${LEASED_DEVICES[local_rank]}"' in script
    assert 'CUDA_VISIBLE_DEVICES="$local_rank"' not in script
    assert 'wait "$pid"' in script
    assert "collision-policy" not in script


def test_precompute_requires_fresh_output_and_one_simple_node_barrier() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    fresh_index = script.index('if [[ -f "$OUTPUT_DIR/manifest.json" ]]')
    launch_index = script.index("pids=()")
    marker_index = script.index('touch "$RUN_BARRIER_DIR/node-$RANK.done"')
    merge_hint_index = script.index("After ALL nodes finish, merge once")

    assert fresh_index < launch_index < marker_index < merge_hint_index
    assert 'RUN_BARRIER_DIR="$OUTPUT_DIR/.nodes"' in script
    assert 'node-$expected_node.done' in script
    assert "--encode-batch-size" not in script
    assert "receipt" not in script.casefold()
    assert "sha256" not in script.casefold()
