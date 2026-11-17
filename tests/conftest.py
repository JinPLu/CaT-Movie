import json
import os
from pathlib import Path

import pytest
import torch  # Import the locked core dependency before legacy tests install stubs.


KNOWN_FAILURES_PATH = Path(__file__).with_name("known_failures.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
_existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(SOURCE_ROOT)
    if not _existing_pythonpath
    else os.pathsep.join((str(SOURCE_ROOT), _existing_pythonpath))
)
# Active wan22_5b p4 configs inject paths via ${MSPC_SHARED_ROOT}. Tests that
# only parse configs do not need the server tree; the historical prefix keeps
# resolved strings stable. Dedicated tests unset this to prove fail-loud.
os.environ.setdefault(
    "MSPC_SHARED_ROOT", "${MSPC_SHARED_ROOT}"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--validate-known-failures",
        action="store_true",
        help="require every strict known-failure node ID to be present in this collection",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Quarantine only the exact P1 baseline failures, with strict XPASS."""

    payload = json.loads(KNOWN_FAILURES_PATH.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise pytest.UsageError(f"invalid known-failure ledger: {KNOWN_FAILURES_PATH}")
    by_node = {entry["node_id"]: entry for entry in entries}
    if len(by_node) != len(entries):
        raise pytest.UsageError("known-failure ledger contains duplicate node IDs")
    collected = {item.nodeid for item in items}
    missing = sorted(set(by_node) - collected)
    if missing and config.getoption("--validate-known-failures"):
        raise pytest.UsageError(f"known-failure node IDs were not collected: {missing}")
    for item in items:
        entry = by_node.get(item.nodeid)
        if entry is not None:
            item.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason=f"{entry['owner']}: {entry['reason']}",
                )
            )
