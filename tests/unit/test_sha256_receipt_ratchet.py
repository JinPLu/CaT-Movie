"""Receipt-class sha256 sites in src/ may only fall. Delete this file at 0."""

from __future__ import annotations

import re

from tests.support.paths import REPO_ROOT

RECEIPT_RE = re.compile(
    r"\bsha256_file\s*\(|\bpayload_sha256s\b|\binput_sha256\b|"
    r"\bcanonical_\w+_sha256\b|\brequire_hex64\s*\("
)
# Frozen after the 2026-08-28 scorer identity cut. Lower it when a site dies.
MAX_RECEIPT_SITES = 175


def _receipt_site_count() -> int:
    total = 0
    for path in (REPO_ROOT / "src").rglob("*.py"):
        total += len(RECEIPT_RE.findall(path.read_text(encoding="utf-8")))
    return total


def test_receipt_sha256_sites_only_fall() -> None:
    count = _receipt_site_count()
    if count == 0:
        raise AssertionError(
            "receipt-class sha256 sites are gone; delete tests/unit/test_sha256_receipt_ratchet.py"
        )
    assert count <= MAX_RECEIPT_SITES, (
        f"receipt-class sha256 sites rose from {MAX_RECEIPT_SITES} to {count}"
    )
