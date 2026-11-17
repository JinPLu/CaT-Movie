from __future__ import annotations

import hashlib

from mspc.data.io import fsync_directory, prefixed_sha256_file, sha256_file


def test_file_sha256_helpers_share_one_digest_contract(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"storyboard\x00artifact")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert sha256_file(artifact) == expected
    assert prefixed_sha256_file(artifact) == f"sha256:{expected}"


def test_fsync_directory_accepts_a_directory(tmp_path) -> None:
    fsync_directory(tmp_path)
