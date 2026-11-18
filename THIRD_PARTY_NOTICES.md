# Third-Party Notices

This repository (`CaT-Movie`) is licensed under the **Apache License 2.0** (see [`LICENSE`](LICENSE)). Portions of the codebase and external dependencies incorporate or build upon third-party open-source projects, whose original notices and licenses remain authoritative for those components.

## MultiShotMaster-Derived Components

- **Upstream:** <https://github.com/KlingAIResearch/MultiShotMaster>
- **Pinned base commit:** `1df812dda262639e4a3ca6e4a1da9000b0a8e124`
- **License:** Apache-2.0 (`third_party/licenses/multishotmaster-LICENSE`)
- **Scope in this repository:** Multi-shot dataset adapter (`src/mspc/data/multishot.py`)
- **Patch boundary:** `third_party/patches/multishotmaster-dataset-to-storyboard.patch`

## Wan2.2-Derived Components

- **Upstream:** <https://github.com/Wan-Video/Wan2.2>
- **Pinned base commit:** `2677beaaf508dc2251dcf5056b0af1ef007b7623`
- **License:** Apache-2.0 (`third_party/licenses/Wan2.2-Apache-2.0.txt`)
- **Scope in this repository:** Official Wan2.2 model and pipeline integration (`src/mspc/inference/`, `src/mspc/training/`), with exact-copy upstream files verified against `third_party/upstreams.toml` (and `third_party/patches/wan22-generate-line-endings.patch`).

## DiffSynth-Studio-Derived Runtime

- **Upstream:** <https://github.com/modelscope/DiffSynth-Studio>
- **Base commit / version:** `6477e11e4a81a5521d0f24972bcabf80838f32da` (`2.0.16`)
- **License:** Apache-2.0 (`third_party/licenses/DiffSynth-Studio-Apache-2.0.txt`)
- **Scope in this repository:** `third_party/wan22_runtime/diffsynth/` (modified from the recorded base commit for CaT-Movie multi-shot reference-routed training and inference).

## Bernini & VeOmni Dependencies

- **Bernini:** <https://github.com/bytedance/Bernini> (`82d573e9a553dccb1405dde4cb0f26466f9df6c5`, Apache-2.0, `third_party/licenses/bernini-LICENSE`)
- **VeOmni:** <https://github.com/ByteDance-Seed/VeOmni> (`6ab293ecdfdd90ef3941fc81065d9f947b5b4e4f`, Apache-2.0, `third_party/licenses/veomni-LICENSE`)

## Prebuilt GPU Wheels

- **`flash-attn` (`2.7.4.post1`, CPython 3.10 Linux `x86_64`):** Not bundled in this repository or in `pyproject.toml`. Checksums and upstream wheel locations are recorded in `third_party/backend_artifacts.json` and verified by `tools/sync/sync_backend.py`. Subject to the upstream `flash-attn` license.

## Offline Upstream Provenance Check

All locked Python dependencies are recorded in `uv.lock`. To verify shipped third-party diffs locally against pinned upstream checkouts without network access:

```bash
python tools/audit/check_upstream_diff.py \
  --manifest third_party/upstreams.toml \
  --root . \
  --materialize-root /path/to/materialized-runtime \
  --materialize-collision replace \
  --source-root /path/to/pinned-checkouts \
  --git-bin "$(which git)"
```
