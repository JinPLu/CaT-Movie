# Third-Party Notices

This repository contains or references third-party software. This notice is an
inventory aid; the license files shipped with each imported component remain
authoritative. The Storyboard / MSPC repository does not yet declare a project
license, so these third-party licenses must not be interpreted as licensing the
repository as a whole.

## MultiShotMaster-derived code

- Upstream: <https://github.com/KlingAIResearch/MultiShotMaster>
- Recorded base commit: `1df812dda262639e4a3ca6e4a1da9000b0a8e124`
- License: Apache-2.0
- Current local scope: the multi-shot dataset adapter `src/mspc/data/multishot.py`.
  The Wan2.1 execution surface (DiffSynth runtime, infer/train entries, the
  captioned-video helper) was retired with the Wan2.1 arm and is no longer
  imported or materialized.
- Canonical license copy: `third_party/licenses/multishotmaster-LICENSE`
- Patch boundary: `third_party/patches/multishotmaster-dataset-to-storyboard.patch`
- Status: the official base, imported-path ledger and exact rebuildable patch
  are locked in `third_party/upstreams.toml`. MultiShotMaster does not publish
  its underlying DiffSynth commit; the repository therefore treats MSM itself
  as the immediate immutable base and does not invent a DiffSynth historical pin.

## Wan2.2-derived code

- Upstream: <https://github.com/Wan-Video/Wan2.2>
- Recorded base commit:
  `2677beaaf508dc2251dcf5056b0af1ef007b7623`
- License: Apache-2.0
- Current local scope: materialized `.runtime/upstreams/wan22/official/` plus
  the integration owners in `src/mspc/{inference,training}/`. That materialized
  `wan/modules/model.py` still carries the 2026-07-16 Shot-RPE patch from
  `ad9934f3`, so the comparison baseline instead uses the unmodified clone
  `.runtime/upstreams/wan22/Wan2.2-1ea34ff/` (commit
  `1ea34ff48f87168174e12956e200b1d908b1c5ff`, all 87 tracked files verified).
- Canonical license copy: `third_party/licenses/Wan2.2-Apache-2.0.txt`
- Patch boundary declared by the current manifest: `wan/` is an
  `exact-copy`; `generate.py` has only the
  `third_party/patches/wan22-generate-line-endings.patch` normalized-diff
  entry.
- Status: the current ledger does not declare a semantic patch to
  `wan/modules/model.py`. The exact-copy and line-ending claims still require
  the pinned upstream checkout plus materialization-root verification; this
  notice does not claim that server-side verification has been completed.

## DiffSynth-Studio derived code

This project has modified DiffSynth-Studio and now maintains the modified
tree in Git.

- Upstream: <https://github.com/modelscope/DiffSynth-Studio>
- Recorded base commit: `6477e11e4a81a5521d0f24972bcabf80838f32da`
- Recorded version: `2.0.16`
- License: Apache-2.0
- License copy: `third_party/licenses/DiffSynth-Studio-Apache-2.0.txt`
- Local tree: `third_party/wan22_runtime/diffsynth/` is derived from that
  commit, **has been modified by this project**, and is maintained here
  rather than regenerated from a patch.
- Status: registered on the `diffsynth-studio-wan22-multishot-runtime` entry
  in `third_party/upstreams.toml` as provenance only (`kind = "dependency"`:
  origin URL, base commit, licence, licence path). The checker does not
  rebuild or fingerprint this tree. Configs point
  `model.diffsynth_root` at `third_party/wan22_runtime`.

## Bernini and VeOmni dependency sources

- Bernini: <https://github.com/bytedance/Bernini>, pinned dependency commit
  `82d573e9a553dccb1405dde4cb0f26466f9df6c5`, Apache-2.0; license copy
  `third_party/licenses/bernini-LICENSE`.
- VeOmni: <https://github.com/ByteDance-Seed/VeOmni>, pinned dependency commit
  `6ab293ecdfdd90ef3941fc81065d9f947b5b4e4f`, Apache-2.0; license copy
  `third_party/licenses/veomni-LICENSE`.

## GPU binary artifacts

- `gpu-wan` uses the upstream flash-attn 2.7.4.post1 CPython 3.10 Linux x86_64
  wheel exclusively through the checked artifact workflow; it is absent from
  the gpu-wan `pyproject.toml` extra and `uv.lock`.
- Canonical binary filename, upstream release URL and SHA-256 are recorded in
  `third_party/backend_artifacts.json`.
- The wheel is not redistributed in this repository. Operators must supply it
  through an explicit external wheelhouse; `tools/sync/sync_backend.py` verifies the
  filename and digest before installation, then checks installed version/import.
- The artifact manifest records binary provenance and does not replace the
  flash-attn upstream license. gpu-bernini's independent 2.8.3 dependency
  remains lock-managed.

Dependency packages and their transitive licenses are recorded in `uv.lock`.
Before redistribution, generate a dependency-license report and review it
together with this source inventory.

## Offline verification

Clone each entry from `third_party/upstreams.toml` at its exact commit into an
explicit source root, then run:

```bash
python tools/audit/check_upstream_diff.py \
  --manifest third_party/upstreams.toml \
  --root . \
  --materialize-root /absolute/path/to/materialized-runtime \
  --materialize-collision replace \
  --source-root /absolute/path/to/pinned-checkouts \
  --git-bin /absolute/path/to/git
```

The checker is network-free: it rejects the wrong checkout HEAD, license or
patch hash, unappliable patches, and any rebuilt tree that differs byte-for-byte
from the declared local boundary. The DiffSynth-derived tree in
`third_party/wan22_runtime/diffsynth/` is provenance-only in the manifest and
is outside the rebuild path.
