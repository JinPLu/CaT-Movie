"""Required shared-workspace root for injected experiment paths.

Two tokens, one owner (:func:`expand_shared_root_token`): ``${MSPC_SHARED_ROOT}`` is the host data root
(models, environments, inputs, outputs) and ``${MSPC_CODE_ROOT}`` is the checkout of the code that is
running. Frozen code roots hold only code, so no per-run symlink joins the two.

Active ``configs/train/wan22_5b/p4/*.json`` store filesystem paths as
``${MSPC_SHARED_ROOT}/...``. ``load_experiment_config`` expands that token and
raises if the env is missing.

This is a thin pydantic surface for the new injection contract only. The rest of
the 1762-line experiment validator stays hand-written.

Remaining absolute ``${MSPC_SHARED_ROOT}`` literals (Bernini
train, P3, runtime media-ready, eval MSVBench, archive, serving) are not
rewritten here. Their loaders do not expand this token. The two consumers that
read baked main-server paths on another surface go through :func:`reroot_source_path`
instead: ``latent_cache_dataset`` (CSV cells) and ``bernini_dev32_four_arm``
(DEV32 config runtime paths and the fixed-reference manifest's image paths).
Server value:

    export MSPC_SHARED_ROOT=${MSPC_SHARED_ROOT}
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, field_validator


SHARED_ROOT_ENV = "MSPC_SHARED_ROOT"
SHARED_ROOT_TOKEN = f"${{{SHARED_ROOT_ENV}}}"
HISTORICAL_SHARED_ROOT = "${MSPC_SHARED_ROOT}"
#: 代码路径的记号：展开成正在运行的这份代码的检出根，不经数据根。
CODE_ROOT_TOKEN = "${MSPC_CODE_ROOT}"
#: 本模块所在检出（``src/mspc/config/shared_root.py`` 往上三级）。
RUNNING_CODE_ROOT = Path(__file__).resolve().parents[3]
#: 经数据根指代码：``${MSPC_SHARED_ROOT}/Storyboard`` 本身或其下 src/third_party/tools/run/configs。
_CODE_THROUGH_DATA_ROOT = re.compile(
    r"\$\{MSPC_SHARED_ROOT\}/Storyboard(?=$|[\"'\s,;:]|/(?:src|third_party|tools|run|configs)(?:$|[/\"'\s,;:]))")


class SharedRootError(ValueError):
    """Raised when the shared-root env is missing or not an absolute path."""


class SharedRootSettings(BaseModel):
    """Validated ``MSPC_SHARED_ROOT`` used to expand injected config paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    root: Path

    @field_validator("root")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"{SHARED_ROOT_ENV} must be an absolute path")
        return value

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> SharedRootSettings:
        env = os.environ if environ is None else environ
        raw = env.get(SHARED_ROOT_ENV)
        if raw is None or not str(raw).strip():
            raise SharedRootError(
                f"{SHARED_ROOT_ENV} must be set to the shared workspace root "
                f"(server: {HISTORICAL_SHARED_ROOT})"
            )
        try:
            return cls(root=Path(str(raw).strip()).expanduser())
        except ValueError as exc:
            raise SharedRootError(str(exc)) from exc


def expand_shared_root_token(
    raw: str,
    location: str,
    *,
    environ: Mapping[str, str] | None = None,
    code_root: str | Path | None = None,
) -> str:
    """``${MSPC_CODE_ROOT}`` 换成代码根，``${MSPC_SHARED_ROOT}`` 换成本机数据根；数据根缺失就报错。

    代码根默认是正在运行的这份检出，调用方为另一份代码展开它的配置时显式给 ``code_root``。数据根只放数据
    （模型、环境、输入、输出），冻结代码根里只放代码，两者之间不建软链。经数据根指代码
    （``${MSPC_SHARED_ROOT}/Storyboard/src`` 之类）只在数据根下的检出就是这份代码时放行——旧的按提交根
    （代码与软链同在一个根下）如此；否则报错，免得悄悄读到另一份代码。
    """

    code = (RUNNING_CODE_ROOT if code_root is None else Path(code_root)).resolve()
    expanded = raw.replace(CODE_ROOT_TOKEN, str(code))
    if SHARED_ROOT_TOKEN not in expanded:
        return expanded
    root = SharedRootSettings.from_env(environ).root
    if _CODE_THROUGH_DATA_ROOT.search(expanded) and (root / "Storyboard").resolve() != code:
        raise SharedRootError(
            f"{location} reaches code through {SHARED_ROOT_TOKEN}/Storyboard, but {root / 'Storyboard'} "
            f"is not the running code {code}; write {CODE_ROOT_TOKEN} for code paths")
    expanded = expanded.replace(SHARED_ROOT_TOKEN, str(root))
    if SHARED_ROOT_TOKEN in expanded:
        raise SharedRootError(
            f"{location} still contains {SHARED_ROOT_TOKEN} after expansion"
        )
    return expanded


def code_paths_to_code_token(raw: str) -> str:
    """把经数据根指代码的写法（``${MSPC_SHARED_ROOT}/Storyboard``、其下 src/third_party/tools/run/configs）换成
    ``${MSPC_CODE_ROOT}``，数据路径（``${MSPC_SHARED_ROOT}/Storyboard/outputs``、``OoVMetric``、``env``）不动。出片入口
    从旧行派生新行时用它，新行的代码就是运行它的冻结代码根。"""
    return _CODE_THROUGH_DATA_ROOT.sub(lambda _: CODE_ROOT_TOKEN, raw)


def reroot_source_path(raw: str, *, location: str) -> str:
    """把产出时烤死的主服务器绝对路径重定到当前 ``MSPC_SHARED_ROOT``。

    两类产物在产出时写死了主服务器前缀 ``${MSPC_SHARED_ROOT}/...``，
    而 ``${MSPC_SHARED_ROOT}`` 展开只作用于写了 token 的 config 值：

    * ``metadata.csv`` 的 ``caption_path`` / ``video`` 等列。训练集群作业 1158198 在
      ``INIT dataset+model`` 之后 77 秒死于 ``FileNotFoundError``，缺的那份 caption
      在训练集群其实存在——差的就是这一次重定根。
    * DEV32 四臂配置的 ``checkpoint_dir`` / ``upstream_root`` / ``veomni_root`` /
      ``ffmpeg_bin`` 与 ``fixed_reference_manifest.json`` 里 81 张参考图的绝对路径。

    只改**带历史前缀**的值，且不改产物文件本身（改 CSV/manifest 会破坏它与 latent
    manifest、dev-96 建表记录、DEV32 出片记录的一致性）。主服务器上 ``MSPC_SHARED_ROOT``
    就是历史前缀，重定根是恒等；训练集群上它换成训练集群根。不带这个前缀的值原样返回：它从来
    不是主服务器烤死的路径，真不存在时由下游读取方在同一个装载期抛错。
    """

    if not raw.startswith(HISTORICAL_SHARED_ROOT):
        return raw
    root = SharedRootSettings.from_env().root
    relative = raw[len(HISTORICAL_SHARED_ROOT) :].lstrip("/")
    if not relative:
        raise ValueError(f"{location}: the value is the bare historical root: {raw}")
    return str(root / relative)
