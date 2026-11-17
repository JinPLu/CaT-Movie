"""全量块标注的批次台账命名；历史试标台账不属于全量轮次。"""

from pathlib import Path


def annotation_round(path: Path) -> int | None:
    name = path.name
    if name == "block-annotation-batches.json":
        return 0
    prefix = "block-annotation-batches-r"
    tail = name.removeprefix(prefix).removesuffix(".json")
    return int(tail) if name.startswith(prefix) and name.endswith(".json") and tail.isdigit() else None


def block_batch_ledgers(directory: Path) -> list[tuple[int, Path]]:
    return sorted((number, path) for path in directory.glob("block-annotation-batches*.json")
                  if (number := annotation_round(path)) is not None)
