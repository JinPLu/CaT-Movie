"""参考帧库：三候选取帧、CLIP 去重、产物元数据核对。

去重和取帧都是纯数学函数，这里用合成向量；镜组取自真实 manifest 行
ci_vid:test:000000012 的 latent_shot_groups [[0, 13], [13, 21]]。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from mspc.data import reference_frame_latents as rfl
from tests.support.cache_provenance import provenance

GROUPS = [[0, 13], [13, 21]]


def test_frame_index_takes_first_middle_last_of_a_4n_plus_1_shot():
    assert [rfl.frame_index(p, 49) for p in rfl.FRAME_POSITIONS] == [1, 24, 48]
    assert rfl.shot_candidates(list(range(5))) == {"first": 1, "middle": 2, "last": 4}
    assert rfl.shot_candidates(list(range(2))) == {"first": 1}
    with pytest.raises(ValueError):
        rfl.frame_index("first", 1)
    assert rfl.shot_candidates(list(range(29))) == {"first": 1, "middle": 14, "last": 28}
    with pytest.raises(ValueError):
        rfl.frame_index("end", 5)


def _unit(*values):
    vector = torch.tensor(values, dtype=torch.float32)
    return vector / vector.norm()


def test_admission_drops_candidates_above_cosine_point_nine_against_the_library_and_itself():
    library = torch.stack([_unit(1, 0, 0)])
    candidates = [_unit(1, 0.4, 0), _unit(0, 1, 0), _unit(0, 1, 0.2)]
    # cos(1,0.4)=0.928 > 0.9 被库挡；第二个入库；第三个与刚入库的第二个 cos=0.98 被挡。
    assert rfl.admitted(candidates, library) == [1]
    assert rfl.admitted([_unit(1, 0.5, 0)], library) == [0]  # cos=0.894 ≤ 0.9
    assert rfl.admitted(candidates, torch.zeros((0, 3))) == [0, 1]


def test_library_encodes_only_admitted_frames():
    vectors = {"a": _unit(1, 0, 0), "b": _unit(1, 0.1, 0), "c": _unit(0, 1, 0), "d": _unit(0, 0, 1)}
    library = rfl.ReferenceLibrary()
    embed = lambda images: torch.stack([vectors[i] for i in images])  # noqa: E731
    encoded = []
    encode = lambda image: encoded.append(image) or image  # noqa: E731
    first = library.add_shot(0, {"first": "a", "middle": "b", "last": "c"}, embed=embed, encode=encode)
    second = library.add_shot(2, {"first": "a", "middle": "c", "last": "d"}, embed=embed, encode=encode)
    assert [(f.shot_index, f.position, f.latent) for f in first] == [(0, "first", "a"), (0, "last", "c")]
    assert [(f.shot_index, f.position, f.latent) for f in second] == [(2, "last", "d")]
    assert encoded == ["a", "c", "d"]  # 被去重挡掉的候选不编码
    with pytest.raises(ValueError, match="increasing order"):
        library.add_shot(1, {"first": "d", "middle": "d", "last": "d"}, embed=embed, encode=encode)


def test_clip_queries_do_not_silently_truncate_and_share_the_image_models_space():
    from transformers import BatchEncoding

    embedder = object.__new__(rfl.ClipImageEmbedder)
    calls = []

    def tokenize(**kwargs):
        calls.append(kwargs)
        return BatchEncoding({"input_ids": torch.zeros((1, len(kwargs["text"][0].split())), dtype=torch.long)})

    embedder.processor = tokenize
    embedder.device = torch.device("cpu")
    embedder.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=3)),
                                    get_text_features=lambda **kwargs: torch.tensor([[3.0, 4.0]]))
    assert torch.allclose(embedder.encode_text(["red container"]), torch.tensor([[0.6, 0.8]]))
    assert calls[0]["truncation"] is False
    with pytest.raises(ValueError, match="exceeding 3"):
        embedder.encode_text(["a query that exceeds capacity"])


def _record():
    return {"case_id": "ci_vid:test:000000012", "cache_file": "latents/case.safetensors",
            "latent_shot_groups": GROUPS, "provenance": provenance()}


def test_written_library_round_trips_entries_and_rejects_another_source(tmp_path):
    library = rfl.ReferenceLibrary([
        rfl.LibraryFrame(0, "first", None, torch.zeros(1, 2, 1, 3, 4)),
        rfl.LibraryFrame(0, "last", None, torch.ones(1, 2, 1, 3, 4)),
        rfl.LibraryFrame(1, "middle", None, torch.full((1, 2, 1, 3, 4), 2.0)),
    ])
    path = rfl.write_case(tmp_path, _record(), library, clip_model="clip-dir")
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
        latents = handle.get_tensor(rfl.TENSOR_KEY)
    assert latents[0, 0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert "source_cache" not in metadata  # 来源由 provenance 认定，文件头不带生产时的目录路径
    assert rfl.validate_reference_source(metadata, _record()) == [(0, "first"), (0, "last"), (1, "middle")]
    other = _record()
    other["provenance"] = dict(other["provenance"], cache_version="another-cache-version")
    with pytest.raises(ValueError, match="differs"):
        rfl.validate_reference_source(metadata, other)
    with pytest.raises(ValueError, match="schema"):  # v3 的库是随机裁剪过的，不许混进来
        rfl.validate_reference_source(dict(metadata, schema="reference_frame_latents_v3"), _record())
    shuffled = dict(metadata, entries=json.dumps([[0, "last"], [0, "first"], [1, "middle"]]))
    with pytest.raises(ValueError, match="ordered"):
        rfl.validate_reference_source(shuffled, _record())
