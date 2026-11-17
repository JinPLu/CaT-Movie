"""Tiny provenance contract for pre-existing mathematical tensor/layout tests."""


def provenance():
    return {"schema": "wan22_cache_provenance_v1", "cache_version": "cache",
        "encoding": {"dtype": "bfloat16", "pixel_conversion": "uint8_to_bfloat16_mul_2_over_255_add_minus1",
            "implementation_revision": "test-revision", "tiled": False, "tile_size": [30,52], "tile_stride": [15,26]},
        "models": {k: {"path": k, "revision": "test-revision"} for k in ("vae", "text_encoder", "tokenizer")},
        "media": {"schema": "test", "files": [{"path": "source.mp4", "bytes": 1, "mtime_ns": 1,
            "source_num_frames": 10, "source_width": 32, "source_height": 32, "source_fps": 15}],
            "source_shot_segments": [[0, 5], [5, 10]],
            "sampling": {"target_fps": 15, "shot_groups": [[0,5],[5,10]],
                "latent_shot_groups": [[0,2],[2,4]], "frame_indices": list(range(10))},
            "preprocessing": {"decoder": "test", "decoder_version": "test", "torchvision_version": "test",
                "pixel_conversion": "test", "spatial": [{"source_to_target_matrix": [[1,0,0],[0,1,0],[0,0,1]]}]}},
        "context": {"prompts": ["first", "second"], "max_tokens": 512, "prompt_template_revision": "test"}}


def contract():
    p = provenance()
    return {"cache_version": p["cache_version"], "models": p["models"], "prompt_template_revision": "test", "implementation_revision": "test-revision", "encoding": p["encoding"]}


def metadata(record):
    import json
    return {key: (str(record[key]) if key in {"case_id", "height", "width", "num_frames"} else json.dumps(record[key]))
        for key in ("provenance", "case_id", "height", "width", "num_frames", "input_latents_shape",
                    "context_shape", "shot_groups", "latent_shot_groups") if key in record}
