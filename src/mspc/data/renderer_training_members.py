"""渲染器训练目标窗口表的唯一生产者：按镜数分层的完整源窗，或显式重放旧时长档抽样。"""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from mspc.core.draws import unit_draw
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA, require_training_media
from mspc.data.shot_targets import member_target_frame_counts, shot_key
from mspc.training.block_dataset import BlockRow, BlockWindowSampler, shows_action_or_change

def norm(text):
    return " ".join(str(text).lower().split()) if text else None


def outfit_states(annotation, position, name):
    shot = annotation["shots"][position]
    entry = next((p for p in shot.get("present", []) if p.get("name") == name), None)
    if entry is None:
        return None
    default = next((c.get("outfit") for c in annotation["characters"] if c["name"] == name), None)
    opening = norm(entry.get("outfit") or default)
    ending, changing = opening, False
    for change in shot.get("changes") or []:
        if change.get("name") == name and change.get("what") == "outfit":
            ending, changing = norm(change.get("after")), True
    return opening, ending, changing


VERB = re.compile(r"\b(?:puts? on|putting on|takes? off|taking off|removes?|removing|dons?|donning|"
                  r"undress(?:es|ing)?|gets? dressed|getting dressed|changes? (?:his |her |their )?(?:clothes|outfit)|"
                  r"changes? into|unbuttons?|unbuttoning|unzips?|unzipping|zips? up|buttons? up|pulls? (?:off|on)|"
                  r"pulling (?:off|on)|slips? (?:on|off|into)|slipping (?:on|off|into)|strips?|stripping)\b", re.I)
NOUN = re.compile(r"\b(?:t[ -]?shirts?|shirts?|blouses?|jackets?|coats?|overcoats?|raincoats?|robes?|dress(?:es)?|trousers|pants|"
                  r"skirts?|sweaters?|sweatshirts?|vests?|hoodies?|uniforms?|suits?|jumpsuits?|shorts|tunics?|kimonos?|aprons?|"
                  r"cardigans?|pajamas?|clothes|clothing|costumes?|outfits?|gowns?|tuxedos?|scarf|scarves|caps?|hats?|helmets?|"
                  r"headbands?|hoods?|gloves?|shoes?|sneakers?|boots?|socks?|necklaces?|earrings?|bracelets?|sunglasses|glasses|"
                  r"goggles|masks?|ties?|bow ?ties?|belts?|crowns?|wigs?|veils?|jewelry|ring)\b", re.I)


def worn_item_changers(shot):
    """动作句含穿脱动词＋衣物名词时，归给本镜有变化记录或动作句点名的角色；只有一个在场角色时归给他。"""
    action = shot.get("action") or ""
    if not (VERB.search(action) and NOUN.search(action)):
        return set()
    people = [p["name"] for p in shot.get("present", []) if p["name"].startswith("Character")]
    named = {c["name"] for c in shot.get("changes") or [] if c.get("name", "").startswith("Character")}
    named |= {n for n in people if n in action}
    return named or (set(people) if len(people) == 1 else set())


def legal_spans(frames, usable, max_shots=None, *, max_frames):
    """块内全部合法窗口 (start, 镜数)：连续可用镜、总帧数不超过显式上限，镜数只在 ``max_shots`` 给出时封顶。"""
    usable = set(usable)
    for start in sorted(usable):
        stop, total = start, 0
        while (stop in usable and (max_shots is None or stop - start < max_shots)
               and total + frames[stop] <= max_frames):
            total += frames[stop]
            stop += 1
        for count in range(1, stop - start + 1):
            yield start, count


def candidates(member, usable, max_shots=None, *, frame_counts, max_frames):
    annotation = member['annotation']
    changes = defaultdict(set)
    for position, shot in enumerate(annotation['shots']):
        for change in shot.get('changes') or []:
            if change.get('what') == 'outfit':
                changes[change['name']].add(position)
        for name in sorted(worn_item_changers(shot)):
            changes[name].add(position)
    result = {}
    for start, count in legal_spans(frame_counts, usable, max_shots, max_frames=max_frames):
        span = range(start, start + count)
        refs = defaultdict(list)
        # 人物按名字、参考按镜号排：集合按字符串哈希遍历，每个进程顺序不同，冻结表就不能跨进程逐字节复现。
        for name, positions in sorted(changes.items()):
            targets = [t for t in span if outfit_states(annotation, t, name) is not None]
            for r in usable:
                if r in span or r in positions or outfit_states(annotation, r, name) is None:
                    continue
                events = sorted({c for c in positions for t in targets
                                 if min(r, t) < c <= max(r, t) or (c == t and r < t)})
                if events:
                    refs[r].append({'character': name, 'change_positions': events})
        if refs:
            result[(start, count)] = dict(sorted(refs.items()))
    return result



def read_rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def write_rows(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_exposure(*, members, sampling, media, limit=None):
    """汇总冻结采样实际目标；同一块同一半开区间才是重复段。"""
    media = require_training_media(media)
    fps = media["target_fps"]
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    segments = Counter()
    for index, sample in enumerate(read_rows(sampling)):
        if limit is not None and index >= limit:
            break
        if sample["sample_index"] != index:
            raise ValueError(f"sample {index}: index differs")
        start, end = sample["span"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError(f"sample {index}: invalid span")
        segments[(sample["case_id"], start, end)] += 1
    if not segments:
        raise ValueError("no samples")
    needed = {case for case, _, _ in segments}
    selected, target_frames = {}, {}
    for member in read_rows(members):
        case = member["case_id"]
        if case in needed:
            if case in selected:
                raise ValueError(f"duplicate member: {case}")
            selected[case] = member["shots"]
            target_frames[case] = member_target_frame_counts(member, media=media)
    if needed - selected.keys():
        raise ValueError(f"{len(needed - selected.keys())} sampled members missing")
    shots = set()
    shot_count, frame_count, source_seconds = 0, 0, 0.0
    histogram = Counter()
    for (case, start, end), count in segments.items():
        selected_shots = selected[case][start:end]
        if len(selected_shots) != end - start or any(s["usable"] is not True for s in selected_shots):
            raise ValueError(f"invalid sampled shots: {case} {start}:{end}")
        for shot in selected_shots:
            if type(shot["frames"]) is not int or shot["frames"] <= 0:
                raise ValueError(f"invalid frame count: {case}")
            if not math.isfinite(shot["seconds"]) or shot["seconds"] <= 0:
                raise ValueError(f"invalid source seconds: {case}")
        histogram[end - start] += count
        shot_count += (end - start) * count
        frame_count += sum(target_frames[case][start:end]) * count
        source_seconds += sum(s["seconds"] for s in selected_shots) * count
        shots.update((case, index) for index in range(start, end))
    draws = sum(segments.values())
    return {"members": str(members), "sampling": str(sampling), "limit": limit, "fps": fps, "media": dict(media),
            "sampled_segments": draws, "unique_case_span_segments": len(segments),
            "repeated_segment_draws": draws - len(segments), "unique_blocks": len(needed),
            "shot_count_histogram": dict(sorted(histogram.items())),
            "target_shot_presentations": shot_count, "unique_target_shots": len(shots),
            "target_frame_presentations": frame_count,
            "target_grid_hours": frame_count / fps / 3600,
            "source_shot_hours_with_repetition": source_seconds / 3600,
            "segment_identity": "case_id and zero-based half-open span",
            "duration_scope": "target shots only; grid hours use frames/fps; source hours sum shot seconds; both count repeated draws"}


def conflict_options(found, frames, sampler):
    """各时长档里带衣着冲突候选参考的窗口：{档: {case: [(start, end), ...]}}；块按 case_id 排序，窗口按合法窗口顺序。"""
    options = {bucket: {} for bucket in range(len(sampler.frame_buckets))}
    for case in sorted(found):
        for start, count in found[case]:
            bucket = sampler.bucket(sum(frames[case][start:start + count]))
            options[bucket].setdefault(case, []).append((start, start + count))
    return options


def forced_span(options, exposure, bucket, *, seed, index):
    """定向样本的目标窗口：在定向曝光不足两次的候选块里均匀选块，再在块内候选窗口里均匀选窗口。"""
    blocks = [case for case in options[bucket] if exposure[case] < 2]
    if not blocks:
        raise ValueError(f"no outfit-conflict block left in bucket {bucket} for sample {index}")
    case = blocks[int(unit_draw("outfit_forced_block", seed, index) * len(blocks))]
    spans = options[bucket][case]
    return case, spans[int(unit_draw("outfit_forced_span", seed, index) * len(spans))]


def _draw_duration_bucket_windows(members, *, media, seed, frame_buckets, bucket_weights, max_shots,
                                 sample_budget, forced_samples):
    """从训练成员冻一份目标窗口表。

    自然样本是 ``BlockWindowSampler(seed).draw(index)``（先按权重选时长档，再在档内均匀取窗口）；衣着冲突定向份额
    ``forced_samples`` 按档权重分到各档，档内按 ``outfit_forced_position`` 选位置，换成 ``forced_span`` 抽出的同档候选
    窗口。参考不进表，训练时由 ``BlockDataset`` 按（种子，样本下标）现抽。``max_shots`` 只给逐镜对照取 1。
    """
    media = require_training_media(media)
    info = {m["case_id"]: m for m in members}
    if len(info) != len(members):
        raise ValueError("duplicate member case_id")
    rows = [BlockRow(m["case_id"], member_target_frame_counts(m, media=media),
                     tuple(i for i, s in enumerate(m["shots"]) if s["usable"])) for m in members]
    sampler = BlockWindowSampler(rows, frame_buckets=frame_buckets, bucket_weights=bucket_weights,
                                 max_shots=max_shots, seed=seed, shot_count_weights=None)
    frames = {row.case_id: row.frame_counts for row in rows}
    draws = [sampler.draw(index) for index in range(sample_budget)]
    found = {row.case_id: candidates(info[row.case_id], row.usable, max_shots,
                                         frame_counts=row.frame_counts, max_frames=frame_buckets[-1]) for row in rows}
    options = conflict_options(found, frames, sampler)
    total = sum(bucket_weights)
    quotas = [round(forced_samples * weight / total) for weight in bucket_weights]
    if sum(quotas) != forced_samples:
        raise ValueError(f"forced samples {forced_samples} do not split exactly over {list(bucket_weights)}")
    positions = defaultdict(list)
    for index, (row, span, _) in enumerate(draws):
        positions[sampler.bucket(sum(row.frame_counts[span[0]:span[1]]))].append(index)
    forced = {}
    for bucket, need in enumerate(quotas):
        if need > len(positions[bucket]):
            raise ValueError(f"insufficient samples in bucket {bucket} for {need} forced samples")
        for index in sorted(positions[bucket], key=lambda i: unit_draw("outfit_forced_position", seed, i))[:need]:
            forced[index] = bucket
    exposure, samples = Counter(), []
    for index, (row, span, _) in enumerate(draws):
        case = row.case_id
        if index in forced:
            case, span = forced_span(options, exposure, forced[index], seed=seed, index=index)
            exposure[case] += 1
        member = info[case]
        samples.append({"sample_index": index, "branch": "forced" if index in forced else "natural",
                        "key": member["key"], "film_key": member["film_key"], "source": member["source"],
                        "case_id": case, "span": list(span),
                        "candidate_reference_positions": found[case].get((span[0], span[1] - span[0]), {})})
    report = {"legal_windows_by_bucket": [len(group) for group in sampler.windows],
              "conflict_blocks_by_bucket": {bucket: len(blocks) for bucket, blocks in options.items()},
              "forced_bucket_quotas": quotas,
              "forced_block_exposure_max": max(exposure.values(), default=0)}
    return samples, report


SAMPLING_SCHEMES = ("shot_count_weighted_windows", "duration_bucket_windows")


def _draw_shot_count_weighted_windows(members, *, media, seed, sample_budget,
                                     partition_max_rgb_frames, max_source_shot_seconds, shot_count_weights,
                                     long_single_shot):
    """共同源16fps网格上按1／2／3／4+镜抽层；候选集与缓存是否存在无关。

    ``long_single_shot=(份额, 最少源帧)``：单镜层内按份额只取长单镜（源16fps帧数不少于最少源帧，且块标注里该镜
    有动作或镜内变化，见 :func:`mspc.training.block_dataset.shows_action_or_change`）；此时成员要带
    ``annotation.shots``（至少 ``activity`` 与 ``changes``）。"""
    if type(partition_max_rgb_frames) is not int or partition_max_rgb_frames <= 0:
        raise ValueError("shot-count windows require a positive partition_max_rgb_frames")
    if type(sample_budget) is not int or sample_budget <= 0:
        raise ValueError("sample_budget must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (type(max_source_shot_seconds) not in (int, float)
            or not math.isfinite(max_source_shot_seconds) or max_source_shot_seconds <= 0):
        raise ValueError("max_source_shot_seconds must be positive and finite")
    if shot_count_weights is None:
        raise ValueError("shot_count_weights must explicitly give the four source shot-count layers")
    info, rows, target_counts = {}, [], {}
    dropped_shots = 0
    action_shots = set()
    for member in members:
        case = member["case_id"]
        if not case or case in info:
            raise ValueError(f"duplicate or empty member case_id: {case!r}")
        info[case] = member
        source_counts = member_target_frame_counts(member, media=ANNOTATION_MEDIA)
        target_counts[case] = member_target_frame_counts(member, media=media)
        usable = []
        for position, shot in enumerate(member["shots"]):
            if type(shot["usable"]) is not bool:
                raise ValueError(f"{case}: usable must be bool")
            if (type(shot["seconds"]) not in (int, float)
                    or not math.isfinite(shot["seconds"]) or shot["seconds"] <= 0):
                raise ValueError(f"{case} shot {position}: source seconds must be positive and finite")
            if shot["usable"]:
                if source_counts[position] != shot["frames"]:
                    raise ValueError(f"{case} shot {position}: annotation frames differ from source sampling media")
                if shot["seconds"] > max_source_shot_seconds:
                    dropped_shots += 1
                else:
                    usable.append(position)
        if long_single_shot is not None:
            annotated = member["annotation"]["shots"]
            if len(annotated) != len(member["shots"]):
                raise ValueError(f"{case}: annotation and member shot counts differ")
            # 只看可用镜（长单镜候选只来自可用镜）；不可用镜的标注可能不带 changes（11-13 真实成员 3/12,153 镜）。
            action_shots.update((case, position) for position in usable
                                if shows_action_or_change(annotated[position]))
        rows.append(BlockRow(case, source_counts, tuple(usable)))
    sampler = BlockWindowSampler(rows, frame_buckets=[partition_max_rgb_frames], bucket_weights=None,
        max_shots=None, seed=seed, shot_count_weights=shot_count_weights,
        long_single_shot=None if long_single_shot is None else (*long_single_shot, frozenset(action_shots)))
    max_source_rgb = max_target_rgb = max_target_latents = long_singletons = 0
    for group in sampler.windows:
        for row, start, stop in group:
            source_rgb = sum(row.frame_counts[start:stop])
            target_rgb = sum(target_counts[row.case_id][start:stop])
            if target_rgb > media["max_rgb_frames"]:
                raise ValueError(
                    f"{row.case_id} source span {start}:{stop}: complete shared window maps to {target_rgb} "
                    f"target RGB frames, exceeding explicit media capacity {media['max_rgb_frames']}; "
                    "do not split, shorten or omit the shared source window")
            max_source_rgb = max(max_source_rgb, source_rgb)
            max_target_rgb = max(max_target_rgb, target_rgb)
            max_target_latents = max(max_target_latents,
                sum((count + 3) // 4 for count in target_counts[row.case_id][start:stop]))
            long_singletons += int(source_rgb > partition_max_rgb_frames)
    samples = []
    long_pool = (frozenset() if long_single_shot is None else
                 frozenset((row.case_id, start) for row, start, _ in sampler.long_single_pool))
    long_samples = 0
    for index in range(sample_budget):
        row, span, _ = sampler.draw(index)
        long_samples += span[1] - span[0] == 1 and (row.case_id, span[0]) in long_pool
        member = info[row.case_id]
        samples.append({"sample_index": index, "branch": "shot_count_weighted_windows",
                        "key": member["key"], "film_key": member["film_key"], "source": member["source"],
                        "case_id": row.case_id, "span": list(span), "candidate_reference_positions": {}})
    return samples, {"source_sampling_media": dict(ANNOTATION_MEDIA),
                     "source_window_max_rgb_frames": partition_max_rgb_frames,
                     "max_source_shot_seconds": max_source_shot_seconds,
                     "source_duration_dropped_usable_shots": dropped_shots,
                     "shot_count_layers": ["1", "2", "3", "4+"],
                     "shot_count_weights": list(shot_count_weights),
                     "long_single_shot": None if long_single_shot is None else {
                         "share": long_single_shot[0], "min_source_frames": long_single_shot[1],
                         "definition": "single-shot window, source frames >= min_source_frames, and annotation "
                                       "activity == action_or_state_change or nonempty changes",
                         "action_or_change_shots": len(action_shots),
                         "legal_windows": len(sampler.long_single_pool),
                         "samples": long_samples, "sample_fraction": long_samples / sample_budget},
                     "legal_windows_by_shot_count_layer": [len(group) for group in sampler.windows],
                     "source_window_long_singletons": long_singletons,
                     "source_window_max_rgb_frames_observed": max_source_rgb,
                     "source_window_target_rgb_frames_max": max_target_rgb,
                     "source_window_target_latents_max": max_target_latents,
                     "sampling_rule": "draw shot-count layer by explicit weights, then uniformly sample every legal continuous complete source window in that layer, with replacement"}


def draw_sampling(members, *, media, seed, sampling_scheme, frame_buckets, bucket_weights,
                  max_shots, sample_budget, forced_samples, partition_max_rgb_frames, max_source_shot_seconds,
                  shot_count_weights, long_single_shot=None):
    """按显式镜数权重冻结源窗；历史时长档只用于已命名的复现。"""
    media = require_training_media(media)
    if sampling_scheme == "shot_count_weighted_windows":
        if any(value is not None for value in (frame_buckets, bucket_weights, max_shots, forced_samples)):
            raise ValueError("shot-count windows do not use duration buckets, shot limits or forced samples")
        return _draw_shot_count_weighted_windows(members, media=media, seed=seed,
            sample_budget=sample_budget, partition_max_rgb_frames=partition_max_rgb_frames,
            max_source_shot_seconds=max_source_shot_seconds, shot_count_weights=shot_count_weights,
            long_single_shot=long_single_shot)
    if sampling_scheme == "duration_bucket_windows":
        if long_single_shot is not None:
            raise ValueError("duration bucket replay does not take a long single-shot share")
        if max_source_shot_seconds is not None or shot_count_weights is not None:
            raise ValueError("duration bucket replay does not accept source duration filtering or shot-count weights")
        if partition_max_rgb_frames is not None or any(value is None for value in (frame_buckets, bucket_weights, forced_samples)):
            raise ValueError("duration bucket replay requires explicit buckets/weights/forced_samples and no partition capacity")
        return _draw_duration_bucket_windows(members, media=media, seed=seed, frame_buckets=frame_buckets,
            bucket_weights=bucket_weights, max_shots=max_shots, sample_budget=sample_budget,
            forced_samples=forced_samples)
    raise ValueError(f"sampling_scheme must be one of {SAMPLING_SCHEMES}")


def _shot_count_member(member, *, with_activity):
    """镜数分层只读的成员字段；长单镜份额另带可用镜的 ``activity`` 与 ``changes``（不可用镜记 None）。"""
    kept = {key: member[key] for key in ("case_id", "key", "film_key", "source", "scene", "shots")}
    if with_activity:
        kept["annotation"] = {"shots": [{name: shot[name] for name in ("activity", "changes")} if member_shot["usable"]
                                        else None
                                        for shot, member_shot in zip(member["annotation"]["shots"], member["shots"],
                                                                     strict=True)]}
    return kept


def write_drawn_sampling(*, members_path, media, seed, sampling_scheme, frame_buckets, bucket_weights, max_shots,
                         sample_budget, forced_samples, partition_max_rgb_frames, max_source_shot_seconds, shot_count_weights, output, limit,
                         long_single_shot=None):
    """写 ``draw_sampling`` 的表；``limit`` 只写前若干行，是同一份全量表的前缀。"""
    if limit is not None and not 0 < limit <= sample_budget:
        raise ValueError("limit must be in 1..sample_budget")
    media = require_training_media(media)
    # 新方案只在长单镜份额时读每镜 activity／changes，无需把1GB成员中的整块标注常驻两套对象。
    members = ([_shot_count_member(member, with_activity=long_single_shot is not None)
                for member in read_rows(members_path)] if sampling_scheme == "shot_count_weighted_windows"
               else list(read_rows(members_path)))
    samples, report = draw_sampling(members, media=media, seed=seed, sampling_scheme=sampling_scheme,
        frame_buckets=frame_buckets, bucket_weights=bucket_weights, max_shots=max_shots,
        sample_budget=sample_budget, forced_samples=forced_samples, partition_max_rgb_frames=partition_max_rgb_frames,
        max_source_shot_seconds=max_source_shot_seconds, shot_count_weights=shot_count_weights,
        long_single_shot=long_single_shot)
    written = samples[:limit] if limit is not None else samples
    with Path(output).open("x", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    frames = {m["case_id"]: member_target_frame_counts(m, media=media) for m in members}
    usable = {m["case_id"]: [i for i, s in enumerate(m["shots"]) if s["usable"]] for m in members}
    window_frames = [sum(frames[s["case_id"]][s["span"][0]:s["span"][1]]) for s in written]
    buckets = (Counter(next(i for i, bound in enumerate(frame_buckets) if n <= bound) for n in window_frames)
               if sampling_scheme == "duration_bucket_windows" else None)
    by_case = {member["case_id"]: member for member in members}
    layer_counts = Counter(min(s["span"][1] - s["span"][0], 4) for s in written)
    unique_source_target_shots = (len({shot_key(by_case[sample["case_id"]]["shots"][position])
        for sample in written for position in range(*sample["span"])})
        if sampling_scheme == "shot_count_weighted_windows" else None)
    summary = {"members": str(members_path), "output": str(output), "seed": seed, "media": dict(media),
               "sampling_scheme": sampling_scheme,
               "shot_count_layer_counts": dict(sorted(layer_counts.items())),
               "shot_count_layer_fractions": {layer: layer_counts[layer] / len(written) for layer in range(1, 5)},
               "long_single_shot_fraction_at_or_above_80_source_frames": sum(
                   1 for s in written if s["span"][1] - s["span"][0] == 1
                   and by_case[s["case_id"]]["shots"][s["span"][0]]["frames"] >= 80) / len(written),
               "unique_source_target_shots": unique_source_target_shots,
               "frame_buckets": list(frame_buckets) if frame_buckets is not None else None,
               "bucket_weights": list(bucket_weights) if bucket_weights is not None else None, "max_shots": max_shots,
               "sample_budget": sample_budget, "forced_samples": forced_samples, "rows_written": len(written),
               "bucket_counts": dict(sorted(buckets.items())) if buckets is not None else None,
               "target_rgb_frames_max_written": max(window_frames),
               "target_latents_max_written": max(sum((count + 3) // 4
                   for count in frames[s["case_id"]][s["span"][0]:s["span"][1]]) for s in written),
               "source_rgb_frames_max_written": max(sum(by_case[s["case_id"]]["shots"][p]["frames"]
                   for p in range(*s["span"])) for s in written),
               "reference_candidate_policy": "all_usable_shots_outside_target_span",
               "shot_counts": dict(sorted(Counter(s["span"][1] - s["span"][0] for s in written).items())),
               # 训练参考只取同块窗口外的可用镜；窗口拉长后这个数会变小，冻表时一并记下。
               "reference_candidate_shots": dict(sorted(Counter(
                   sum(1 for i in usable[s["case_id"]] if not s["span"][0] <= i < s["span"][1])
                   for s in written).items())),
               **report, "exposure": summarize_exposure(members=members_path, sampling=output, media=media)}
    Path(output).with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
