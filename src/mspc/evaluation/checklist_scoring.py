"""第13版计分复用第11版镜对／整片判官回执，支持状态建立／回访子项和非相邻同机位回归门控。

readout-v11／readout-v12保留历史门控。readout-v13 --answer-scope existing 显式报告缺测，只聚合可判题目；readout-v14 同第13版口径，只是物体题按第14版题库收窄为两镜都要求的可移动道具。
各入口均核题库与切镜项版本；不发请求。以下第9版说明及readout命令供历史复现。

第 9 版＝第 8 版去掉三类计分题：难反例退回只作监测，事件题去掉静态动作与故事内重复，
去掉场景类物体的物体题。去哪些题由 ``question_bank_v6 scored-v9`` 写出的计分题库给定（``--scored-bank``），判官提示与
回执不变。

切镜执行项由 `question_writer.cut_items`
出、`cut_execution` 判（TransNetV2、帧相似度、判官看四帧）。只在 CPU 上重算，不发请求。

- 每道计分题有期望答案，通过＝判官答案等于期望 yes（难反例 `monitor` 期望 no，第 9 版起不计分）。
- 一致性题（人物、物体、地点）与主表状态复查题：所在镜对有切镜执行项时，还要该项判成新机位（相邻项另要
  检出切点）；不成立则这道题不通过，记为被前提作废。依赖只有"同一镜对的前提 → 依赖它的题"一种，不跨镜对传导。
- 事件、变化那一镜的镜末状态只看自己的答案。
- 人物、物体、地点跨镜题的答案来源由 ``--consistency-judge`` 明给：``shot_pair`` 读 ``shot_pair_consistency`` 的一步判答案
  （每对镜头一次请求看原分辨率帧；作者盲标留出 0.78，整片判官 0.70），``whole_video`` 用第 6 版整片判官原答。
- 切镜执行列：每个计划要求前后两镜不同的计划切点，判成新机位才通过（没检出切点判 no_cut，不通过）。
- 六列各取故事内通过比例，故事等权；某列无题的故事不进该列均值。全过＝该故事全部计分题都通过，含切镜项。
- 每行另报：被前提作废的通过数；只在前提成立（或无前提）的题上算的一致性列及其覆盖率；前提不成立的项数；
  第 6 版计分（`question_writer.story_scores`，附录用）。第 7 版（前提为判官看整段答"不是同一个镜头"）的读数冻结在
  第 7 版运行根 `readouts/`，计算代码见 git 历史。

检出切点写成一份切点表（行 → 故事 → 检出切点秒数与片长），两个来源：非 VLM 运行已存的整片检出（`stored-cuts`，
测试集主表各行），或在主服务器 CPU 上对判分行成片整片跑 TransNetV2（`detect-cuts`；11-04 CPU 重跑 5 部与 A100 已存检出
逐个相同）。

用法::

    PYTHONPATH=src python -m mspc.evaluation.checklist_scoring stored-cuts --v6-root V6_ROOT \\
        --metrics-root NON_VLM/metrics --pair ROW METHOD [--pair ...] --output CUTS.json
    PYTHONPATH=src python -m mspc.evaluation.checklist_scoring detect-cuts --v6-root ROOT_WITH_ROWS --rows ROW ... \\
        --ssh 'ssh -p PORT root@HOST' --python TF_PYTHON --repo REMOTE_CODE --resource-root TRANSNET_ROOT --output CUTS.json
    PYTHONPATH=src python -m mspc.evaluation.checklist_scoring readout --v6-root V6_ROOT --v8-root CUT_EXECUTION_ROOT \\
        --items cut-items.json --scored-bank questions.scored.v9.json --rows ROW ... --consistency-judge shot_pair --shot-pair-root SHOT_PAIR_ROOT --output O.json
    # 配对区间（右减左，故事等权，按源片重采样）
    PYTHONPATH=src python -m mspc.evaluation.checklist_scoring paired --readout O.json [...] \\
        --requests configs/benchmark/story_requests_v4.json --pair LEFT RIGHT [--pair ...] --output P.json

第11／12／13版按镜长读数时，readout-v11／readout-v12／readout-v13另给 ``--storyboards``（实际计划清单，含每故事fps及逐镜frame_count）；
paired另给 ``--event-duration-groups``。一侧三行仍写成逗号分隔，先在故事内平均种子，不把种子当独立故事。
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import accumulate
from pathlib import Path
from statistics import mean

from mspc.evaluation.question_writer import story_scores as v6_story_scores
from mspc.evaluation.video_metrics import tolerance_pairs

WINDOW_SECONDS = 0.5
COLUMNS = ('character', 'object', 'scene', 'event', 'state', 'cut')
CONSISTENCY_COLUMNS = ('character', 'object', 'scene')
EVENT_DURATION_GROUPS = ('at_most_5_seconds', 'over_5_seconds')
STATE_ROLES = ('establishment', 'revisit')
# 挂前提的题所在的列：一致性三列与状态（第 8 版起的状态复查题）。
GATED_COLUMNS = ('character', 'object', 'scene', 'state')
_DIMENSION_COLUMN = {'character': 'character', 'object': 'object', 'place': 'scene', 'event': 'event', 'state': 'state'}


def expected_answer(row: dict) -> str:
    return 'no' if row['dimension'] == 'monitor' else row.get('expected_answer', 'yes')


def column_of(row: dict) -> str | None:
    """计分列；画风题、难反例（只作监测）、旧题库布局题与前提题本身返回 None（前提题经切镜列计分）。"""
    return _DIMENSION_COLUMN.get(row['dimension'])


def planned_cut_hits(planned: list[float], detected: list[float]) -> set[int]:
    """在 ±0.5 秒内配上检出切点的计划切点下标；两边一一配对，时间差小的先配。"""
    return {i for i, _ in tolerance_pairs(planned, detected, WINDOW_SECONDS)}


def story_scores(rows: list[dict], premises: list[dict], answers: dict, holds: dict[str, bool | None],
                 cut_passed: dict[str, bool | None], *, answer_scope: str = 'complete') -> dict:
    """一个故事的六列分。``rows`` 是第 6 版题，``answers`` 是它们的判官答案（requirement_id → 'yes'／'no'）；
    ``premises`` 是该故事的前提项（``id``、``planned_cut``、``dependents``），``holds`` 是每项前提是否成立，
    ``cut_passed`` 是每个相邻项（计划切点）的切镜执行是否通过，由 ``v8_premises`` 从新机位判定得出。
    ``existing``范围将缺答／未知前提逐题列为缺测并排除，前提已失败的题照常计不过；有缺测时全过为None。"""
    if answer_scope not in ('complete', 'existing'):
        raise ValueError(f'unknown answer scope: {answer_scope}')
    dependent_of = {rid: p['id'] for p in premises for rid in p['dependents']}
    passed, holding, questions = {}, {}, {}
    totals = dict.fromkeys(COLUMNS, 0)

    def missing(rid: str, column: str, reason: str) -> None:
        if answer_scope == 'complete':
            raise ValueError(f'{rid}: {reason}')
        questions[rid] = {'column': column, 'status': 'missing', 'reason': reason, 'passed': None}

    voided = dict.fromkeys(GATED_COLUMNS, 0)
    for row in rows:
        column = column_of(row)
        if column is None:
            continue
        rid = row['requirement_id']
        totals[column] += 1
        if row.get('requires_fresh_answer'):
            missing(rid, column, 'question contract changed; existing judge answer is not reusable')
            continue
        premise = dependent_of.get(rid)
        if premise is not None and holds[premise] is None:
            missing(rid, column, 'no decided cut premise')
            continue
        if premise is not None and holds[premise] is False:
            # 前提不成立记不过；第 11 版起这类题不送判官，有答案时才计被作废的通过数。
            if rid in answers:
                voided[column] += answers[rid] == expected_answer(row)
            passed[rid] = (column, False)
            questions[rid] = {'column': column, 'status': 'gated', 'passed': False,
                              'has_judge_answer': rid in answers}
            continue
        if rid not in answers:
            missing(rid, column, 'no judge answer although its premise holds')
            continue
        ok = answers[rid] == expected_answer(row)
        passed[rid] = (column, ok)
        questions[rid] = {'column': column, 'status': 'answered', 'passed': ok, 'has_judge_answer': True}
        if column in CONSISTENCY_COLUMNS:
            holding[rid] = (column, ok)
    unknown = set(dependent_of) - {row['requirement_id'] for row in rows}
    if unknown:
        raise ValueError(f'premises name questions missing from the judged bank: {sorted(unknown)[:3]}')
    for p in premises:
        if p['planned_cut']:
            totals['cut'] += 1
            if cut_passed[p['id']] is None:
                missing(p['id'], 'cut', 'no decided cut verdict')
                continue
            passed[p['id']] = ('cut', cut_passed[p['id']])
            questions[p['id']] = {'column': 'cut', 'status': 'answered', 'passed': cut_passed[p['id']]}

    def rate(table: dict, column: str) -> float | None:
        values = [ok for c, ok in table.values() if c == column]
        return sum(values) / len(values) if values else None

    consistency_questions = {c: sum(1 for k, _ in passed.values() if k == c) for c in CONSISTENCY_COLUMNS}
    auxiliary = {}
    for role in STATE_ROLES:
        selected = [r for r in rows if r['dimension'] == 'state' and r.get('state_role') == role]
        # 第16版起状态建立要有变化：镜末成立且同镜做出变化的事件题也通过（change_event）；事件题缺测则这道建立题缺测。
        values = [passed[r['requirement_id']][1] and passed[r['change_event']][1] if r.get('change_event') else passed[r['requirement_id']][1]
                  for r in selected if r['requirement_id'] in passed and (not r.get('change_event') or r['change_event'] in passed)]
        auxiliary[f'state_{role}'] = {'score': sum(values) / len(values) if values else None,
                                    'questions': len(values), 'of': len(selected),
                                    'missing': len(selected) - len(values)}
    measurement_missing = [rid for rid, value in questions.items() if value['status'] == 'missing']
    return {'scores': {column: rate(passed, column) for column in COLUMNS},
            'all_pass': None if measurement_missing else all(ok for _, ok in passed.values()),
            'auxiliary': auxiliary,
            'question_coverage': {c: {'questions': sum(k == c for k, _ in passed.values()), 'of': totals[c],
                                       'missing': totals[c] - sum(k == c for k, _ in passed.values())} for c in COLUMNS},
            'questions': questions, 'measurement_missing': measurement_missing,
            'voided': voided,
            'premise_holding': {c: {'score': rate(holding, c), 'questions': sum(1 for k, _ in holding.values() if k == c),
                                    'of': consistency_questions[c]} for c in CONSISTENCY_COLUMNS},
            'premises': len(premises), 'premises_failed': sum(v is False for v in holds.values()),
            'premises_missing': sum(v is None for v in holds.values())}


def v8_premises(items: list[dict], verdicts: dict, *, scoring_version: str = 'v11',
                answer_scope: str = 'complete') -> tuple:
    """历史版本要求本镜对为new；v13的非相邻、计划同机位镜对还可由corr_mid<0.90且区间至少两个相邻new切点成立。
    计划要求不同的镜对始终要求自身new；计划切镜列只计相邻项，不由回归分支改写。未知判定保持缺测。"""
    premises = [{'id': i['item_id'], 'planned_cut': i['planned_cut'], 'dependents': i['dependents']} for i in items]
    if scoring_version not in ('v9', 'v11', 'v12', 'v13'):
        raise ValueError(f'unknown scoring version: {scoring_version}')
    if answer_scope not in ('complete', 'existing'):
        raise ValueError(f'unknown answer scope: {answer_scope}')
    undecided = {i['item_id'] for i in items if i['item_id'] not in verdicts or verdicts[i['item_id']]['verdict'] is None}
    holds = {p['id']: None if p['id'] in undecided else verdicts[p['id']]['verdict'] == 'new' for p in premises}
    if scoring_version == 'v13':
        for item in items:
            key, earlier, later = item['item_id'], item['reference_shot'], item['shot']
            if holds[key] or later <= earlier + 1 or item['planned_difference'] is not None:
                continue
            # 原verdict的similarity是max(corr_mid,corr_cut)，不能代替中帧相关系数。
            if key not in verdicts or 'corr_mid' not in verdicts[key]:
                holds[key] = None
                continue
            corr_mid = verdicts[key]['corr_mid']
            if isinstance(corr_mid, bool) or not isinstance(corr_mid, (int, float)) or not math.isfinite(corr_mid):
                raise ValueError(f'{key}: corr_mid must be finite')
            if corr_mid >= 0.90:
                continue
            adjacent = [i['item_id'] for i in items
                        if earlier <= i['reference_shot'] < later and i['shot'] == i['reference_shot'] + 1]
            new = sum(k not in undecided and verdicts[k]['verdict'] == 'new' for k in adjacent)
            unknown = sum(k in undecided for k in adjacent)
            holds[key] = True if new >= 2 else None if key in undecided or new + unknown >= 2 else False
    unresolved = [key for key, value in holds.items() if value is None]
    if unresolved and answer_scope == 'complete':
        raise ValueError(f'undecided cut items: {unresolved[:3]}')
    return premises, holds, {p['id']: holds[p['id']] for p in premises if p['planned_cut']}


def judged_answers(row_dir: Path) -> dict[str, dict[str, str]]:
    """故事 → (requirement_id → 'yes'／'no')，只读有效回执。"""
    out: dict[str, dict[str, str]] = {}
    for path in (row_dir / 'output' / 'metrics' / 'first-read').rglob('*.json'):
        record = json.loads(path.read_text())
        if record['status'] == 'rated':
            out.setdefault(record['story_id'], {}).update(
                {a['requirement_id']: a['answer'] for a in record['result']['answers']})
    return out


def row_readout(v6_row: Path, premises_of, scored: dict[str, set[str]]) -> dict:
    """一行的新版与第 6 版读数。``premises_of(story, answers)`` 给出该故事的前提来源：None 表示还没判完（两版都不计），
    否则是 ``(前提项, 是否成立, 切镜执行是否通过, 判官答案)``。``scored`` 是第 9 版计分题库（故事 → 计分题号）：新版只计
    其中的题，前提的依赖也只留其中的题；第 6 版读数照旧用原题。第 6 版测量失败（无有效回执）的故事两版都不计并单列，不补零。"""
    v6 = json.loads((v6_row / 'prepared.json').read_text())
    unavailable = [s['story_id'] for s in v6['unavailable_stories']]  # 生成失败：不判、不进各列均值，全过记不通过
    old_answers = judged_answers(v6_row)
    stories, pending, failed = [], [], []
    for story in v6['stories']:
        sid = story['story_id']
        if sid not in old_answers:
            failed.append(sid)
            continue
        source = premises_of(story, old_answers[sid])
        if source is None:
            pending.append(sid)
            continue
        premises, holds, cut_passed, answers = source
        keep = scored[sid]
        new = story_scores([r for r in story['requirements'] if r['requirement_id'] in keep],
                           [{**p, 'dependents': [d for d in p['dependents'] if d in keep]} for p in premises],
                           answers, holds, cut_passed)
        old = v6_story_scores(story['requirements'], old_answers[sid])
        old_all = all(old_answers[sid][r['requirement_id']] == 'yes' for r in story['requirements']
                      if r['dimension'] in _DIMENSION_COLUMN)
        stories.append({'story_id': sid, 'v8': new, 'v6': {'scores': old['scores'], 'all_pass': old_all}})

    def column_mean(values: list) -> dict:
        present = [v for v in values if v is not None]
        return {'mean': 100 * mean(present) if present else None, 'n': len(present)}

    new_columns = {c: column_mean([s['v8']['scores'][c] for s in stories]) for c in COLUMNS}
    v6_columns = {c: column_mean([s['v6']['scores'][c] for s in stories]) for c in COLUMNS if c != 'cut'}
    holding = {c: column_mean([s['v8']['premise_holding'][c]['score'] for s in stories]) for c in CONSISTENCY_COLUMNS}
    for c in CONSISTENCY_COLUMNS:
        kept = sum(s['v8']['premise_holding'][c]['questions'] for s in stories)
        total = sum(s['v8']['premise_holding'][c]['of'] for s in stories)
        holding[c]['coverage'] = kept / total if total else None
    return {'stories': len(stories), 'premise_pending': pending, 'v6_measurement_failed': failed,
            'generation_unavailable': unavailable,
            'v8': {'columns': new_columns, 'all_pass': sum(s['v8']['all_pass'] for s in stories)},
            'v6': {'columns': v6_columns, 'all_pass': sum(s['v6']['all_pass'] for s in stories)},
            'change_points': {c: new_columns[c]['mean'] - v6_columns[c]['mean'] for c in v6_columns
                              if new_columns[c]['mean'] is not None and v6_columns[c]['mean'] is not None},
            'voided_passes': {c: sum(s['v8']['voided'][c] for s in stories) for c in GATED_COLUMNS},
            'consistency_where_premise_holds': holding,
            'premises_failed': {'count': sum(s['v8']['premises_failed'] for s in stories),
                                'of': sum(s['v8']['premises'] for s in stories)},
            'per_story': stories}


def v8_source(items: dict, verdicts: dict, shot_pair: dict | None):
    """``shot_pair`` 是 ``shot_pair_consistency`` 写出的一行答案（故事 → 题号 → yes／no），人物／物体／地点跨镜题改读它
    （``--consistency-judge shot_pair``）；为 None 时这些题用整片判官原答（``--consistency-judge whole_video``）。
    跨镜难反例与事件、状态题始终用整片判官原答。"""
    def premises_of(story: dict, old: dict):
        sid = story['story_id']
        answers = old
        if shot_pair is not None:
            wanted = {r['requirement_id'] for r in story['requirements'] if r['dimension'] in _DIMENSION_COLUMN
                      and _DIMENSION_COLUMN[r['dimension']] in CONSISTENCY_COLUMNS}
            missing = wanted - set(shot_pair.get(sid, {}))
            if missing:
                raise ValueError(f'{sid}: shot-pair answers missing for {sorted(missing)[:3]}')
            answers = {**old, **{rid: shot_pair[sid][rid] for rid in wanted}}
        return (*v8_premises(items[sid]['items'], verdicts), answers)
    return premises_of


def v11_readout(bank_path: Path, items_path: Path, cut_root: Path, pair_root: Path, video_root: Path,
                rows: list[str], storyboards_path: Path | None = None, *, answer_scope: str = 'complete') -> dict:
    """第11版判官链的读数，计分身份取正式v11／v12／v13题库：人物、物体、地点题读镜对判官（``consistency_pair_batch``），事件与状态题读整片判官
    （``gemini_checklist_batch``），前提与切镜执行列读 ``cut_execution`` 的判定。前提不成立的题记不过、不要答案；
    complete范围把前提成立却没有有效答案的故事列为待判，不计入均值；existing范围按可判题目聚合并另报缺测／覆盖。
    生成失败的故事不进各列均值，全过记不通过。"""
    from mspc.evaluation import consistency_pair_batch
    bank_data = json.loads(Path(bank_path).read_text())
    item_data = json.loads(Path(items_path).read_text())
    version = bank_data['version']
    if version not in ('v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17') or item_data['from_bank'] != version:
        raise ValueError('readout requires matching v11 to v17 question and cut-item banks')
    # 第14、15版只从第13版题库移出物体题与含糊状态题，第16版只给状态建立题配上变化事件，第17版只移出输入没给足的物体题，判定与前提口径与第13版相同。
    rules = 'v13' if version in ('v14', 'v15', 'v16', 'v17') else version
    if answer_scope not in ('complete', 'existing') or (answer_scope == 'existing' and rules != 'v13'):
        raise ValueError('existing answer scope requires a v13 to v17 bank')
    bank, items = bank_data['stories'], item_data['stories']
    durations = None if storyboards_path is None else storyboard_shot_seconds(storyboards_path)
    out = {}
    for row in rows:
        verdicts = json.loads((cut_root / 'verdicts' / f'{row}.json').read_text())
        if rules == 'v13':
            metrics = json.loads((cut_root / 'pairs' / f'{row}.json').read_text())
            verdicts = {**verdicts,
                        **{key: {**verdicts.get(key, {'verdict': None}), 'corr_mid': value['corr_mid']}
                           for key, value in metrics.items()}}
        pair = consistency_pair_batch.answers(pair_root, row)
        video_dir = video_root / 'rows' / row
        prepared = json.loads((video_dir / 'prepared.json').read_text())
        video = judged_answers(video_dir)
        stories, pending = [], {}
        for story in prepared['stories']:
            sid = story['story_id']
            answers = {**video.get(sid, {}), **{rid: a for rid, a in pair.items() if rid.startswith(sid + '/')}}
            requirements = bank[sid]['requirements']
            if rules == 'v13' and any(q.get('requires_fresh_answer') for q in requirements):
                # 改过锚点的题只能用实际按新时段送判的答案；新回执到位后即可正常计分。
                judged = {q['requirement_id']: q for q in story['requirements']}
                requirements = [
                    {k: v for k, v in q.items() if k != 'requires_fresh_answer'}
                    if q.get('requires_fresh_answer') and q['requirement_id'] in judged
                    and all(q.get(k) == judged[q['requirement_id']].get(k)
                            for k in ('shot', 'reference_shot', 'anchor', 'question')) else q
                    for q in requirements]
            try:
                premises, holds, cut_passed = v8_premises(items[sid]['items'], verdicts,
                                                         scoring_version=rules, answer_scope=answer_scope)
                scores = story_scores(requirements, premises, answers, holds, cut_passed,
                                      answer_scope=answer_scope)
            except ValueError as error:
                pending[sid] = str(error)
                continue
            if durations is not None:
                scores['event_duration_groups'] = event_duration_scores(
                    bank[sid]['requirements'], premises, answers, holds, durations[sid], answer_scope=answer_scope)
            stories.append({'story_id': sid, **scores})

        def column_mean(values: list) -> dict:
            present = [v for v in values if v is not None]
            return {'mean': 100 * mean(present) if present else None, 'n': len(present)}
        holding = {c: column_mean([s['premise_holding'][c]['score'] for s in stories]) for c in CONSISTENCY_COLUMNS}
        out[row] = {'stories': len(stories), 'pending': pending,
                    'generation_unavailable': [s['story_id'] for s in prepared['unavailable_stories']],
                    'columns': {c: column_mean([s['scores'][c] for s in stories]) for c in COLUMNS},
                    'all_pass': sum(s['all_pass'] is True for s in stories),
                    'all_pass_coverage': sum(s['all_pass'] is not None for s in stories),
                    'premises_failed': {'count': sum(s['premises_failed'] for s in stories),
                                        'of': sum(s['premises'] for s in stories)},
                    'consistency_where_premise_holds': holding, 'per_story': stories}
        if rules == 'v13':
            out[row]['auxiliary'] = {
                key: {**column_mean([s['auxiliary'][key]['score'] for s in stories]),
                      **{field: sum(s['auxiliary'][key][field] for s in stories)
                         for field in ('questions', 'of', 'missing')},
                      'covered_stories': sum(s['auxiliary'][key]['score'] is not None for s in stories),
                      'applicable_stories': sum(s['auxiliary'][key]['of'] > 0 for s in stories),
                      'complete_stories': sum(s['auxiliary'][key]['of'] > 0 and s['auxiliary'][key]['missing'] == 0
                                              for s in stories)}
                for key in (f'state_{role}' for role in STATE_ROLES)}
            out[row]['question_coverage'] = {
                c: {**{field: sum(s['question_coverage'][c][field] for s in stories)
                       for field in ('questions', 'of', 'missing')},
                    'complete_stories': sum(s['question_coverage'][c]['of'] > 0 and
                                            s['question_coverage'][c]['missing'] == 0 for s in stories)} for c in COLUMNS}
            out[row]['measurement_missing'] = {
                s['story_id']: s['measurement_missing'] for s in stories if s['measurement_missing']}
            out[row]['complete_stories'] = sum(not s['measurement_missing'] for s in stories)
        if durations is not None:
            out[row]['event_duration_groups'] = {
                group: {**column_mean([s['event_duration_groups'][group]['score'] for s in stories]),
                        'questions': sum(s['event_duration_groups'][group]['questions'] for s in stories)}
                for group in EVENT_DURATION_GROUPS}
    return {'version': version, 'judge_contract': 'v11', 'answer_scope': answer_scope,
            'score_coverage': 'available questions; missing excluded' if answer_scope == 'existing' else 'complete stories',
            'bank': str(bank_path), 'cut_items': str(items_path), 'cut_root': str(cut_root),
            'pair_root': str(pair_root), 'video_root': str(video_root),
            'storyboard_durations': None if storyboards_path is None else str(storyboards_path), 'rows': out}


def storyboard_shot_seconds(path: Path) -> dict[str, dict[int, float]]:
    """从实际给定计划的显式 FPS 和逐镜帧数取秒数；不假定底座或输出为16fps。"""
    out = {}
    for story in json.loads(path.read_text())['stories']:
        sid, fps = story['story_id'], story['fps']
        if sid in out:
            raise ValueError(f'{sid}: duplicate storyboard')
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'{sid}: fps must be a positive finite number')
        shots = {}
        for shot in story['storyboard']['shots']:
            number, frames = shot['shot'], shot['frame_count']
            if type(number) is not int or number < 1 or number in shots:
                raise ValueError(f'{sid}: shot numbers must be positive unique integers')
            if type(frames) is not int or frames < 1:
                raise ValueError(f'{sid}/shot-{number}: frame_count must be a positive integer')
            shots[number] = frames / fps
        out[sid] = shots
    return out


def event_duration_scores(rows: list[dict], premises: list[dict], answers: dict, holds: dict[str, bool],
                          shot_seconds: dict[int, float], *, answer_scope: str = 'complete') -> dict:
    """同一计分owner按计划镜长切事件题，前提规则复用story_scores；不以参考片或生成片改题目分组。"""
    groups = {}
    for group in EVENT_DURATION_GROUPS:
        selected = [r for r in rows if r['dimension'] == 'event'
                    and ('at_most_5_seconds' if shot_seconds[r['shot']] <= 5 else 'over_5_seconds') == group]
        ids = {r['requirement_id'] for r in selected}
        dependencies = [{**p, 'planned_cut': False, 'dependents': [r for r in p['dependents'] if r in ids]}
                        for p in premises if any(r in ids for r in p['dependents'])]
        result = story_scores(selected, dependencies, answers, holds, {}, answer_scope=answer_scope)
        groups[group] = {'score': result['scores']['event'], 'questions': result['question_coverage']['event']['questions']}
    return groups


def stored_cuts(v6_root: Path, pairs: list[tuple[str, str]], metrics_root: Path) -> dict:
    """非 VLM 运行已存的整片检出切点（与判分行同一部成片、同一档、seed 42）。"""
    out = {}
    for row, method in pairs:
        prepared = json.loads((v6_root / 'rows' / row / 'prepared.json').read_text())
        folder = metrics_root / method / method / prepared['input_tier'] / 'seed-42'
        out[row] = {}
        for story in prepared['stories']:
            record = json.loads((folder / f"{story['story_id']}.json").read_text())
            out[row][story['story_id']] = {'detected_cut_times_seconds': record['non_vlm']['detected_cut_times_seconds'],
                                           'duration_seconds': record['media']['duration_seconds']}
    return out


def detect_cuts(v6_root: Path, rows: list[str], ssh: str, python: str, repo: str, resource_root: str) -> dict:
    """在主服务器 CPU 上对判分行的远端成片整片跑 TransNetV2；成片与帧率取判分行的 ``media-map.json``。"""
    from mspc.evaluation.same_shot_calibration import run_splice
    jobs, fps = [], {}
    for row in rows:
        media = json.loads((v6_root / 'rows' / row / 'media-map.json').read_text())
        for story in json.loads((v6_root / 'rows' / row / 'prepared.json').read_text())['stories']:
            entry = media[story['story_id']]
            key = f"{row}|{story['story_id']}"
            fps[key] = entry['probe']['fps']
            jobs.append({'key': key, 'video': entry['remote_video_path'], 'fps': fps[key],
                         'ranges': [[0, sum(entry['shot_frames'])]]})
    found = run_splice(jobs, ssh, python, repo, resource_root, None)
    out = {row: {} for row in rows}
    for key, value in found.items():
        row, sid = key.split('|')
        out[row][sid] = {'detected_cut_times_seconds': [b / fps[key] for b in value['boundaries']],
                         'duration_seconds': value['frames'] / fps[key]}
    return out


def paired_intervals(readout: dict, pairs: list[tuple[str, str]], films: dict[str, str],
                     event_duration_groups: bool = False) -> list[dict]:
    """右减左的显式版本配对差（百分点）：六列、全过及第13版起的状态子项，故事等权，区间按源片重采样。

    一侧可写成逗号分隔的几行（同一设置的几个种子）：每个故事先在有分的几行上取平均（某个种子成片或测量失败的故事
    用其余种子的平均，三种子页「合并」规则）；种子不另作重采样单位（项目约定：不确定性来自按源片重采样同一批故事）。"""
    from mspc.benchmark.layout_baseline import film_paired_interval
    version = readout_version(readout)
    if event_duration_groups and version not in ('v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17'):
        raise ValueError('event duration groups require a v11, v12, v13 or v14 readout')
    metrics = (*COLUMNS, 'all_pass')
    state_metrics = tuple(f'state_{role}' for role in STATE_ROLES) if version in ('v13', 'v14', 'v15', 'v16', 'v17') else ()
    metrics += state_metrics
    if event_duration_groups:
        metrics += tuple(f'event_{g}' for g in EVENT_DURATION_GROUPS)

    def one(row: str) -> dict:
        out = {}
        for story in readout['rows'][row]['per_story']:
            scored = story['v8'] if version == 'v9' else story
            values = {**{c: None if v is None else 100 * v for c, v in scored['scores'].items()},
                      'all_pass': None if scored['all_pass'] is None else 100.0 * scored['all_pass']}
            for metric in state_metrics:
                score = scored['auxiliary'][metric]['score']
                values[metric] = None if score is None else 100 * score
            if event_duration_groups:
                for group in EVENT_DURATION_GROUPS:
                    score = scored['event_duration_groups'][group]['score']
                    values[f'event_{group}'] = None if score is None else 100 * score
            out[story['story_id']] = values
        return out

    def values(side: str) -> dict:
        rows = [one(row) for row in side.split(',')]
        out = {}
        for sid in set().union(*rows):
            out[sid] = {}
            for m in metrics:
                got = [r[sid][m] for r in rows if sid in r]
                out[sid][m] = None if any(v is None for v in got) else sum(got) / len(got)
        return out

    out = []
    for left, right in pairs:
        a, b = values(left), values(right)
        out.append({'left': left, 'right': right,
                    'metrics': {m: film_paired_interval(a, b, films, m) for m in metrics}})
    return out


def readout_version(readout: dict) -> str:
    """版本来自产物合同，不从某个字段是否碰巧存在猜格式。"""
    if readout.get('version') in ('v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17') and readout.get('judge_contract') == 'v11':
        return readout['version']
    if readout.get('scoring') == 'v9':
        return 'v9'
    raise ValueError('paired readout must declare scoring=v9 or version=v11/v12/v13/v14 with judge_contract=v11')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    s = commands.add_parser('stored-cuts')
    s.add_argument('--v6-root', type=Path, required=True, help='第 6 版题库的判分运行根')
    s.add_argument('--metrics-root', type=Path, required=True, help='非 VLM 运行的 metrics 目录')
    s.add_argument('--pair', nargs=2, action='append', metavar=('ROW', 'METHOD'), required=True)
    s.add_argument('--output', type=Path, required=True)
    d = commands.add_parser('detect-cuts')
    d.add_argument('--v6-root', type=Path, required=True)
    d.add_argument('--rows', nargs='+', required=True)
    d.add_argument('--ssh', required=True)
    d.add_argument('--python', required=True, help='主服务器上 TF-only 的 TransNetV2 解释器')
    d.add_argument('--repo', required=True, help='主服务器上与本机同一提交的代码目录')
    d.add_argument('--resource-root', required=True)
    d.add_argument('--output', type=Path, required=True)
    for version in ('v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17'):
        v = commands.add_parser(f'readout-{version}')
        v.add_argument('--bank', type=Path, required=True, help=f'{version}题库')
        v.add_argument('--cut-items', type=Path, required=True, help=f'{version}切镜项')
        v.add_argument('--cut-root', type=Path, required=True, help='cut_execution 运行根（verdicts/<行>.json）')
        v.add_argument('--pair-root', type=Path, required=True, help='consistency_pair_batch 运行根')
        v.add_argument('--video-root', type=Path, required=True, help='gemini_checklist_batch 运行根（事件与状态）')
        v.add_argument('--rows', nargs='+', required=True)
        v.add_argument('--storyboards', type=Path, help='实际给定计划清单；每故事显式fps和逐镜frame_count，用于≤5秒／>5秒事件读数')
        if version in ('v13', 'v14', 'v15', 'v16', 'v17'):
            v.add_argument('--answer-scope', choices=('complete', 'existing'), default='complete',
                           help='complete要求完整回执；existing逐题报告缺测，仅聚合可判题目（前提失败仍计0）')
        v.add_argument('--output', type=Path, required=True)
    e = commands.add_parser('readout')
    e.add_argument('--v6-root', type=Path, required=True, help='第 6 版题库的判分运行根')
    e.add_argument('--v8-root', type=Path, required=True, help='cut_execution 的运行根（verdicts/<行>.json）')
    e.add_argument('--items', type=Path, required=True, help='question_bank_v6 cut-items 写出的切镜执行项')
    e.add_argument('--scored-bank', type=Path, required=True, help='question_bank_v6 scored-v9 写出的第 9 版计分题库')
    e.add_argument('--rows', nargs='+', required=True)
    e.add_argument('--consistency-judge', choices=['whole_video', 'shot_pair'], required=True,
                   help='人物／物体／地点跨镜题用整片判官原答，还是 shot_pair_consistency 的一步判答案')
    e.add_argument('--shot-pair-root', type=Path, help='shot_pair_consistency 的运行根（answers/<行>.json）；shot_pair 时必给')
    e.add_argument('--output', type=Path, required=True)
    q = commands.add_parser('paired')
    q.add_argument('--readout', type=Path, nargs='+', required=True, help='readout 写出的读数（可多份，行名不重复）')
    q.add_argument('--requests', type=Path, required=True, help='含 film_key 的请求文件')
    q.add_argument('--pair', nargs=2, action='append', metavar=('LEFT', 'RIGHT'), required=True)
    q.add_argument('--event-duration-groups', action='store_true', help='同时配对readout --storyboards产生的≤5秒／>5秒事件列')
    q.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'paired':
        merged = {'rows': {}}
        version = None
        for path in args.readout:
            source = json.loads(path.read_text())
            found_version = readout_version(source)
            if version is not None and found_version != version:
                raise ValueError('paired readouts must share the same scoring version')
            version = found_version
            for row, value in source['rows'].items():
                if row in merged['rows']:
                    raise ValueError(f'{row}: given twice')
                merged['rows'][row] = value
        merged.update({'scoring': 'v9'} if version == 'v9' else {'version': version, 'judge_contract': 'v11'})
        films = {s['story_id']: s['film_key'] for s in json.loads(args.requests.read_text())['stories']}
        result = {'version': version, 'difference': 'right - left, percentage points', 'weighting': 'story-equal',
                  'interval': '95% bootstrap over source films (requests film_key), 10000 rounds, seed 42',
                  'readouts': [str(p) for p in args.readout],
                  'pairs': paired_intervals(merged, args.pair, films, args.event_duration_groups)}
        if version in ('v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17'):
            result['judge_contract'] = 'v11'
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
        for pair in result['pairs']:
            print(pair['left'], '->', pair['right'], {m: (round(v['mean'], 1), [round(x, 1) for x in v['ci95']], v['shots'])
                                                     for m, v in pair['metrics'].items()})
        return
    if args.command in ('stored-cuts', 'detect-cuts'):
        if args.command == 'stored-cuts':
            rows = stored_cuts(args.v6_root, args.pair, args.metrics_root)
            source = {'stored_non_vlm_metrics': str(args.metrics_root)}
        else:
            rows = detect_cuts(args.v6_root, args.rows, args.ssh, args.python, args.repo, args.resource_root)
            source = {'transnetv2_cpu': args.resource_root, 'code': args.repo}
        args.output.write_text(json.dumps({'source': source, 'rows': rows}, ensure_ascii=False, indent=1) + '\n')
        print(json.dumps({row: len(stories) for row, stories in rows.items()}))
        return
    if args.command in ('readout-v11', 'readout-v12', 'readout-v13', 'readout-v14', 'readout-v15', 'readout-v16', 'readout-v17'):
        if json.loads(args.bank.read_text())['version'] != args.command.removeprefix('readout-'):
            raise ValueError('readout command and bank version differ')
        result = v11_readout(args.bank, args.cut_items, args.cut_root, args.pair_root, args.video_root, args.rows,
                             args.storyboards, answer_scope=args.answer_scope if args.command in ('readout-v13', 'readout-v14', 'readout-v15', 'readout-v16', 'readout-v17') else 'complete')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
        for row, value in result['rows'].items():
            print(json.dumps({'row': row, 'stories': value['stories'], 'pending': len(value['pending']),
                              **{c: None if v['mean'] is None else round(v['mean'], 1) for c, v in value['columns'].items()}}))
        return
    if args.command == 'readout':
        if (args.consistency_judge == 'shot_pair') != (args.shot_pair_root is not None):
            raise ValueError('--shot-pair-root is given exactly when --consistency-judge shot_pair')
        items = json.loads(args.items.read_text())['stories']
        scored = {sid: {r['requirement_id'] for r in story['requirements']}
                  for sid, story in json.loads(args.scored_bank.read_text())['stories'].items()}
        result = {'rules': 'mspc.evaluation.cut_execution', 'scoring': 'v9',
                  'tool': 'src/mspc/evaluation/checklist_scoring.py readout',
                  'sources': {'v6_answers': str(args.v6_root), 'cut_items': str(args.items), 'scored_bank': str(args.scored_bank),
                              'verdicts': str(args.v8_root / 'verdicts'), 'consistency_judge': args.consistency_judge,
                              'shot_pair_answers': None if args.shot_pair_root is None else str(args.shot_pair_root / 'answers')},
                  'rows': {}}
        for row in args.rows:
            verdicts = json.loads((args.v8_root / 'verdicts' / f'{row}.json').read_text())
            shot_pair = None
            if args.consistency_judge == 'shot_pair':
                shot_pair = json.loads((args.shot_pair_root / 'answers' / f'{row}.json').read_text())
            result['rows'][row] = row_readout(args.v6_root / 'rows' / row, v8_source(items, verdicts, shot_pair), scored)
        report(result, args.output)


def report(result: dict, output: Path) -> None:
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    for row, value in result['rows'].items():
        print(json.dumps({'row': row, 'stories': value['stories'], 'pending': len(value['premise_pending']),
                          'change': {k: round(v, 2) for k, v in value['change_points'].items()},
                          'cut': value['v8']['columns']['cut'], 'all_pass': (value['v6']['all_pass'], value['v8']['all_pass']),
                          'voided': value['voided_passes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
