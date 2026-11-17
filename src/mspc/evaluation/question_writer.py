"""出题：模型只从请求与源故事标注里抽动作与结果，题目由三条固定规则生成。

规则：
1. 事件：每个作用到物体或另一个人、或整个身体位置改变的动作一题；为同一目的的连续步骤算一个动作。
2. 状态：动作的可见结果在该镜末一题；结果改变人或物的外观时，在该人／物之后第一次与最后一次出现时各再问一题；
   请求之后又改变同一人／物外观的，复查只到那次改变之前（「最后一次」取改变前最后一次出现）。持有与位置只问镜末。
   保持复查（11-03 追加的单列题族 `persistence`，由 `persistence_questions` 在已复核题库上追加，不改本规则已出的题）：
   外观结果在该人／物此后每一次出现都再问一题，同样只到下一次外观改变之前；纯持有结果（题面只问某人拿着某物，
   不带举起、贴耳、指向等暂时姿态，且能从题面点出唯一持有者与被持物）在持有者与被持物都出现的后续镜各再问一题，
   只到请求里下一个参与者或结果涉及二者之一的动作所在镜之前。复查题问镜中，与已有同镜同题面的状态题重复的不出；
   源结果的镜末题或本规则原有的首末复查在复核中被删改的，整条不复查；某人／物在某镜有原有复查被复核删掉的，
   该镜不再对它复查（复核删题的理由是「不是持续变化」或「该镜不要求它出镜」）。位置与暂时姿态仍只问镜末。
   标注保持复查（11-03 追加的单列题族 `persistence_annotation`；由
   `annotation_persistence_questions` 在已复核题库与上面的保持复查题之后追加）：题源换成给定分镜表标注里的
   `changes`，只取持有、状态、服装三类（姿态不出），所以只适用于方法吃同一份给定分镜表的读数。某人／物 E 在第 s 镜
   发生这类变化后，对此后每个 E 在场、且标注仍是变化后取值的镜 k 各问一题：持有＝变化后所持之物都还在 E 的
   `holding` 里，状态＝E 的 `state` 与变化后取值相同，服装＝E 的 `outfit` 相同（比较时忽略大小写与多余空白）。
   在场要标注与请求两边都在（请求一侧同规则 3，按抽取）。第13版持有复查只要求持有者按请求出镜，
   且标注的 ``holding`` 确认持有未变；请求没有再次点名被持物不再成为删题理由。持有者本身未请求出镜仍不出题。
   请求明确抛出钓线时，鱼饵已离手，该镜起终止鱼饵持有链；不沿用之后残留的 ``holding`` 标注。
   标注没写这一字段的镜不问也不停；遇到 E 同一属性下一次变化所在的镜、或标注写出别的取值的镜即停，该镜起不再问。
   变化后不再持有任何物（放下）不出题。镜 k 已有同一人／物同类状态题的不出：持有对已有纯持有题（同一持有者、
   已问之物包含本题之物），状态与服装对已有以 E 为对象的外观结果题。题面（名字用请求里的称呼 `writer_output.names`）：
   持有「Is H holding O?」（多件写 O1 and O2）；状态取值以 with 开头时「Does E have …?」，其余「Is E …?」；
   服装「Is E wearing …?」。都是镜中（during）题，`depends_on` 为 E（持有另加所持之物），`source_shot` 记 s，
   `source_change` 记变化类别。
3. 一致性：在两镜及以上出现的人、物、地点，首次出现分别对第二次与最后一次出现问是否同一个；人、动物、群体另对
   每一次离场后再现（中间至少一镜不在场）各问一题（长程一致要查回来的人）。
4. 切镜执行项（第 8 版，新机位口径；`cut_items` 在已复核题库上生成）：计划要求
   两镜不同——标注地点不同，或请求写的景别、角度不同（逐镜档请求每镜写明景别，非平视另写角度，与分镜表 `camera`
   逐镜一致），或换人（两镜在场的人没有交集，如正反打、切到无人空镜；在场按 ``appearances``）——的镜对出一项。镜对取三处：一致性题与
   跨镜难反例比较的两镜，主表状态复查题的变化镜与复查镜，以及每个相邻计划切点前后两镜（切镜执行列）；同一镜对只出
   一项，记下以它为前提的题。判定（TransNetV2、帧相似度、判官看四帧）在 `cut_execution`，计分在 `checklist_scoring`。
   第 7 版的前提题（问判官"是不是同一个镜头"）只留在已出的题库文件里，供按第 7 版补判的行用。
   起因：没切开的镜对一致性反而更高，构图太像的照抄同样不该算一致，所以 11-05 改为新机位口径。
在场与地点只按请求：由 ``request_presence`` 让 gpt-6.1-sol 只读
请求，逐镜给出请求要求出镜的人、物（只被看着、说到、提到的不算）与读者能看出的地点（看不出的镜不出地点题，
也不按地点算计划差异），经 ``with_request_presence`` 换进写题抽取；源片标注不参与。
规则 2 另补（`annotation_change_states`）：标注里持有、状态、服装的变化，变化那一镜还没有
以该人／物为对象的镜末状态题、且请求在该镜提到它（持有另要所持之物）的，在变化那一镜镜末补一题，题面同标注保持复查。换人只在人物、物体题扣分：事件与状态题按判官对该题的回答计，
不继承人物题的失败；人或物不在画面时
判官本就答 no。

第 6 版计分（`story_scores`，第 7 版的底层答案与附录读数；现行第 7 版见 `checklist_scoring`）五项：人物、物体、场景、事件、状态；每题只看判官是否答 yes，不依赖别的题。「场景一致性」只计地点题。
布局题（dimension `layout`，问两个实体切镜后左右或前后顺序是否保持）不再出题、不计分、不进全部适用题通过，也不再报。实测理由：100 道布局题考的是请求没写的源片站位；
过半跨 4 镜以上（最远 11 镜）、跨机位比较；只有开发集 13、测试集 31 个故事有；读数不稳（开发集段容量 2／3／4 镜给出
51.4／79.2／88.5）。11-03 前的题库里仍有布局题：照常读入、照原样进判官提示（已判回执按题号对回答案），计分时略过。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / 'configs/eval/question_writer/prompt.md'
SCHEMA = ROOT / 'configs/eval/question_writer/schema.json'


def registry(board: dict) -> dict:
    """第 18 版分镜表的登记：编号 → 类别（person／animal／group／object／place）。"""
    kinds = {c['name']: c['kind'] for c in board['characters']}
    kinds.update({o['name']: 'object' for o in board['objects']})
    kinds.update({p['name']: 'place' for p in board['places']})
    return kinds


def writer_input(request: str, board: dict) -> str:
    """请求原文＋同一故事的第 18 版分镜表（登记表、每镜动作、在场、地点），后者只用来对齐编号与镜号。"""
    registered = [{'id': c['name'], 'kind': c['kind'], 'description': c['appearance']} for c in board['characters']]
    registered += [{'id': o['name'], 'kind': 'object', 'description': o['appearance']} for o in board['objects']]
    registered += [{'id': p['name'], 'kind': 'place', 'description': p['appearance']} for p in board['places']]
    shots = [{'shot': s['shot'], 'description': s['action'], 'present': [e['name'] for e in s['present']], 'place': s.get('place')}
             for s in board['shots']]
    return (PROMPT.read_text() + '\n\n## Request\n' + request + '\n\n## Annotation\n'
            + json.dumps({'registered': registered, 'shots': shots}, ensure_ascii=False, indent=1))


def validate(output: dict, board: dict) -> None:
    ids = set(registry(board))
    assert {n['id'] for n in output['names']} == ids
    count = len(board['shots'])
    assert [s['shot'] for s in output['shots']] == list(range(1, count + 1))
    for shot in output['shots']:
        assert set(shot['present']) <= ids
    for action in output['actions']:
        assert 1 <= action['shot'] <= count and action['question'].strip()
        assert set(action['participants']) <= ids
        if action['result']:
            assert action['result']['entity'] in ids and action['result']['question'].strip()


LIVING = ('person', 'animal', 'group')


def with_request_presence(output: dict, presence: dict) -> dict:
    """写题抽取换上只按请求判的逐镜在场与地点（``request_presence`` 的结果）：``present`` 是请求要求出镜的人、物，
    ``location`` 是读者能从请求看出的地点（看不出为 None）。"""
    by_shot = {s['shot']: s for s in presence['shots']}
    if sorted(by_shot) != [s['shot'] for s in output['shots']]:
        raise ValueError('request presence and writer output cover different shots')
    return {**output, 'shots': [{'shot': k, 'present': by_shot[k]['on_screen'], 'location': by_shot[k]['location']}
                                for k in sorted(by_shot)]}


def appearances(output: dict, board: dict) -> dict:
    """每个人／物／地点在哪几镜（1 起），只按请求：
    ``output['shots']`` 是 ``with_request_presence`` 换上的逐镜在场与地点。只被看着、说到、提到的不算在场；
    请求看不出在哪个地点的镜不算该地点出现。``board`` 只用来校验编号。"""
    kinds = registry(board)
    seen = {}
    for shot in output['shots']:
        for name in shot['present']:
            seen.setdefault(name, []).append(shot['shot'])
        if shot['location'] is not None:
            if kinds[shot['location']] != 'place':
                raise ValueError(f"shot {shot['shot']}: {shot['location']} is not a registered place")
            seen.setdefault(shot['location'], []).append(shot['shot'])
    return seen


def questions(story_id: str, output: dict, board: dict) -> list[dict]:
    """按三条规则把抽取结果变成题目；`{shot k}` 在打分时换成该方法第 k 镜的实际时间段。"""
    validate(output, board)
    names = {n['id']: n['name'] for n in output['names']}
    kinds = registry(board)
    seen = appearances(output, board)
    changed = {}  # 人／物 → 请求里改变其外观的镜号
    for action in output['actions']:
        if action['result'] and action['result']['kind'] == 'appearance':
            changed.setdefault(action['result']['entity'], []).append(action['shot'])
    rows = []

    def add(dimension, shot, text, depends_on, reference_shot=None, anchor='during'):
        rows.append({'requirement_id': f'{story_id}/q{len(rows) + 1:02d}', 'dimension': dimension, 'shot': shot,
                     'reference_shot': reference_shot, 'anchor': anchor, 'question': text, 'depends_on': depends_on})

    for action in output['actions']:
        add('event', action['shot'], action['question'], action['participants'])
        result = action['result']
        if not result:
            continue
        add('state', action['shot'], result['question'], [result['entity']], anchor='end')
        if result['kind'] == 'appearance':
            until = min([k for k in changed[result['entity']] if k > action['shot']], default=len(board['shots']) + 1)
            later = [k for k in seen.get(result['entity'], []) if action['shot'] < k < until]
            for k in sorted({later[0], later[-1]} if later else set()):
                add('state', k, result['question'], [result['entity']])
    for entity, shots in seen.items():
        if len(shots) < 2:
            continue
        kind = kinds[entity]
        if kind == 'place':
            # 与人、物同一规则：首次出现对第二次、最后一次。正反打交替时“每次回归”会让地点题数随镜数暴涨，不再单列。
            for k in sorted({shots[1], shots[-1]}):
                add('place', k, f'Are {{shot {k}}} and {{shot {shots[0]}}} in the same place ({names[entity]})?', [],
                    reference_shot=shots[0])
        else:
            living = kind in LIVING
            noun = {'person': 'person', 'animal': 'animal', 'group': 'group'}.get(kind, 'object')
            for k in consistency_shots(shots, living):
                add('character' if living else 'object', k, consistency_question(names[entity], noun, k, shots[0]), [entity],
                    reference_shot=shots[0])
    return rows


def consistency_question(name: str, noun: str, shot: int, first: int) -> str:
    return f'Is {name} at {{shot {shot}}} the same {noun} as at {{shot {first}}}?'


def consistency_shots(shots: list[int], living: bool) -> list[int]:
    """规则 3 拿来与首次出现比较的镜：第二次与最后一次出现；人、动物、群体另加每一次离场后再现。"""
    picked = {shots[1], shots[-1]}
    if living:
        picked |= {k for previous, k in zip(shots, shots[1:]) if k - previous > 1}
    return sorted(picked)


# 纯持有的题面，持有者记 H、被持物记 O；带姿态的（held up、held to her ear、in a raised position）不在内。
PLAIN_HOLDING = re.compile(r"Is O (?:now )?held by H\?|Is H (?:now )?holding O\?|Does H (?:now )?hold O\?"
                           r"|Is O (?:now )?in H's hands?\?")


def held(question: str, names: dict, kinds: dict) -> tuple[str, list[str]] | None:
    """纯持有结果的持有者与被持物编号；不是纯持有题面、或点不出唯一持有者与至少一件物体时返回 None。"""
    text, holders, objects = question, [], []
    for entity in sorted(names, key=lambda e: -len(names[e])):
        pattern = rf'(?<!\w){re.escape(names[entity])}(?!\w)'
        if kinds[entity] == 'place' or not re.search(pattern, text):
            continue
        living = kinds[entity] != 'object'
        (holders if living else objects).append(entity)
        text = re.sub(pattern, 'H' if living else 'O', text)
    text = re.sub(r'O(?:, O)*,? and O', 'O', text)
    if len(holders) != 1 or not objects or not PLAIN_HOLDING.fullmatch(text):
        return None
    return holders[0], objects


def persistence_questions(story_id: str, output: dict, board: dict, existing: list[dict]) -> list[dict]:
    """规则 2 的保持复查题，只返回新增题；编号接在 ``existing`` 最大 q 号之后，已有题与编号不动。"""
    validate(output, board)
    names = {n['id']: n['name'] for n in output['names']}
    kinds = registry(board)
    seen = appearances(output, board)
    present = {shot['shot']: set(shot['present']) for shot in output['shots']}
    end = len(board['shots']) + 1
    asked = {(q['dimension'], q['shot'], q['anchor'], q['question']) for q in existing}
    have = {(q['dimension'], q['shot'], q['question']) for q in existing}
    tails = [q['requirement_id'].rsplit('/', 1)[1] for q in existing]
    number = max((int(tail[1:]) for tail in tails if tail.startswith('q')), default=0)
    changed = {}
    for action in output['actions']:
        if action['result'] and action['result']['kind'] == 'appearance':
            changed.setdefault(action['result']['entity'], []).append(action['shot'])

    def window(action):
        until = min([k for k in changed[action['result']['entity']] if k > action['shot']], default=end)
        return [k for k in seen.get(action['result']['entity'], []) if action['shot'] < k < until]

    looks = [a for a in output['actions'] if a['result'] and a['result']['kind'] == 'appearance']
    rejected = {(a['result']['entity'], k) for a in looks for k in window(a)[:1] + window(a)[-1:]
                if ('state', k, 'during', a['result']['question']) not in asked}
    rows = []
    for action in output['actions']:
        result, shot = action['result'], action['shot']
        if not result or ('state', shot, 'end', result['question']) not in asked:
            continue
        if result['kind'] == 'appearance':
            checks = window(action)
            if any(('state', k, 'during', result['question']) not in asked for k in checks[:1] + checks[-1:]):
                continue
            involved = {result['entity']}
        else:
            holding = held(result['question'], names, kinds)
            if holding is None:
                continue
            involved = {holding[0], *holding[1]}
            until = min([a['shot'] for a in output['actions'] if a['shot'] > shot and involved & {
                *a['participants'], *([a['result']['entity']] if a['result'] else [])}], default=end)
            checks = [k for k in range(shot + 1, until) if involved <= present[k]]
        for k in checks:
            if ('state', k, result['question']) in have or any((e, k) in rejected for e in involved):
                continue
            have.add(('state', k, result['question']))
            number += 1
            rows.append({'requirement_id': f'{story_id}/q{number:02d}', 'dimension': 'state', 'shot': k,
                         'reference_shot': None, 'anchor': 'during', 'question': result['question'],
                         'depends_on': [result['entity']], 'family': 'persistence', 'source_shot': shot})
    return rows


#: 分镜表 `camera.shot_size` 七档，由近到远（景别差按档数计）。
SHOT_SIZES = ('extreme close-up shot', 'close-up shot', 'medium close-up shot', 'medium shot',
              'medium wide shot', 'wide shot', 'extreme wide shot')
def planned_difference(earlier: dict, later: dict, cast: tuple[set, set]) -> dict | None:
    """两镜在计划里差在哪；同地点、同景别、同角度、同一批人返回 None。

    没写角度按平视，任一镜没写景别不按景别算差；地点按请求，任一镜看不出地点不按地点算差。``cast`` 是两镜按请求
    在场的人、动物与群体；换人＝至少一镜有人且两镜没有共同的人（正反打、切到无人空镜）。"""
    first, second = earlier.get('camera') or {}, later.get('camera') or {}
    sizes = first.get('shot_size'), second.get('shot_size')
    gap = abs(SHOT_SIZES.index(sizes[0]) - SHOT_SIZES.index(sizes[1])) if all(sizes) else 0
    angle = (first.get('angle') or 'eye-level shot') != (second.get('angle') or 'eye-level shot')
    place = earlier['place'] is not None and later['place'] is not None and earlier['place'] != later['place']
    recast = bool(cast[0] | cast[1]) and not cast[0] & cast[1]
    return ({'place': place, 'size_gap': gap, 'angle': angle, 'cast': recast}
            if place or gap or angle or recast else None)


def change_shot(row: dict, output: dict) -> int | None:
    """镜中状态题（``anchor`` 为 during）复查的变化在哪一镜：出题记录里结果题面相同、镜号在它之前的最后一个动作。
    结果出在本镜的（镜中就问结果）不是复查，返回 None。"""
    if 'source_shot' in row:
        source = row['source_shot']
        if not isinstance(source, int) or not 1 <= source < row['shot']:
            raise ValueError(f"{row['requirement_id']}: source_shot must precede the revisit")
        return source
    shots = [a['shot'] for a in output['actions'] if a.get('result') and a['result']['question'] == row['question']]
    if not shots:
        raise ValueError(f"{row['requirement_id']}: no action has this result")
    earlier = [s for s in shots if s < row['shot']]
    return max(earlier) if earlier else None


def cut_items(story_id: str, output: dict, board: dict, existing: list[dict]) -> list[dict]:
    """规则 4 的切镜执行项（第 8 版，新机位口径），编号 ``c01`` 起。

    镜对取三处：一致性题与跨镜难反例比较的两镜；主表状态复查题的变化镜与复查镜；每个相邻计划切点前后两镜。
    同一镜对只出一项，``dependents`` 列出以它为前提的题。每一项的前提都是：两镜之间检出切点、且后一镜是新机位。
    - 计划要求两镜不同（``planned_difference`` 非空）：相邻的（``planned_cut``）同时是切镜执行列的一项。
    - 有题比较、计划却不要求不同（``planned_difference`` 为 None）：前提同样要求切开且换了机位；不进切镜执行列，``planned_cut`` 记 False。
    计划不要求两镜不同、也没有题比较的相邻切点不出项。"""
    validate(output, board)
    kinds = registry(board)
    seen = appearances(output, board)
    cast = {shot['shot']: {n for n, ks in seen.items() if kinds[n] in LIVING and shot['shot'] in ks}
            for shot in board['shots']}
    located = {shot['shot']: shot['location'] for shot in output['shots']}
    shots = {shot['shot']: {**shot, 'place': located[shot['shot']]} for shot in board['shots']}  # 地点按请求
    gated: dict[tuple[int, int], list[str]] = {}
    for q in existing:
        if q['dimension'] in ('character', 'object', 'place', 'monitor') and q['reference_shot'] not in (None, q['shot']):
            pair = tuple(sorted((q['reference_shot'], q['shot'])))
        elif (q['dimension'] == 'state' and q['anchor'] == 'during'
              and q.get('state_role') != 'establishment'
              and q.get('family') not in ('annotation_change', 'event_to_state')
              and change_shot(q, output) is not None):
            # 标注变化题问的就是变化那一镜；由事件题改成的姿态、持握状态题问的是请求那一镜本身。两类都不是跨镜复查。
            pair = (change_shot(q, output), q['shot'])
        else:
            continue
        gated.setdefault(pair, []).append(q['requirement_id'])
    rows = []
    for earlier, later in sorted(set(gated) | {(k, k + 1) for k in shots if k + 1 in shots}):
        difference = planned_difference(shots[earlier], shots[later], (cast[earlier], cast[later]))
        if difference is None and (earlier, later) not in gated:
            continue
        rows.append({'item_id': f'{story_id}/c{len(rows) + 1:02d}', 'shot': later, 'reference_shot': earlier,
                     'planned_cut': difference is not None and later == earlier + 1, 'planned_difference': difference,
                     'dependents': gated.get((earlier, later), [])})
    return rows


ANNOTATION_CHANGES = ('holding', 'state', 'outfit')


def _same(value, after, what: str) -> bool:
    if what == 'holding':
        return set(after) <= set(value)
    return ' '.join(value.split()).lower() == ' '.join(after.split()).lower()


def _listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def _annotation_question(name: str, what: str, after, names: dict) -> str:
    if what == 'holding':
        return f'Is {name} holding {_listed([names[o] for o in after])}?'
    if what == 'outfit':
        return f"Is {name} wearing {after.removeprefix('wearing ')}?"
    if after.startswith('with '):
        return f"Does {name} have {after.removeprefix('with ')}?"
    if after.startswith('eyes '):  # 标注写「eyes closed」这类以身体部位开头的取值
        return f"Are {name}'s {after}?"
    return f'Is {name} {after}?'


def _holding_release_shots(output: dict, board: dict, holder: str, objects: list[str]) -> list[int]:
    """请求中的抛竿终止持饵；鱼竿仍在手中，不能把所有钓具一并当作离手。

    真实标注在抛竿后的镜仍残留持有鱼饵。只按请求抽取里同一持有者明确执行的 fishing-line cast
    以及登记为 lure／bait 的被持物识别这一冲突，不因后镜省略物体名称而停止其他持物回访。
    """
    descriptions = {e['name']: e['appearance'] for group in ('characters', 'objects') for e in board[group]}
    if not any(re.search(r'\b(?:lure|bait)\b', descriptions[o], re.IGNORECASE) for o in objects):
        return []
    return [a['shot'] for a in output['actions'] if holder in a['participants']
            and re.search(r'\bcast(?:s|ing)?\b[^?]*\bfishing line\b', a['question'], re.IGNORECASE)]


def annotation_persistence_questions(story_id: str, output: dict, board: dict, existing: list[dict]) -> list[dict]:
    """规则 2 的标注保持复查题，只返回新增题；``existing`` 是已复核题与请求保持复查题，编号接在其最大 q 号之后。"""
    validate(output, board)
    names = {n['id']: n['name'] for n in output['names']}
    kinds = registry(board)
    looks = {a['result']['question'] for a in output['actions'] if a['result'] and a['result']['kind'] == 'appearance'}
    asked_looks = {(q['shot'], e) for q in existing if q['dimension'] == 'state' and q['question'] in looks
                   for e in q['depends_on']}
    asked_holding = {}
    for q in existing:
        if q['dimension'] == 'state' and (pair := held(q['question'], names, kinds)):
            asked_holding.setdefault((q['shot'], pair[0]), []).append(set(pair[1]))
    tails = [q['requirement_id'].rsplit('/', 1)[1] for q in existing]
    number = max((int(tail[1:]) for tail in tails if tail.startswith('q')), default=0)
    requested = {shot['shot']: set(shot['present']) for shot in output['shots']}
    shots = board['shots']
    rows: list[dict] = []
    for shot in shots:
        for change in shot['changes']:
            entity, what, after = change['name'], change['what'], change['after']
            if what not in ANNOTATION_CHANGES or (what == 'holding' and not after):
                continue
            released = _holding_release_shots(output, board, entity, after) if what == 'holding' else []
            until = min((k for k in released if k >= shot['shot']), default=len(shots) + 1)
            for later in (s for s in shots if s['shot'] > shot['shot']):
                k = later['shot']
                if k >= until:
                    break
                if any(c['name'] == entity and c['what'] == what for c in later['changes']):
                    break
                entry = next((e for e in later['present'] if e['name'] == entity), None)
                if entry is None or what not in entry:
                    continue
                if not _same(entry[what], after, what):
                    break
                depends_on = [entity, *after] if what == 'holding' else [entity]
                if entity not in requested[k]:
                    continue
                if what == 'holding':
                    if any(set(after) <= objects for objects in asked_holding.get((k, entity), [])):
                        continue
                elif (k, entity) in asked_looks:
                    continue
                number += 1
                rows.append({'requirement_id': f'{story_id}/q{number:02d}', 'dimension': 'state', 'shot': k,
                             'reference_shot': None, 'anchor': 'during',
                             'question': _annotation_question(names[entity], what, after, names),
                             'depends_on': depends_on,
                             'family': 'persistence_annotation', 'source_shot': shot['shot'],
                             'source_change': what})
    return rows


# 结果题中可持续的物态词；完整词匹配避免把 coffee 的 off 当作物态。off 的空间离开用法另行排除。
PHYSICAL_RESULT = re.compile(r'\b(?:open|closed|broken|lit|off|empty|full|poured|folded|wet|clean|covered|tied|wrapped|'
                             r'sliced|cut|peeled|melted|frozen|glowing|bleeding|stained)\b', re.IGNORECASE)
SPATIAL_OFF_RESULT = re.compile(r'\boff\s+(?:the|a|an|Object|Location)\b', re.IGNORECASE)


def physical_result(question: str) -> bool:
    """可持续物态的完整词；off 后接地点／物体时是空间离开，不能当关机状态。"""
    return PHYSICAL_RESULT.search(question) is not None and SPATIAL_OFF_RESULT.search(question) is None


def persistent_result_questions(story_id: str, output: dict, board: dict, existing: list[dict]) -> list[dict]:
    """第13版补齐请求抽取的持物／物态结果回访；已有题及编号不动，不绕过后镜的状态变更。

    持物只取 ``held`` 可识别的纯持有题。持有者须在后镜请求出镜，标注 ``present.holding`` 须明确确认，
    与已问的同镜同持有者／被持物题按语义去重。物态结果在结果实体后续请求出镜镜头复查，遇到该实体下一次
    结果动作或标注状态变化即停。只使用已保留的源结果题，不恢复源结果本身已被复核删除的动作链。
    """
    validate(output, board)
    names = {n['id']: n['name'] for n in output['names']}
    kinds = registry(board)
    requested = {s['shot']: set(s['present']) for s in output['shots']}
    have = {(q['shot'], q['question']) for q in existing if q['dimension'] == 'state'}
    holding_asked: dict[tuple[int, str], list[set[str]]] = {}
    for q in existing:
        if q['dimension'] == 'state' and (pair := held(q['question'], names, kinds)):
            holding_asked.setdefault((q['shot'], pair[0]), []).append(set(pair[1]))
    number = max((int(q['requirement_id'].rsplit('/q', 1)[1]) for q in existing
                  if re.search(r'/q\d+$', q['requirement_id'])), default=0)
    rows = []
    for action in output['actions']:
        result, source = action.get('result'), action['shot']
        if not result or (source, result['question']) not in have:
            continue
        holding = held(result['question'], names, kinds)
        physical = physical_result(result['question'])
        if holding is None and not physical:
            continue
        entity = result['entity']
        until = min((a['shot'] for a in output['actions'] if a['shot'] > source and a.get('result')
                     and a['result']['entity'] == entity), default=len(board['shots']) + 1)
        if holding:
            until = min(until, min((k for k in _holding_release_shots(output, board, *holding) if k >= source),
                                   default=len(board['shots']) + 1))
        for later in board['shots']:
            k = later['shot']
            if not source < k < until:
                continue
            subject, what = (holding[0], 'holding') if holding else (entity, 'state')
            if any(c['name'] == subject and c['what'] == what for c in later['changes']):
                break
            entry = next((e for e in later['present'] if e['name'] == subject), None)
            if holding:
                objects = set(holding[1])
                if entry is not None and 'holding' in entry and not objects <= set(entry['holding']):
                    break
                if subject not in requested[k] or entry is None or 'holding' not in entry:
                    continue
                if any(objects <= asked for asked in holding_asked.get((k, subject), [])):
                    continue
            elif entity not in requested[k]:
                continue
            if (k, result['question']) in have:
                continue
            number += 1
            rows.append({'requirement_id': f'{story_id}/q{number:02d}', 'dimension': 'state', 'shot': k,
                         'reference_shot': None, 'anchor': 'during', 'question': result['question'],
                         'depends_on': [holding[0], *holding[1]] if holding else [entity],
                         'family': 'persistence_writer_result', 'source_shot': source,
                         'source_result_kind': 'holding' if holding else 'physical'})
            have.add((k, result['question']))
            if holding:
                holding_asked.setdefault((k, subject), []).append(objects)
    return rows


# 标注变化补题不出的两类状态取值：面部表情（判官判不准，StoryEval 也去掉面部表情）与
# 正在做的动作（不是结果状态，更像事件）。
EXPRESSION_STATE = re.compile(r'\b(?:crying|tears?|surprised|confused|blushing|scared|shocked|smiling|worried|expression)\b')
ONGOING_STATE = re.compile(r'\b(?:speaking|clapping|wiping|chewing|poured|pouring|spinning|winking|dissolving|looking)\b')


def annotation_change_states(story_id: str, output: dict, board: dict, existing: list[dict]) -> list[dict]:
    """规则 2 的标注变化补题，只返回新增题，编号接在 ``existing`` 最大 q 号之后：标注里持有、状态、服装的变化，变化那一镜
    还没有以该人／物（持有另算所持之物）为对象的镜末状态题、且请求在该镜提到它们的，在镜末补一题。放下（变化后不持有
    任何物）不出题；状态取值是面部表情或正在做的动作的不出，但照样占一个编号，使其余题的编号与源片复核时一致。"""
    validate(output, board)
    names = {n['id']: n['name'] for n in output['names']}
    requested = {shot['shot']: set(shot['present']) for shot in output['shots']}
    covered = {(q['shot'], e) for q in existing if q['dimension'] == 'state' and q['anchor'] == 'end'
               for e in re.findall(r'(?:Character|Object) \d+', q['question']) + q['depends_on']}
    tails = [q['requirement_id'].rsplit('/', 1)[1] for q in existing]
    number = max((int(tail[1:]) for tail in tails if tail.startswith('q')), default=0)
    rows: list[dict] = []
    for shot in board['shots']:
        k = shot['shot']
        for change in shot['changes']:
            entity, what, after = change['name'], change['what'], change['after']
            if what not in ANNOTATION_CHANGES or (what == 'holding' and not after):
                continue
            involved = [entity, *after] if what == 'holding' else [entity]
            if any((k, e) in covered for e in involved) or not set(involved) <= requested[k]:
                continue
            covered.update((k, e) for e in involved)
            number += 1
            if what == 'state' and (EXPRESSION_STATE.search(after) or ONGOING_STATE.search(after)):
                continue
            rows.append({'requirement_id': f'{story_id}/q{number:02d}', 'dimension': 'state', 'shot': k,
                         'reference_shot': None, 'anchor': 'end',
                         'question': _annotation_question(names[entity], what, after, names),
                         'depends_on': involved, 'family': 'annotation_change', 'source_change': what})
    return rows


# 第 6 版判官说明（10-29 判官校准选定）：跨镜比较比个体本身，不比是否符合描述。
INSTRUCTIONS = """Based on the video and the request above, answer each question.
- yes: you can clearly see it in the video.
- no: it does not happen, the person or object is not there, or it cannot be clearly seen.
An action counts only if you see it happen; seeing only its end result is not enough.
When a question compares a person, animal, object or place across two time ranges, compare the actual individual you see, not whether both match the request's description. The same person must have the same face and build; the same object must be the same individual object; the same place must be the same specific setting with the same layout and landmarks. Matching clothing, hair colour or a similar kind of place alone is not enough. If the visual style changes between the two time ranges (for example live action versus cartoon or 3D animation), it is not the same person, object or place. Clothing may change only where the request says so.
Give a short reason and the time (m:ss) where you saw it."""


def clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f'{seconds // 60}:{seconds % 60:02d}'


def judge_prompt(request: str, shot_seconds: list[float], rows: list[dict]) -> str:
    """第 6 版整片判官收到的文字（视频在它之前）：原请求、该视频各镜时间段、编号问题、判断说明。"""
    return judge_questions_text(request, shot_seconds, rows) + '\n\n' + INSTRUCTIONS


def judge_questions_text(request: str, shot_seconds: list[float], rows: list[dict]) -> str:
    """原请求、该视频各镜时间段与编号问题；题里的 ``{shot k}`` 换成该方法第 k 镜的实际时间段。"""
    edges = [0.0]
    for seconds in shot_seconds:
        edges.append(edges[-1] + seconds)
    window = {k: f'{clock(edges[k - 1])}\u2013{clock(edges[k])}' for k in range(1, len(shot_seconds) + 1)}
    lines = ['## Request', request, '', '## Shots in this video',
             ' · '.join(f'Shot {k}: {w}' for k, w in window.items()), '', '## Questions']
    for number, row in enumerate(rows, 1):
        text = row['question']
        for k, w in window.items():
            text = text.replace('{shot %d}' % k, w)
        prefix = '' if row['dimension'] in CROSS_SHOT + UNSCORED else \
            f"({'end of ' if row['anchor'] == 'end' else ''}{window[row['shot']]}) "
        if '{shot' in text:
            raise ValueError(f'unresolved shot placeholder: {row["requirement_id"]}')
        lines.append(f'{number}. {prefix}{text}')
    return '\n'.join(lines)


# 题目类别（题库里 `dimension` 的取值）与计分项分开：计分项「场景」只计地点题。
# 第 6 版另有两类不计入五项的题：style（整片画风是否一致，单列报告）与 monitor（按构造应答 no 的难反例，
# 判官答 yes 的比例即误判率，只用来监测判官）。layout 只在 11-03 前的题库里：读入、送判，不计分也不报。
# same_shot 是规则 4 的两镜前提题（期望答 no），由 checklist_scoring 计分；本模块的 story_scores 是第 6 版计分（第 7 版的底层答案与附录读数）。
QUESTION_DIMENSIONS = ('character', 'object', 'place', 'layout', 'event', 'state', 'style', 'monitor', 'same_shot')
UNSCORED = ('style', 'monitor')
# 跨镜比较题：题面自带 {shot j} 与 {shot i} 两个时间段，判官提示不再加单镜前缀；不计分的两类题题面自带范围，同样不加。
CROSS_SHOT = ('character', 'object', 'place', 'layout', 'same_shot')
# 计分项 → 计入它的题目类别。
SCORE_ITEMS = {'character': 'character', 'object': 'object', 'scene': 'place', 'event': 'event', 'state': 'state'}
SCORE_DIMENSIONS = tuple(SCORE_ITEMS)


def question_passes(rows: list[dict], answers: dict) -> dict:
    """计分题逐题是否通过（requirement_id → bool）：判官答 yes 即通过，不依赖别的题。

    只含五项计入的题；画风题、监测题与旧题库的布局题不在其中，所以全部适用题通过＝返回值全为 True。"""
    scored = set(SCORE_ITEMS.values())
    return {row['requirement_id']: answers[row['requirement_id']] == 'yes' for row in rows if row['dimension'] in scored}


def story_scores(rows: list[dict], answers: dict) -> dict:
    """一个故事的五项分，返回 ``{'scores': {计分项: 分}, 'auxiliary': {'style': 分, 'monitor_yes': 分}}``。

    每项＝计入它的那类题的通过比例（场景＝地点题）。
    任何一项没有题时为 None，不计入该项均值（第 6 版起物体无题不再记 1）。不计分的画风题与监测题放在 ``auxiliary``：画风为通过比例，监测为判官答 yes 的比例。
    `answers` 是 requirement_id → 'yes'／'no'。"""
    ok = question_passes(rows, answers)
    said_yes = {rid: answer == 'yes' for rid, answer in answers.items()}

    def rate(dimension, passed):
        values = [passed[row['requirement_id']] for row in rows if row['dimension'] == dimension]
        return sum(values) / len(values) if values else None

    return {'scores': {item: rate(dimension, ok) for item, dimension in SCORE_ITEMS.items()},
            'auxiliary': {'style': rate('style', said_yes), 'monitor_yes': rate('monitor', said_yes)}}


def generate_catalog(requests_path: Path, boards_path: Path, reuse_path: Path,
                     output: Path, workers: int) -> dict:
    """逐故事调用 Codex 抽取；原样复用冻结题库，原提示词、schema 和三条规则不变。

    每个未出题故事只调用一次；失败保留原答和退出码，不自动重试。模型工作目录只含
    本次文字输入，且使用只读沙箱，不向出题器提供源片或标注里的变化／状态字段。
    Codex保留本地线程，stderr中的session id用于真实中断后的人工恢复核查。
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from collections import Counter
    from datetime import datetime, timezone

    if workers < 1:
        raise ValueError('workers must be positive')
    requests = json.loads(requests_path.read_text())
    boards = {s['story_id']: s for s in json.loads(boards_path.read_text())['stories']}
    reuse = json.loads(reuse_path.read_text())['stories']
    stories = requests['stories']
    if len({s['story_id'] for s in stories}) != len(stories) or set(boards) != {s['story_id'] for s in stories}:
        raise ValueError('requests and boards must contain the same unique stories')
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / 'inputs'
    inputs.mkdir(exist_ok=True)
    for source, name in ((requests_path, 'requests.json'), (boards_path, 'storyboards.json'),
                         (reuse_path, 'reused-questions.json'), (PROMPT, 'prompt.md'), (SCHEMA, 'schema.json')):
        target = inputs / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f'frozen input differs: {target}')
        target.write_bytes(source.read_bytes())

    def extract(story):
        sid = story['story_id']
        board = boards[sid]['storyboard']
        request = story['inputs']['shots']['text']
        if request != boards[sid]['inputs']['shots']['text']:
            raise ValueError(f'common storyboard request differs: {sid}')
        if sid in reuse:
            frozen = reuse[sid]
            if frozen['request'] != request or frozen['split'] != story['split']:
                raise ValueError(f'frozen question request or split differs: {sid}')
            # 更正可能只删适用性错误的计分题，并保留原抽取；不得从原抽取复活已删题。
            rows = frozen['requirements']
            if len({q['requirement_id'] for q in rows}) != len(rows):
                raise ValueError(f'duplicate frozen requirement id: {sid}')
            print(json.dumps({'story_id': sid, 'origin': 'reused_catalog', 'questions': len(rows)}), flush=True)
            return sid, frozen
        work = output / 'writer' / sid
        work.mkdir(parents=True, exist_ok=True)
        prompt = writer_input(request, board)
        prompt_path = work / 'input.md'
        if prompt_path.exists() and prompt_path.read_text() != prompt:
            raise ValueError(f'writer input differs: {sid}')
        prompt_path.write_text(prompt)
        answer = work / 'output.json'
        receipt = work / 'receipt.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous['returncode'] != 0:
                raise ValueError(f'previous writer call failed, manual review required: {sid}')
        else:
            if answer.exists():
                raise ValueError(f'writer output without receipt: {sid}')
            command = ['codex', 'exec', '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort=high',
                       '--sandbox', 'read-only', '--skip-git-repo-check',
                       '--output-schema', str((inputs / 'schema.json').resolve()),
                       '--output-last-message', str(answer.resolve()), '-C', str(work.resolve()), '-']
            started = datetime.now(timezone.utc).isoformat()
            with (work / 'stdout.log').open('w') as stdout, (work / 'stderr.log').open('w') as stderr:
                result = subprocess.run(command, input=prompt, text=True, stdout=stdout, stderr=stderr)
            receipt.write_text(json.dumps({'command': command, 'started_at': started,
                'finished_at': datetime.now(timezone.utc).isoformat(), 'returncode': result.returncode}, indent=2))
            if result.returncode:
                raise ValueError(f'writer failed: {sid}; see {work}')
        if not answer.is_file():
            raise ValueError(f'writer returned without structured output: {sid}; see {receipt}')
        extracted = json.loads(answer.read_text())
        origin = 'codex_exec'
        rows = questions(sid, extracted, board)
        print(json.dumps({'story_id': sid, 'origin': origin, 'questions': len(rows)}), flush=True)
        return sid, {'split': story['split'], 'request': request, 'writer_output': extracted,
                     'requirements': rows, 'origin': origin}

    # 只领当前并发数；一条失败便停止领新故事，收齐已在途回执并交付成功部分。
    catalog, failures = {}, {}
    remaining = iter(stories)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        def submit_next():
            if (output / 'stop-new-stories').exists():
                return
            story = next(remaining, None)
            if story is not None:
                pending[pool.submit(extract, story)] = story['story_id']
        for _ in range(workers):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                sid = pending.pop(future)
                try:
                    _, item = future.result()
                    catalog[sid] = item
                except Exception as exc:
                    failures[sid] = {'type': type(exc).__name__, 'message': str(exc)}
            if not failures:
                for _ in done:
                    submit_next()
    catalog = {s['story_id']: catalog[s['story_id']] for s in stories if s['story_id'] in catalog}
    new_ids = {s['story_id'] for s in stories} - set(reuse)
    launched = {sid for sid in new_ids if (output / 'writer' / sid / 'receipt.json').exists()
                and json.loads((output / 'writer' / sid / 'receipt.json').read_text())['started_at'] is not None}
    successful = set(catalog) & new_ids
    result = {'benchmark': str(requests_path), 'storyboards': str(boards_path), 'request_tier': 'shots',
              'writer': {'model': 'gpt-6.1-sol', 'reasoning_effort': 'high', 'workers': workers,
                         'launched_calls': len(launched), 'successful_new_stories': len(successful),
                         'unsuccessful_new_stories': len(launched - successful),
                         'not_started_stories': len(new_ids - launched),
                         'reused_stories': len(set(catalog) & set(reuse))},
              'questions_from': 'mspc.evaluation.question_writer.questions', 'stories': catalog}
    partial = len(catalog) != len(stories)
    suffix = '.partial' if partial else ''
    (output / f'questions{suffix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    counts = {sid: {'split': s['split'], 'total': len(s['requirements']),
                   **{d: Counter(q['dimension'] for q in s['requirements'])[d] for d in QUESTION_DIMENSIONS}}
              for sid, s in catalog.items()}
    (output / f'counts{suffix}.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2) + '\n')
    if partial:
        (output / 'incomplete.json').write_text(json.dumps({'failures': failures,
            'not_started': [s['story_id'] for s in remaining], 'successful_stories': len(catalog)}, indent=2) + '\n')
        raise ValueError(f'writer incomplete; successful outputs retained at {output}/questions.partial.json')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='按现行三条规则为冻结请求出题，不调用付费判官。')
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--storyboards', type=Path, required=True)
    parser.add_argument('--reuse', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    generate_catalog(args.requests, args.storyboards, args.reuse, args.output, args.workers)
