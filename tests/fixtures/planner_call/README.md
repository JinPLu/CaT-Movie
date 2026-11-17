# FX2 真实历史夹具

`real_case.json`与`real_history.jpg`取自主服务器已提交的200窗真实切片，case=`ci_vid:train:000000753/g2/synopsis`。剧本来自 `planner-zero-shot-and-data-20261007/inputs/planner-examples.jsonl` 首行；像素来自 `event-log-compiler-real-slice-20261007/review/frames/01_s01_last.jpg`，frame_ledger登记第一镜独立源文件index67。静态表/镜长来自同一行event_log。第一镜对齐帧数41，末帧对齐index40、15fps时间40/15；目标第二镜起点41/15。这是重采样故事轴时间，不冒充原视频PTS或源文件全局时间。对应source_slice metadata的source_frame_indices_by_shot映射仍留主服务器。夹具不包含实际生成历史，也不把真片复制一份冒充自产历史。

用途仅为HTTP图文组成、模式配方、时间泄漏与分母处理的回归；该反写样本及静态表未获正式语义准入，不能用于公平A2能力比较。完整200窗输入留原服务器；模型version/有效档synthetic transport测试明确标记，不是上游回执。

接续版本：gold_event_log、静态表与地点锚点使用同一case已入库的v2 `tests/fixtures/event_log_real_slice/04.json`；不把旧v1 events直接重命名为v2。旧反写脚本仅保留作为实际输入夹具，未重做语义准入。

`9b_real_span_failure.json` 是r2真实9B非thinking/script_only返回的原final（stop、1137输出tokens），来源`planner-v2-execution-20261007-r2/baseline/9b/nonthinking/script_only.responses.jsonl`。4个event.shot_span均为单元素；保留原字节字符串，验证修正后的唯一AtomicFact schema会直接拒绝这4个非法字段，而非让校验器索引崩溃。r2冻结schema原先只限制整数数组，历史评估不回写；后续请求的schema明确minItems=maxItems=2、单镜例[0,0]。它是实际模型接口失败回归，不是新像素/语义金标。

`27b_real_mm_cache_failure.json` 保留thinking接续真历史请求的实际HTTP500体与服务异常行：同服务先/tokenize只更新前端sender缓存，后续生成丢弃已缓存图像数据，而engine receiver没有对应条目。fixture回归保留真实失败体与完整输入；服务配置验收显式关闭该IPC processor缓存（`--mm-processor-cache-gb 0`）。CPU真实图像缓存复现及修后单项GPU回归另在对应运行目录记录，mock transport测试本身不证明服务已修好。

注：2026-10-09 事件日志 schema 升到 event_log_v3，本目录固件（含模型原始回复）里的 schema 字符串已按 v3 改写，非逐字原文；其余内容未动。

注：2026-10-09 规划请求改为 `planner_request` 唯一构造（不再把金标主体、地点与镜表放进输入），`real_case.json` 去掉了 subjects/locations/shots 三键、补上 `script_detail="synopsis"`（即原 case 的 g2/synopsis）；同时 gold_event_log 里 unknown 值去掉 evidence 字段，与已迁移的 `event_log_real_slice/04.json` 逐字段相同。
