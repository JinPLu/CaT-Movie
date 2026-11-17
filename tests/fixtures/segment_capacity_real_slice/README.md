# 段容量真实边界夹具

来源：主服务器 `Storyboard/outputs/experiments/renderer-capacity-20261007/real_slice-rerun3/event_logs.jsonl`，同一冻结 200 行源切片中已接受的两条窗。`rows.json` 原样摘录事件日志，不改镜长或文字（2026-10-09 删去已废的 style/segments/cut_type 键、旧镜级 frame_evidence 与 anchor_sources；测试里的逐帧证据由 `renderer_segment_preflight.evidence_rows` 生成）。

- `ci_vid:train:000028799`：[153,153]，306 帧 2 镜；旧 309/5 一段，新 228/4 拆成两个完整单镜段。
- `ci_vid:train:000063854`：[49,21,25,21,21]，137 帧 5 镜；旧 309/5 一段，新 228/4 在第 4 镜后切，保留短尾段。

latents 由测试使用可逐位检查的张量替身；这不代表像素、生成或标注质量验收。

注：2026-10-09 事件日志 schema 升到 event_log_v3，本目录固件（含模型原始回复）里的 schema 字符串已按 v3 改写，非逐字原文；其余内容未动。
