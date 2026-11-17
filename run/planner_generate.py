#!/usr/bin/env python3
"""规划器生成（占 GPU，单进程单卡）：基座或 LoRA checkpoint 对规划器样本逐条生成分镜表。

请求文字由 ``planner_dataset.planner_messages`` 构造，与训练编码同一处；生成用非思考模板。
输出 jsonl 每行 ``case_id``（=样本号）、``raw_content``、``status``（以停止符收尾为 completed，撞上限为 incomplete）、
``finish_reason``（stop／length）。停止符是模型的 eos 加聊天回合结束符 ``<|im_end|>``（见 ``inference/planner_decode``）。
评测入口 ``run/planner_eval.py`` 读它。输出文件已存在时续跑：跳过已写的样本。
同一张卡上可以并行起几个进程分片生成（``--shard k/n``：第 i 条样本归 i % n == k 的进程），各写各的输出文件，
``--done-from`` 列出别处已经写好的答复文件，这些样本也跳过；全部写完后把各文件拼成一份交给评测入口。
采样参数必须显式给（``--temperature 0`` 为贪心）；HF generate 没有 presence_penalty，不传。
"""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA checkpoint 目录；不给就是基座")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="只生成前 N 条（链路验证用）")
    parser.add_argument("--shard", default="0/1", help="k/n：只处理第 i 条满足 i %% n == k 的样本")
    parser.add_argument("--done-from", type=Path, nargs="*", default=[], help="别处已写好的答复 jsonl，其中的样本跳过")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from mspc.data.planner_dataset import (EMPTY_THINK, PLANNER_CONTRACT_VERSION, canonicalize_script,
                                         planner_messages, validate_planner_example)
    from mspc.inference.planner_decode import answer_row, stop_token_ids
    from mspc.evaluation.storyboard_planner import assess_answer

    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        samples = samples[:args.limit]
    for sample in samples:
        validate_planner_example(sample)
    shard, shards = map(int, args.shard.split("/"))
    if not 0 <= shard < shards:
        raise ValueError("--shard must be k/n with 0 <= k < n")
    done = set()
    for path in (args.output, *args.done_from):
        if path.exists():
            done |= {json.loads(line)["case_id"] for line in path.read_text().splitlines() if line.strip()}
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to("cuda")
    if args.adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    stop_ids = stop_token_ids(processor.tokenizer, model.generation_config)
    sampling = ({"do_sample": False} if args.temperature == 0 else
                {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        for index, sample in enumerate(samples):
            if sample["sample_id"] in done or index % shards != shard:
                continue
            messages = planner_messages(sample)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if not prompt.endswith(EMPTY_THINK):
                raise ValueError("non-thinking generation prompt must end with the empty think block")
            inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
            torch.manual_seed(args.seed + index)
            started = time.monotonic()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, eos_token_id=stop_ids,
                                        pad_token_id=processor.tokenizer.pad_token_id, **sampling)
            new = output[0, inputs["input_ids"].shape[1]:].tolist()
            row = answer_row(sample["sample_id"], processor.tokenizer.decode(new, skip_special_tokens=True), new, stop_ids,
                             max_new_tokens=args.max_new_tokens, prompt_tokens=int(inputs["input_ids"].shape[1]),
                             elapsed_seconds=time.monotonic() - started)
            row.update(contract_version=PLANNER_CONTRACT_VERSION, request=messages,
                       input_name_mapping=canonicalize_script(sample['script'])[1])
            row.update(assess_answer(row, timing=None))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
