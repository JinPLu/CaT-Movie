# IC-LoRA tokenizer dependency

IC-LoRA runs in the StoryMem environment (torch 2.4.0, diffusers 0.32.2, transformers 4.45.2, peft 0.14.0), which can
import the FLUX pipeline but lacks SentencePiece, needed by FLUX's T5 tokenizer. This project locks only
`sentencepiece==0.2.1` (the version in the main `uv.lock`) and installs it into a separate directory that goes first on
the `PYTHONPATH` of the IC-LoRA process only; other methods never read it.

```bash
uv export --frozen --project src/mspc/comparison_methods/environments/ic_lora --no-dev --no-emit-project \
  --output-file src/mspc/comparison_methods/environments/ic_lora/requirements.txt
uv pip install --python "$MSPC_SHARED_ROOT/env/storymem_inference_sm80/bin/python" --no-deps --require-hashes \
  --target "$MSPC_SHARED_ROOT/env/ic_lora_dependencies" -r src/mspc/comparison_methods/environments/ic_lora/requirements.txt
```
