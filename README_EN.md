# Project Overview
- LoRA-based DPO (Direct Preference Optimization) fine-tuning to fix repetitive-output “bad cases”.
- Each `badcaseX/` folder is self-contained with core data, a training script, and an inference script.
- Framework requirement: training depends on the LLaMA-Factory project (`../../../src/llamafactory/launcher.py`); clone and install that framework before running any scripts.

# Layout
- `badcase1|2|3/`
  - `dpo fine-tuning script.sh`: launches LLaMA-Factory with Deepspeed (LoRA + DPO).
  - `dataset_info.json`: dataset registry; `DATASET=generalized_dpo_preference_dataset`.
  - `generalized_dpo_preference_dataset.json`: preference samples (fields: `instruction`, `input`, `chosen`, `rejected`; `badcase3` file is a single sample—actual training should use the path in `dataset_info.json`).
  - `test.json`: inference test set (`text`, `standard_answer`) to replay bad cases after fine-tuning.
  - `Inference.py`: vLLM-based inference exporting results to Excel.
- `deepspeed/qwen_ds_config_zero3.json`: ZeRO-3 config (fp16/bf16 auto, AdamW, Warmup LR, etc.).
- `Overview-DPO fine-tuning solves the problem of repeaters.xlsx`: overview notes (not executed by scripts).

# Requirements
- OS: Linux (Bash; Deepspeed command assumes 8 GPUs `localhost:0-7`).
- Python: 3.10+ recommended.
- Training deps: `torch` (bf16/AMP), `deepspeed`, `transformers`, `peft`, `accelerate`, `wandb` (optional), LLaMA-Factory (`../../../src/llamafactory/launcher.py` is invoked by the scripts).
- Inference deps: `vllm` (with LoRA), `transformers`, `pandas`, `uuid`.
- Hardware: multi-GPU; writable `/data/...` paths or adjust to your environment.

# Data
- Preference data (DPO) mapping in `dataset_info.json`:
  - `prompt -> instruction`, `query -> input`, `chosen -> chosen`, `rejected -> rejected`
  - `file_name` defaults to `/data/llm/LLaMA-Factory-24090501/sun/data/generalized_dpo_preference_dataset.json`; update to your local dataset path if needed.
- Test set: `test.json` is an array of `{ "text": ..., "standard_answer": ... }`, used to replay bad cases post-DPO.

# Training (DPO fine-tuning)
1. Enter a case folder (example: badcase1):
   - `cd badcase1`
2. Review and adjust environment variables in the script:
   - Model & output: `MODEL_BASE_PATH`, `MODEL_NAME`, `OUTPUT_DIR`
   - Data: `DATASET_DIR`, `DATASET` (must match the key in `dataset_info.json`), `DS_CONFIG` (default `../../deepspeed/qwen_ds_config_zero3.json`)
   - Hyperparameters: `CUTOFF_LEN=4096`, batch size 1, `LORA_RANK=8`, `LORA_ALPHA=16`, `LORA_DROPOUT=0.05`, `LEARNING_RATE=1e-4`, `NUM_TRAIN_EPOCHS=1`, `VAL_SIZE=0.05`
   - Misc: `WANDB_PROJECT`/`WANDB_LOG_MODEL`, `CUTLASS_PATH`, optional `RESUME_FROM_CHECKPOINT`
3. Run training:
   - `bash "dpo fine-tuning script.sh"`
4. Outputs: LoRA adapters saved to `OUTPUT_DIR`; timing stats written to `time_consuming_statistics.txt`.
5. badcase2 and badcase3 follow the same flow; main differences are `MODEL_NAME`, `VERSION`, and data samples.

# Inference
- Check and adjust paths in `Inference.py`:
  - `input_texts`: default `/data/llm/axolotl/xxx/inference/test.json`; you can point it to `./test.json` in each folder.
  - `model_paths`: list of LoRA adapter folders, e.g., `/data/sft/lora/b16s1/saves/Qwen2-72B-Instruct/v4.0`
  - `llm_model_path`: base model, e.g., `/data/models/Qwen/Qwen2-72B-Instruct`
  - Set `CUDA_VISIBLE_DEVICES` if you need specific GPUs.
- Sampling: `temperature=0`, `top_p=1`, `top_k=-1`, `max_tokens=4096`, `use_beam_search=True`, `best_of=5`.
- Result: Excel with `adapter_name_or_path`, `text`, `response`, `standard_answer`; default path `/data/llm/axolotl/xxx/inference/output/inference_result.xlsx` (filename uses fixed `result` suffix).

# Tunable knobs
- GPUs & parallelism: edit the `deepspeed -i` device list and `tensor_parallel_size`.
- Training length & LR: `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `WARMUP_RATIO`, `WEIGHT_DECAY`.
- Sequence length & sampling: `CUTOFF_LEN`, `MAX_SAMPLES`, `VAL_SIZE`.
- Vocab resize: enable `RESIZE_VOCAB` to add `--additional_target embed_tokens,lm_head`.
- Resume: set `RESUME_FROM_CHECKPOINT` to a prior checkpoint path.

# Quick Start
```
cd badcase1
# Ensure the LLaMA-Factory project is downloaded and deployed correctly, and the data and model paths are configured before execution.
bash "dpo fine-tuning script.sh"

# For inference, point test.json inside Inference.py to the current folder
python Inference.py
```

# Validation
- Use training logs and (optional) W&B for loss/metric tracking.
- Compare `response` vs `standard_answer` in the Excel output to confirm bad cases are resolved; spot-check manually if needed.
