# 项目简介
- 基于 LoRA 的 DPO（Direct Preference Optimization）微调流程，针对重复输出的“badcase”进行修复。
- 每个 `badcaseX/` 目录都包含独立的核心数据、训练脚本与推理脚本。
- 框架依赖：训练脚本依赖 LLaMA-Factory 项目（调用 `../../../src/llamafactory/launcher.py`），请先克隆并安装后再执行。

# 目录结构
- `badcase1|2|3/`
  - `dpo fine-tuning script.sh`：调用 LLaMA-Factory 的 Deepspeed 启动脚本（LoRA+DPO）。
  - `dataset_info.json`：数据集注册信息，`DATASET=generalized_dpo_preference_dataset`。
  - `generalized_dpo_preference_dataset.json`：偏好数据示例（字段：`instruction`、`input`、`chosen`、`rejected`；badcase3 本地文件为单条示例，实际训练使用 `dataset_info.json` 指向的路径）。
  - `test.json`：推理测试集（含 `text` 与 `standard_answer`）。
  - `Inference.py`：加载 LoRA 适配器的推理与结果导出脚本。
- `deepspeed/qwen_ds_config_zero3.json`：ZeRO-3 训练配置（fp16/bf16 auto、AdamW、warmup LR 等）。
- `Overview-DPO fine-tuning solves the problem of repeaters.xlsx`：概览说明（不参与脚本执行）。

# 环境依赖
- 系统：Linux（脚本使用 Bash，Deepspeed 指定 8 卡 `localhost:0-7`）。
- Python：建议 3.10+。
- 训练依赖：`torch`（支持 bf16/AMP）、`deepspeed`、`transformers`、`peft`、`accelerate`、`wandb`（可选日志）、LLaMA-Factory（脚本从 `../../../src/llamafactory/launcher.py` 启动）。
- 推理依赖：`vllm`（含 LoRA 支持）、`transformers`、`pandas`、`uuid`。
- 硬件：多 GPU（脚本默认 8 卡 Tensor Parallel），磁盘需可写 `/data/...` 相关路径或自行修改。

# 数据说明
- 偏好数据（DPO）：字段映射在 `dataset_info.json` 内
  - `prompt -> instruction`，`query -> input`，`chosen -> chosen`，`rejected -> rejected`
  - `file_name` 默认指向 `/data/llm/LLaMA-Factory-24090501/sun/data/generalized_dpo_preference_dataset.json`，请确保该路径存在或修改为本地数据。
- 测试集：`test.json` 为数组，每项包含待测 `text` 与期望输出 `standard_answer`，用于微调后回放 badcase。

# 训练（DPO 微调）
1. 进入对应目录（示例以 badcase1）：
   - `cd badcase1`
2. 核查并按需修改脚本中的环境变量：
   - 模型与输出：`MODEL_BASE_PATH`、`MODEL_NAME`、`OUTPUT_DIR`
   - 数据：`DATASET_DIR`、`DATASET`（需与 `dataset_info.json` 的键一致）、`DS_CONFIG`（默认 `../../deepspeed/qwen_ds_config_zero3.json`）
   - 资源与超参：`CUTOFF_LEN=4096`，batch 均为 1，`LORA_RANK=8`、`LORA_ALPHA=16`、`LORA_DROPOUT=0.05`，`LEARNING_RATE=1e-4`，`NUM_TRAIN_EPOCHS=1`，`VAL_SIZE=0.05`
   - 其他：`WANDB_PROJECT`/`WANDB_LOG_MODEL`、`CUTLASS_PATH`（如需）、是否 `RESUME_FROM_CHECKPOINT`
3. 运行训练：
   - `bash "dpo fine-tuning script.sh"`
4. 输出：LoRA 适配器保存至 `OUTPUT_DIR`，并生成 `time_consuming_statistics.txt` 记录起止时间。
5. badcase2、badcase3 流程一致，主要差异在 `MODEL_NAME`、`VERSION` 与数据样例。

# 推理验证
- 核查 `Inference.py` 中的路径并按需调整：
  - `input_texts`：默认读取 `/data/llm/axolotl/xxx/inference/test.json`，可改为本目录的 `./test.json`
  - `model_paths`：LoRA 适配器路径列表，例如 `/data/sft/lora/b16s1/saves/Qwen2-72B-Instruct/v4.0`
  - `llm_model_path`：基础模型路径（如 `/data/models/Qwen/Qwen2-72B-Instruct`）
  - GPU 选择可设置 `CUDA_VISIBLE_DEVICES`
- 采样配置：`temperature=0`、`top_p=1`、`top_k=-1`、`max_tokens=4096`、`use_beam_search=True`、`best_of=5`
- 结果：将 `adapter_name_or_path`、`text`、`response`、`standard_answer` 保存到 Excel，默认输出 `/data/llm/axolotl/xxx/inference/output/inference_result.xlsx`（文件名中 version 被固定为 `result`）。

# 常见可调整项
- GPU 与并行：修改 `deepspeed -i` 的设备列表与 `tensor_parallel_size`。
- 训练时长与学习率：`NUM_TRAIN_EPOCHS`、`LEARNING_RATE`、`WARMUP_RATIO`、`WEIGHT_DECAY`。
- 序列长度与样本量：`CUTOFF_LEN`、`MAX_SAMPLES`、`VAL_SIZE`。
- 词表扩展：开启 `RESIZE_VOCAB` 后需追加 `--additional_target embed_tokens,lm_head`。
- 恢复训练：设置 `RESUME_FROM_CHECKPOINT` 为检查点路径。

# 快速上手（示例）
```
cd badcase1
# 确保下载并部署好LLaMA-Factory项目、配置好数据路径与模型路径后执行
bash "dpo fine-tuning script.sh"

# 推理，先将 Inference.py 中的 test.json 指向本目录
python Inference.py
```

# 结果检查
- 训练日志与 W&B（如开启）可用于对比 loss 与 DPO 指标。
- 推理 Excel 对比 `response` 与 `standard_answer` 以验证 badcase 是否消失。可按样本 ID 或文本人工抽检。
