from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import os, json
import pandas as pd
import uuid

# device = "cuda"  # the device to load the model onto
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# Load input texts
input_texts = []
with open("/data/llm/axolotl/xxx/inference/test.json", "r") as fp:
    input_texts = json.load(fp)

results = []
# Model paths
model_paths = [
    "/data/sft/lora/b16s1/saves/Qwen2-72B-Instruct/v4.0",
]

llm_model_path = "/data/models/Qwen/Qwen2-72B-Instruct"

# Load the vllm model
llm = LLM(llm_model_path, enable_lora=True, tensor_parallel_size=4,
          gpu_memory_utilization=0.9)  # adjust tensor parallel size as needed
tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

# Iterate over each model path
for adapter_name_or_path in model_paths:
    # 使用 uuid 生成全局唯一的 int 类型 id 和 custom_adapter
    unique_id = hash(uuid.uuid4())  # 使用 hash 函数将 UUID 转换为整数
    unique_adapter_name = f"custom_adapter_{unique_id}"  # 生成唯一的adapter名称

    # 加载 LoRA 适配器
    lora_request = LoRARequest(unique_adapter_name, unique_id, adapter_name_or_path)

    print(f"Loaded model: {adapter_name_or_path}, id: {unique_id}, adapter: {unique_adapter_name}")

    # Iterate over each input text
    for i, input_data in enumerate(input_texts):
        print(f"====================================================")
        print(f"Model: {adapter_name_or_path}, i: {i}")
        input_text = input_data["text"]
        standard_answer = input_data["standard_answer"]

        # Prepare the messages for the chat template
        messages = [
            {"role": "user", "content": input_text}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the input text
        model_inputs = tokenizer([text], return_tensors="pt")

        # Set sampling parameters for vllm
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            max_tokens=4096,
            use_beam_search=True,
            best_of=5
        )
        # sampling_params = SamplingParams(
        #     temperature=0,
        #     presence_penalty=1.2,
        #     max_tokens=4096
        # )
        # sampling_params = SamplingParams(
        #     temperature=0,
        #     max_tokens=4096
        # )

        # Perform inference using vllm
        outputs = llm.generate(text, sampling_params, lora_request=lora_request)
        response = outputs[0].outputs[0].text

        print(f"Model: {adapter_name_or_path}, i: {i}, len(response): {len(response)}")

        # Append the result to the list
        results.append([os.path.basename(adapter_name_or_path),
                        input_text, response, standard_answer])

        # Create a pandas DataFrame from the results
        df = pd.DataFrame(results, columns=[
            "adapter_name_or_path", "text", "response", "standard_answer"])

        # Extract version number from adapter path
        version_number = 'result'

        # Save the DataFrame to an Excel file with the version number in the filename
        output_file = f"/data/llm/axolotl/xxx/inference/output/inference_{version_number}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")