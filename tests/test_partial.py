from datasets import load_dataset
from specpv import Speculator, SpecConfig
import torch

# Prepare dataset
dataset = load_dataset("parquet", data_files="data/pg19_test/pg19_test_30k.parquet")['train']
text = dataset[0]['text']

base_model_path = '/home/lab6033/zhendong/models/LLAMA3.1-8B-Instruct' 
EAGLE_model_path = "/home/lab6033/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K"
# load model
model = Speculator.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()

tokenizer = model.tokenizer
system_prompt = (
    "You are a creative writing assistant. Continue the following story "
    "in a coherent, engaging, and stylistically consistent way."
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": text}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_inputs = tokenizer([text], return_tensors="pt").to(device)
input_ids = model_inputs["input_ids"]


def generate_and_print(label, spec_config):
    output_ids, metrics = model.spec_generate(
        input_ids,
        temperature=0,
        max_new_tokens=512,
        max_length=35000,
        log=True,
        is_llama3=True,
        spec_config=spec_config,
    )
    output = model.tokenizer.decode(
        output_ids[0][input_ids.shape[-1] :],
        skip_special_tokens=True,
    )
    print(f"\n{'=' * 24} {label} {'=' * 24}")
    print(output)
    print(metrics)


# Use the same prompt and deterministic decoding for both runs.  The full run
# is deliberately first so its text is the reference shown to the user.
full_config = SpecConfig(enable_offload=False, enable_partial_kv=False)
generate_and_print("FULL KV", full_config)

# spec_generate caches its KV-cache allocation on the model. Force a fresh
# allocation so this run uses the partial-cache configuration below rather than
# the cache created for the full-KV run above.
model.max_length = 0
partial_config = SpecConfig(
    enable_offload=False,
    enable_partial_kv=True,
    n_retrieval_blocks=512,
    partial_spec_tokens=20,
)
generate_and_print("PARTIAL KV", partial_config)
