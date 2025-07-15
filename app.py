import os, getpass, shutil
from huggingface_hub import login, create_repo, snapshot_download, upload_folder, HfApi
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

token = getpass.getpass("HF Write Token: ")
login(token=token)

hf_model = input("HF Model ID: ")

api = HfApi()
user = api.whoami()['name']
dest_repo = f"{user}/{hf_model.split('/')[-1]}-AWQ"
local_dir = hf_model.replace('/', '_')

create_repo(repo_id=dest_repo, exist_ok=True)
snapshot_download(repo_id=hf_model, local_dir=local_dir, allow_patterns=["README.md", "*model.safetensors", "*.json"] , ignore_patterns=["vocab.json"])

if not os.path.exists(os.path.join(local_dir, "model.safetensors")):
    print("Error: 'model.safetensors' not found")
    shutil.rmtree(local_dir, ignore_errors=True)
    exit(1)

print("Quantizing model...")
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(hf_model)
tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)

model.save_quantized(local_dir)
tokenizer.save_pretrained(local_dir)

print(f"Model quantized and saved to: {local_dir}")

upload_folder(
    folder_path=local_dir,
    repo_id=dest_repo
)

shutil.rmtree(local_dir, ignore_errors=True)
print(f"Upload to https://huggingface.co/{dest_repo}")
