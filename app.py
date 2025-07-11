import os, getpass
from huggingface_hub import login, create_repo, snapshot_download, upload_folder, HfApi

token = getpass.getpass("HF Write Token: ")
login(token=token)
hf_model = input("HF Model ID: ")

api = HfApi()
user = api.whoami()['name']


repo_name = f"{user}/{hf_model.split('/')[-1]}-GPTQ"
local_dir = hf_model.replace('/', '_')

snapshot_download(repo_id=hf_model, local_dir=local_dir, allow_patterns=["*.md", "*model.safetensors", "tokenizer.json , config.json , generation_config.json"])


create_repo(repo_id=repo_name, exist_ok=True)
upload_folder(
    folder_path=local_dir,
    repo_id=repo_name
)

shutil.rmtree(local_dir, ignore_errors=True)

print(f": Upolad to  https://huggingface.co/{repo_name}")
