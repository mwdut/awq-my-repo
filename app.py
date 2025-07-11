import os, getpass
from huggingface_hub import login, create_repo, snapshot_download, upload_folder, HfApi

token = getpass.getpass("HF Write Token: ")
login(token=token)

hf_model = input("HF Model ID: ")

api = HfApi()
user = api.whoami()['name']
dest_repo = f"{user}/{hf_model.split('/')[-1]}-GPTQ"
local_dir = hf_model.replace('/', '_')

create_repo(repo_id=dest_repo, exist_ok=True)

snapshot_download(repo_id=hf_model, local_dir=local_dir, allow_patterns=["*.md", "*model.safetensors", "tokenizer.json , config.json , generation_config.json"])

upload_folder(
    folder_path=local_dir,
    repo_id=dest_repo
)

shutil.rmtree(local_dir, ignore_errors=True)

print(f": Upolad to  https://huggingface.co/{dest_repo}")
