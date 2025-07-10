import os, getpass
from huggingface_hub import login, create_repo, snapshot_download, upload_file, HfApi

token = getpass.getpass("HF Write Token: ")
login(token=token)

src_repo = input("HF Model ID: ")

api = HfApi()
user = api.whoami()['name']
dest_repo = f"{user}/{src_repo.split('/')[-1]}-GPTQ"
local_dir = src_repo.replace('/', '_')

create_repo(repo_id=dest_repo, exist_ok=True)

snapshot_download(repo_id=src_repo, local_dir=local_dir, allow_patterns=["*.md", "*model.safetensors", "tokenizer.json , config.json , generation_config.json"])

for f in os.listdir(local_dir):
    path_model = os.path.join(local_dir, f)
    if os.path.isfile(path_model):
        upload_file(
            path_or_fileobj=path_model,
            path_in_repo=f,
            repo_id=dest_repo
        )

print(f"Sucessfull!!!: Upolad to  https://huggingface.co/{dest_repo}")
