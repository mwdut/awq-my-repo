import os

from huggingface_hub import snapshot_download

hf_model = input("Insert Model Hub ID ")

snapshot_download(repo_id=hf_model, local_dir='./model-d')

print("Model downloaded successully!")
