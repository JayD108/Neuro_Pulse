import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = "JayF14/chb-mit-preprocessed-eeg"

print(f"Uploading README.md to the dataset repo {repo_id}...")

api.upload_file(
    path_or_fileobj="dataset_README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
)

print("README.md uploaded successfully!")
