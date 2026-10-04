import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "JayF14/chb-mit-preprocessed-eeg"

print(f"Creating dataset repo {repo_id}...")
try:
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"Repo creation error (may already exist): {e}")

print("Uploading processed data to the dataset repo...")
# Upload all files in the processed folder to the root of the dataset repo
api.upload_folder(
    folder_path="ml/data/processed",
    repo_id=repo_id,
    repo_type="dataset",
)

print("Upload completed successfully!")
