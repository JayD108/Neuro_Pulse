import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "JayF14/Neuro_Pulse"

print(f"Creating repo {repo_id}...")
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Repo creation error: {e}")

print("Uploading README.md...")
api.upload_file(
    path_or_fileobj="HF_README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)

print("Uploading trained models...")
api.upload_folder(
    folder_path="models/trained",
    path_in_repo="models/trained",
    repo_id=repo_id,
    repo_type="model",
)

print("Uploading processed data...")
api.upload_folder(
    folder_path="ml/data/processed",
    path_in_repo="ml/data/processed",
    repo_id=repo_id,
    repo_type="model",
)

print("Upload completed successfully!")
