def upload_model_to_hf(model_path, repo_id, commit_message="Upload trained model weights"):
    """
    Uploads the given model file to the Hugging Face Hub using HfApi.upload_file.

    Args:
        model_path (str): Path to the saved model file.
        repo_id (str): Repository ID in the format "<username>/<repo-name>".
        commit_message (str): Commit message for the upload.
    """
    import os
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        commit_message=commit_message
    )