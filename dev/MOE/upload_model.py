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


    # ------------------- Save and Upload Model to Hugging Face ----------------------
    # from upload_model import upload_model_to_hf
    # model_save_path = "hybridv3_weights.bin"
    # torch.save(final_model.state_dict(), model_save_path)
    
    # # Set your Hugging Face repository ID in the format "<username>/<repo-name>"
    # repo_id = "hzarashid/ForensiX"  # <-- CHANGE THIS to your repo id.
    
    # upload_model_to_hf(model_save_path, repo_id, commit_message="Upload trained model weights")
    # print(f"Model uploaded to Hugging Face repository: {repo_id}")