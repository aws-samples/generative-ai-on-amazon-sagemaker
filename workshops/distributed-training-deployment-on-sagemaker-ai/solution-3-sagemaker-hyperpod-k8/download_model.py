from huggingface_hub import snapshot_download
import os


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def download_model(path, model_name):
    print("Downloading model ", model_name)

    os.makedirs(path, exist_ok=True)

    snapshot_download(repo_id=model_name, local_dir=path)

    print(f"Model {model_name} downloaded under {path}")


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_dir = f"/mnt/custom-file-systems/{'/'.join(script_dir.split('/')[4:])}"

    download_model(f"{script_dir}/{MODEL_ID.split('/')[-1]}", MODEL_ID)
