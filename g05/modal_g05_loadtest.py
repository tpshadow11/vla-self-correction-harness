import modal

app = modal.App("g05-loadtest")

g05_ckpt = modal.Volume.from_name("g05-checkpoints", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "build-essential", "clang", "cmake", "ffmpeg")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .run_commands("git clone https://github.com/OpenGalaxea/GalaxeaVLA.git /root/GalaxeaVLA")
    .run_commands("sed -i 's/>=3.10.16/>=3.10.13/' /root/GalaxeaVLA/pyproject.toml")
    .run_commands("cd /root/GalaxeaVLA && pip install -e .")
    .pip_install("flash-attn-4>=4.0.0b15", extra_options="--no-build-isolation")
    .pip_install("huggingface_hub")
    .env({"HF_HOME": "/root/.cache/huggingface"})
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("g05-hf-token")],
    volumes={"/root/GalaxeaVLA/checkpoints": g05_ckpt},
)
def download_and_load():
    import os
    import subprocess
    from huggingface_hub import hf_hub_download

    files = [
        "g05-libero/model.pt",
        "g05-libero/.hydra/config.yaml",
        "g05-libero/dataset_stats.json",
        "action_tokenizer.pt",
    ]
    processor_files = [
        "qwen3_5_2b_base_processor/config.json",
        "qwen3_5_2b_base_processor/configuration.json",
        "qwen3_5_2b_base_processor/merges.txt",
        "qwen3_5_2b_base_processor/preprocessor_config.json",
        "qwen3_5_2b_base_processor/tokenizer.json",
        "qwen3_5_2b_base_processor/tokenizer_config.json",
        "qwen3_5_2b_base_processor/video_preprocessor_config.json",
        "qwen3_5_2b_base_processor/vocab.json",
    ]
    ckpt_dir = "/root/GalaxeaVLA/checkpoints"
    for f in files:
        dest = os.path.join(ckpt_dir, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            path = hf_hub_download(
                "OpenGalaxea/G05", f, token=os.environ["HF_TOKEN"],
                local_dir=ckpt_dir,
            )
            print(f"[downloaded] {f} -> {path}")
        else:
            print(f"[cached] {f}")

    import shutil as _shutil
    _shutil.copy(
        os.path.join(ckpt_dir, "action_tokenizer.pt"),
        os.path.join(ckpt_dir, "g05-libero", "action_tokenizer.pt"),
    )
    print("[copied] action_tokenizer.pt -> g05-libero/action_tokenizer.pt")

    hf_processor_dir = os.path.join(ckpt_dir, "g05-libero", "hf_processor")
    os.makedirs(hf_processor_dir, exist_ok=True)
    for f in processor_files:
        fname = os.path.basename(f)
        dest = os.path.join(hf_processor_dir, fname)
        if not os.path.exists(dest):
            path = hf_hub_download(
                "OpenGalaxea/G05", f, token=os.environ["HF_TOKEN"],
                local_dir="/tmp/qwen_dl",
            )
            import shutil
            shutil.copy(path, dest)
            print(f"[downloaded->hf_processor] {fname}")
        else:
            print(f"[cached] hf_processor/{fname}")
    g05_ckpt.commit()

    print("[*] attempting model load via g05.models.g05.inferencer.PolicyInferencer path...")
    os.chdir("/root/GalaxeaVLA")
    result = subprocess.run(
        [
            "python", "-c",
            (
                "import torch;"
                "from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir;"
                "from scripts.serve_policy import setup;"
                "run_dir = find_run_dir('checkpoints/g05-libero/model.pt');"
                "cfg = load_config_from_run_dir(run_dir, 'checkpoints/g05-libero/model.pt', []);"
                "policy, processor = setup(cfg, device='cuda');"
                "print('[SUCCESS] model loaded:', type(policy).__name__)"
            ),
        ],
        capture_output=True, text=True, timeout=600,
    )
    print("STDOUT:\n", result.stdout[-4000:])
    print("STDERR:\n", result.stderr[-4000:])
    print("returncode:", result.returncode)


@app.local_entrypoint()
def main():
    download_and_load.remote()
