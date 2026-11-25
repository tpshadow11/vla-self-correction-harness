import modal

OPENVLA = "/root/openvla"
DATA_DIR = "/data"
RUN_DIR = "/runs"

app = modal.App("openvla-reflect-finetune")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
rlds_vol = modal.Volume.from_name("rlds-reflect-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("reflect-finetune-runs", create_if_missing=True)

IGNORE = ["**/.git", "**/__pycache__", "**/*.pyc", "**/.DS_Store",
          "**/wandb", "**/rollouts", "**/experiments/logs"]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libgl1-mesa-glx", "libegl1", "libosmesa6",
        "libglew-dev", "libglfw3", "libglib2.0-0", "ffmpeg",
        "build-essential", "clang", "cmake",
    )
    .pip_install("torch==2.2.0", "torchvision==0.17.0", "packaging", "ninja", "wheel", "setuptools")
    .pip_install("flash-attn==2.5.5", extra_options="--no-build-isolation")
    .add_local_dir("openvla", OPENVLA, copy=True, ignore=IGNORE)
    .run_commands(f"pip install -e {OPENVLA}")
    .run_commands(
        "pip install --no-deps 'tensorflow-metadata==1.15.0'",
        "pip install 'numpy<2'",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "WANDB_MODE": "disabled"})
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 6,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        DATA_DIR: rlds_vol,
        RUN_DIR: runs_vol,
    },
)
def run_finetune(max_steps: int = 2000, batch_size: int = 8):
    import os
    import subprocess

    os.chdir(OPENVLA)
    cmd = [
        "torchrun", "--standalone", "--nnodes", "1", "--nproc-per-node", "1",
        "vla-scripts/finetune.py",
        "--vla_path", "openvla/openvla-7b-finetuned-libero-spatial",
        "--data_root_dir", DATA_DIR,
        "--dataset_name", "libero_spatial_reflect_mix",
        "--run_root_dir", RUN_DIR,
        "--adapter_tmp_dir", f"{RUN_DIR}/adapter-tmp",
        "--batch_size", str(batch_size),
        "--grad_accumulation_steps", "1",
        "--max_steps", str(max_steps),
        "--save_steps", str(max_steps),  # only save at the end, single checkpoint
        "--learning_rate", "5e-4",
        "--run_id_note", "erp-reflect-v1",
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print("returncode:", proc.returncode)
    runs_vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"finetune.py failed with code {proc.returncode}")


@app.local_entrypoint()
def main(max_steps: int = 2000, batch_size: int = 8):
    run_finetune.remote(max_steps=max_steps, batch_size=batch_size)
