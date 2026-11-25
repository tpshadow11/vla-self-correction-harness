import modal

LIBERO = "/root/LIBERO"
GALAXEA = "/root/GalaxeaVLA"

app = modal.App("g05-libero-eval")

g05_ckpt = modal.Volume.from_name("g05-checkpoints", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "build-essential", "clang", "cmake", "ffmpeg", "iproute2",
        "libgl1-mesa-glx", "libegl1", "libosmesa6",
        "libglew-dev", "libglfw3", "libglib2.0-0",
    )
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
    # Ship the same local LIBERO source already proven working in the
    # OpenVLA pipeline (modal_libero_eval.py), rather than re-cloning a
    # possibly different version from GitHub.
    .add_local_dir("LIBERO", LIBERO, copy=True,
                    ignore=["**/.git", "**/__pycache__", "**/*.pyc"])
    .run_commands(f"pip install -e {LIBERO}")
    # LIBERO's benchmark loader predates PyTorch 2.6's weights_only=True
    # default; G05 needs torch 2.7.1 (unlike the OpenVLA pipeline's torch
    # 2.2.0, which still had the old default and never hit this). We trust
    # LIBERO's own init-state files (numpy arrays, not arbitrary code), so
    # explicitly opt back in to the pre-2.6 behavior for this one call site.
    .run_commands(
        "sed -i "
        "'s/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' "
        f"{LIBERO}/libero/libero/benchmark/__init__.py"
    )
    # NOTE: unlike the OpenVLA pipeline, we do NOT downgrade mujoco here --
    # G05's own pyproject.toml pins mujoco==3.3.2 with robosuite==1.4.0
    # (not 1.4.1), a combination they presumably test against; the OpenVLA
    # pipeline's mujoco==2.3.7 downgrade was specifically to work around
    # robosuite==1.4.1's old mj_fullM signature, which doesn't apply here.
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HOME": "/root/.cache/huggingface"})
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("g05-hf-token")],
    volumes={f"{GALAXEA}/checkpoints": g05_ckpt},
)
def run_libero_eval_smoke():
    import os
    import subprocess

    os.chdir(GALAXEA)
    result = subprocess.run(
        [
            "bash", "scripts/run/eval_libero.sh",
            "checkpoints/g05-libero/model.pt",
            "--suites", "libero_spatial",
            "--num_trials", "5",
            "--num_parallel", "5",
            "--output_dir", "/tmp/g05_libero_smoke",
        ],
        capture_output=True, text=True, timeout=60 * 50,
    )
    print("=== STDOUT (tail) ===")
    print(result.stdout[-8000:])
    print("=== STDERR (tail) ===")
    print(result.stderr[-8000:])
    print("returncode:", result.returncode)

    # Surface server/client logs too, since eval_libero.sh backgrounds them.
    for log_path in [
        "/tmp/g05_libero_smoke/server.log",
        "/tmp/g05_libero_smoke/libero_spatial/client.log",
    ]:
        if os.path.exists(log_path):
            print(f"=== {log_path} (tail) ===")
            with open(log_path) as f:
                print(f.read()[-4000:])


@app.local_entrypoint()
def main():
    run_libero_eval_smoke.remote()
