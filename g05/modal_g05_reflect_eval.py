from pathlib import Path

import modal

LIBERO = "/root/LIBERO"
GALAXEA = "/root/GalaxeaVLA"

app = modal.App("g05-reflect-eval")

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
    .add_local_dir("LIBERO", LIBERO, copy=True,
                    ignore=["**/.git", "**/__pycache__", "**/*.pyc"])
    .run_commands(f"pip install -e {LIBERO}")
    .run_commands(
        "sed -i "
        "'s/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' "
        f"{LIBERO}/libero/libero/benchmark/__init__.py"
    )
    .add_local_file(
        str(Path(__file__).parent / "eval_libero_reflect.py"),
        f"{GALAXEA}/experiments/libero/eval_libero_reflect.py",
        copy=True,
    )
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HOME": "/root/.cache/huggingface"})
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 90,
    secrets=[modal.Secret.from_name("g05-hf-token")],
    volumes={f"{GALAXEA}/checkpoints": g05_ckpt},
)
def run_reflect_smoke():
    import os
    import socket
    import subprocess
    import time

    os.chdir(GALAXEA)
    os.makedirs("/tmp/g05_reflect_smoke", exist_ok=True)

    port = 12345
    server_log = open("/tmp/g05_reflect_smoke/server.log", "w")
    server_proc = subprocess.Popen(
        [
            "python", "scripts/serve_policy_batched.py",
            "--ckpt_path", "checkpoints/g05-libero/model.pt",
            "--host", "0.0.0.0", "--port", str(port),
            "eval_embodiment=libero",
            "--action_steps", "10",
            "--max_batch_size", "30",
            "--max_wait_ms", "1000",
        ],
        stdout=server_log, stderr=subprocess.STDOUT,
    )
    print(f"[*] server PID {server_proc.pid}, waiting for port {port} to open...")

    ready = False
    for _ in range(300):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                ready = True
                break
        except OSError:
            if server_proc.poll() is not None:
                print("[FAIL] server process died during startup")
                break
            time.sleep(2)

    if not ready:
        server_log.close()
        with open("/tmp/g05_reflect_smoke/server.log") as f:
            print("=== server.log (tail) ===")
            print(f.read()[-4000:])
        raise RuntimeError("server never became ready")

    print("[*] server ready, running reflection eval...")
    result = subprocess.run(
        [
            "python", "experiments/libero/eval_libero_reflect.py",
            "--server_uri", f"ws://127.0.0.1:{port}",
            "--task_suite_name", "libero_10",
            "--num_trials_per_task", "5",
            "--num_parallel", "5",
            "--output_dir", "/tmp/g05_reflect_smoke",
        ],
        capture_output=True, text=True, timeout=60 * 80,
    )
    print("=== eval STDOUT (tail) ===")
    print(result.stdout[-6000:])
    print("=== eval STDERR (tail) ===")
    print(result.stderr[-10000:])
    print("returncode:", result.returncode)

    server_proc.terminate()
    server_log.close()

    summary_path = "/tmp/g05_reflect_smoke/libero_10_reflect_results.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            print("=== summary.json ===")
            print(f.read())


@app.local_entrypoint()
def main():
    run_reflect_smoke.remote()
