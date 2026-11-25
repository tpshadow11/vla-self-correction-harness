import modal

OPENVLA = "/root/openvla"
LIBERO = "/root/LIBERO"
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
    .add_local_dir("LIBERO", LIBERO, copy=True, ignore=IGNORE)
    .add_local_dir("openvla", OPENVLA, copy=True, ignore=IGNORE)
    .run_commands(
        f"pip install -e {LIBERO}",
        f"pip install -e {OPENVLA}",
        f"pip install -r {OPENVLA}/experiments/robot/libero/libero_requirements.txt",
    )
    .run_commands(
        "pip install --no-deps 'mujoco==2.3.7'",
        "pip install 'numpy<2'",
        "pip install --no-deps 'tensorflow-metadata==1.15.0'",
        "echo 'N' | python -c 'import libero.libero'",
    )
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HOME": "/root/.cache/huggingface"})
)

app = modal.App("eval-finetuned-reflect")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("libero-results", create_if_missing=True)
rollouts = modal.Volume.from_name("libero-rollouts", create_if_missing=True)
runs_vol = modal.Volume.from_name("reflect-finetune-runs", create_if_missing=True)

CHECKPOINT_NAME = (
    "openvla-7b-finetuned-libero-spatial+libero_spatial_reflect_mix"
    "+b8+lr-0.0005+lora-r32+dropout-0.0--erp-reflect-v1--image_aug"
)
CHECKPOINT_PATH = f"/runs/{CHECKPOINT_NAME}"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/openvla/experiments/logs": results,
        "/root/openvla/rollouts": rollouts,
        "/runs": runs_vol,
    },
)
def run_finetuned_eval(task_suite: str = "libero_spatial", num_trials_per_task: int = 10,
                        num_tasks: int = 2, task_offset: int = 0, center_crop: bool = True,
                        no_reflection_prompt: bool = False):
    import glob
    import os
    import subprocess
    import time

    tag = "control" if no_reflection_prompt else "reflect"
    capture_dir = f"{OPENVLA}/rollouts/failures/{task_suite}_ft_{tag}_{int(time.time())}"
    os.makedirs(capture_dir, exist_ok=True)
    env = {**os.environ, "ERP_CAPTURE_DIR": capture_dir}
    if num_tasks and num_tasks > 0:
        env["ERP_MAX_TASKS"] = str(num_tasks)
    if task_offset:
        env["ERP_TASK_OFFSET"] = str(task_offset)
    if no_reflection_prompt:
        env["ERP_NO_REFLECTION_PROMPT"] = "1"

    cmd = [
        "python", "experiments/robot/libero/run_libero_eval.py",
        "--model_family", "openvla",
        "--pretrained_checkpoint", CHECKPOINT_PATH,
        "--task_suite_name", task_suite,
        "--center_crop", str(center_crop),
        "--num_trials_per_task", str(num_trials_per_task),
        # stamp the condition into run_id/log filename -- without this the
        # eval log is anonymous w.r.t. ERP_NO_REFLECTION_PROMPT and can't be
        # independently verified after the fact
        "--run_id_note", f"{tag}-off{task_offset}",
    ]
    proc = subprocess.Popen(cmd, cwd=OPENVLA, env=env)
    committed = set()
    try:
        while proc.poll() is None:
            new = set(glob.glob(f"{capture_dir}/fail_*.json")) - committed
            if new:
                rollouts.commit()
                results.commit()
                committed |= new
                print(f"[erp-capture] committed {len(new)} new failure(s) ({len(committed)} total)")
            time.sleep(10)
    finally:
        rollouts.commit()
        results.commit()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"[erp-capture] done, checkpoint: {CHECKPOINT_PATH}, tag: {tag}")


@app.local_entrypoint()
def main(task_suite: str = "libero_spatial", num_trials_per_task: int = 10,
          num_tasks: int = 2, task_offset: int = 0, no_reflection_prompt: bool = False):
    run_finetuned_eval.remote(
        task_suite=task_suite, num_trials_per_task=num_trials_per_task,
        num_tasks=num_tasks, task_offset=task_offset,
        no_reflection_prompt=no_reflection_prompt,
    )
