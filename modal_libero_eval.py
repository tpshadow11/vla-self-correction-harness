# modal_libero_eval.py
import modal

OPENVLA = "/root/openvla"
LIBERO = "/root/LIBERO"

# Ignore heavy / irrelevant dirs when copying local repos into the image
IGNORE = ["**/.git", "**/__pycache__", "**/*.pyc", "**/.DS_Store",
          "**/wandb", "**/rollouts", "**/experiments/logs"]

# NOTE: The ERP failure-capture hook lives *directly in the openvla repo* —
# see openvla/experiments/robot/libero/{libero_capture.py, run_libero_eval.py}.
# It is mounted verbatim by the add_local_dir("openvla", ...) step below, so
# there is no build-time source patching here. The hook is inert unless
# ERP_CAPTURE_DIR is set (run_eval sets it), leaving default behaviour unchanged.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
    )
    # System deps for headless MuJoCo/robosuite (EGL) + video encoding.
    # Toolchain (clang/gcc/cmake) is required because robosuite==1.4.1 drags in
    # packages that compile native code during pip install:
    #   - evdev (via pynput): C extension; the standalone python invokes `clang`
    #   - egl_probe (direct robosuite dep): builds via CMake + a C compiler
    .apt_install(
        "git", "libgl1-mesa-glx", "libegl1", "libosmesa6",
        "libglew-dev", "libglfw3", "libglib2.0-0", "ffmpeg",
        "build-essential", "clang", "cmake",
    )
    # NOTE: wheel + setuptools must be present BEFORE the flash-attn step because
    # --no-build-isolation makes flash-attn's setup.py build against these packages.
    .pip_install("torch==2.2.0", "torchvision==0.17.0", "packaging", "ninja", "wheel", "setuptools")
    # flash-attn is required for OpenVLA eval; needs --no-build-isolation
    .pip_install("flash-attn==2.5.5", extra_options="--no-build-isolation")
    # Ship + install both local repos (openvla already carries the ERP capture
    # hook: libero_capture.py + the edited run_libero_eval.py).
    .add_local_dir("LIBERO", LIBERO, copy=True, ignore=IGNORE)
    .add_local_dir("openvla", OPENVLA, copy=True, ignore=IGNORE)
    .run_commands(
        f"pip install -e {LIBERO}",
        f"pip install -e {OPENVLA}",
        f"pip install -r {OPENVLA}/experiments/robot/libero/libero_requirements.txt",
    )
    # Separate layer (below flash-attn) so these fixes don't trigger a recompile:
    #  1. torch 2.2.0 + robosuite were built against NumPy 1.x; NumPy 2.x crashes them.
    #  2. openvla's prismatic pkg eagerly imports its RLDS/tfds *training* pipeline even
    #     for eval. The latest tensorflow-metadata was rebuilt against protobuf 5.26
    #     (its _pb2 imports `runtime_version`), but TF 2.15 pins protobuf <5, so the
    #     import fails. Pin metadata back to a protobuf-4-era build; --no-deps leaves
    #     TF's protobuf 4.25 untouched.
    #  3. LIBERO's __init__ interactively prompts for a dataset path on first import,
    #     which deadlocks in a headless container. Feeding 'N' writes the default
    #     ~/.libero/config.yaml at build time so runtime imports are non-interactive.
    .run_commands(
        # robosuite 1.4.1 (mujoco>=2.3.0, no upper bound) calls the old
        # mj_fullM(m, dst, M) signature; mujoco 3.x changed it to mj_fullM(m, d, dst),
        # so the unpinned latest (3.x) raises TypeError. Pin the last 2.3.x, which is
        # contemporaneous with robosuite 1.4.1. --no-deps leaves numpy/glfw untouched.
        "pip install --no-deps 'mujoco==2.3.7'",
        "pip install 'numpy<2'",
        "pip install --no-deps 'tensorflow-metadata==1.15.0'",
        "echo 'N' | python -c 'import libero.libero'",
    )
    # CRITICAL: headless GPU rendering backend for MuJoCo
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HOME": "/root/.cache/huggingface"})
)

app = modal.App("openvla-libero-eval")

# Cache HF checkpoints across runs so we download the 7B once
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# Persist eval logs (run_libero_eval.py writes to ./experiments/logs)
results = modal.Volume.from_name("libero-results", create_if_missing=True)
# Persist rollout MP4s (save_rollout_video writes to ./rollouts, relative to cwd)
rollouts = modal.Volume.from_name("libero-rollouts", create_if_missing=True)

# Map task suite -> released finetuned checkpoint
CHECKPOINTS = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object":  "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal":    "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10":      "openvla/openvla-7b-finetuned-libero-10",
}

@app.function(
    image=image,
    gpu="A10G",                     # bump to "A100" if you hit OOM
    # A full suite is long: 50 trials/task x 10 tasks x ~67s/trial ~= 9.3h, so
    # 4h was not enough for a real run. 12h leaves margin (incl. one-time 7B
    # checkpoint download). Timeout is only a ceiling — Modal bills actual runtime.
    timeout=60 * 60 * 12,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/openvla/experiments/logs": results,
        "/root/openvla/rollouts": rollouts,
    },
)
def run_eval(task_suite: str = "libero_spatial", center_crop: bool = True,
             num_trials_per_task: int = 50, num_tasks: int = 0, task_offset: int = 0):
    import glob
    import json
    import os
    import subprocess
    import time

    ckpt = CHECKPOINTS[task_suite]

    # Per-run capture dir on the (persisted) rollouts volume; the subprocess
    # inherits ERP_CAPTURE_DIR and libero_capture.maybe_record writes one
    # fail_*.json per failed episode there.
    capture_dir = f"{OPENVLA}/rollouts/failures/{task_suite}_{int(time.time())}"
    os.makedirs(capture_dir, exist_ok=True)
    env = {**os.environ, "ERP_CAPTURE_DIR": capture_dir}
    # num_tasks > 0 caps the run to N task_ids of the suite, starting at
    # task_offset -- lets one suite be split into several independent batches
    # (e.g. to keep any single run short enough to survive a flaky connection).
    if num_tasks and num_tasks > 0:
        env["ERP_MAX_TASKS"] = str(num_tasks)
    if task_offset:
        env["ERP_TASK_OFFSET"] = str(task_offset)

    cmd = [
        "python", "experiments/robot/libero/run_libero_eval.py",
        "--model_family", "openvla",
        "--pretrained_checkpoint", ckpt,
        "--task_suite_name", task_suite,
        "--center_crop", str(center_crop),
        "--num_trials_per_task", str(num_trials_per_task),
    ]

    # Run the eval as a child process so we can persist captured failures *as
    # they land* rather than only at the very end. libero_capture writes one
    # fail_*.json per failed episode into capture_dir (on the rollouts volume);
    # each time new file(s) appear we commit, so an interruption keeps every
    # failure captured so far instead of discarding the whole run.
    proc = subprocess.Popen(cmd, cwd=OPENVLA, env=env)
    committed = set()
    try:
        while proc.poll() is None:
            new = set(glob.glob(f"{capture_dir}/fail_*.json")) - committed
            if new:
                rollouts.commit()   # persist newly-captured failure(s) + videos
                results.commit()    # + flush eval logs written so far
                committed |= new
                print(f"[erp-capture] committed {len(new)} new failure(s) "
                      f"({len(committed)} total) to the volume")
            time.sleep(10)
    finally:
        # Final sweep: persist anything written since the last poll (the last
        # few failures + the completed eval log), even on error/cancellation.
        rollouts.commit()
        results.commit()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # Read the captured failures back so the local entrypoint can persist them.
    failures = []
    for p in sorted(glob.glob(f"{capture_dir}/fail_*.json")):
        with open(p) as f:
            failures.append(json.load(f))
    print(f"[erp-capture] {len(failures)} failure(s) captured -> {capture_dir}")

    return {
        "task_suite": task_suite,
        "num_trials_per_task": num_trials_per_task,
        "task_offset": task_offset,
        "n_failures": len(failures),
        "failures": failures,
        "capture_dir": capture_dir,
    }


@app.function(
    image=image,
    gpu="A10G",                     # bump to "A100" if you hit OOM
    timeout=60 * 60 * 2,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/openvla/experiments/logs": results,
        "/root/openvla/rollouts": rollouts,
    },
)
def run_rerun(task_suite: str, plan_records: list, center_crop: bool = True,
              inject: bool = True):
    """Stage 3: replay the previously-failed (task_id, episode_idx) pairs in
    `plan_records` with each record's ERP diagnosis injected into the prompt.

    Returns the recovery rate (fraction of replayed failures that now succeed).
    Set inject=False to run the C2 baseline (same episodes, no diagnosis) — it
    should reproduce the original failures, confirming determinism.
    """
    import glob
    import json
    import os
    import subprocess
    import time

    ckpt = CHECKPOINTS[task_suite]

    # Write the rerun plan into the container for the patched eval to load.
    plan_path = "/root/rerun_plan.jsonl"
    with open(plan_path, "w") as f:
        for r in plan_records:
            f.write(json.dumps(r) + "\n")

    # Derive loop bounds from the plan so the eval reaches every planned pair.
    max_task_id = max(int(r["task_id"]) for r in plan_records)
    max_episode_idx = max(int(r["episode_idx"]) for r in plan_records)

    out_dir = f"{OPENVLA}/rollouts/rerun/{task_suite}_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)
    env = {
        **os.environ,
        "ERP_RERUN_PLAN": plan_path,
        "ERP_RERUN_OUT": out_dir,
        "ERP_MAX_TASKS": str(max_task_id + 1),
    }
    if not inject:
        env["ERP_RERUN_NO_INJECT"] = "1"

    subprocess.run(
        [
            "python", "experiments/robot/libero/run_libero_eval.py",
            "--model_family", "openvla",
            "--pretrained_checkpoint", ckpt,
            "--task_suite_name", task_suite,
            "--center_crop", str(center_crop),
            "--num_trials_per_task", str(max_episode_idx + 1),
        ],
        cwd=OPENVLA,
        check=True,
        env=env,
    )
    results.commit()
    rollouts.commit()

    outcomes = []
    for p in sorted(glob.glob(f"{out_dir}/rerun_*.json")):
        with open(p) as f:
            outcomes.append(json.load(f))
    n = len(outcomes)
    recovered = sum(1 for o in outcomes if o["recovered"])
    rate = (recovered / n) if n else 0.0
    tag = "with diagnosis (C1)" if inject else "baseline, no diagnosis (C2)"
    print(f"[erp-rerun] {tag}: recovery {recovered}/{n} = {rate:.1%} -> {out_dir}")

    return {
        "task_suite": task_suite,
        "injected": inject,
        "n": n,
        "recovered": recovered,
        "recovery_rate": rate,
        "outcomes": outcomes,
        "out_dir": out_dir,
    }


@app.function(image=image, gpu="A10G", timeout=60 * 15)
def smoke(task_suite: str = "libero_spatial"):
    """Cheap (~2 min) validation of the runtime *before* the full eval:
    confirms flash-attn imports on GPU and that a LIBERO env can be created,
    reset, rendered headlessly through EGL, and stepped. Does NOT download the
    7B checkpoint or run trials. Invoke with:
        modal run modal_libero_eval.py::smoke
    """
    import os, sys
    os.chdir(OPENVLA)                 # so `experiments.*` imports resolve
    sys.path.insert(0, OPENVLA)

    import flash_attn                 # fails fast if the flash-attn build/GPU is bad
    print(f"[smoke] flash_attn {flash_attn.__version__} imported OK")

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_env, get_libero_dummy_action, get_libero_image,
    )

    task_suite_obj = benchmark.get_benchmark_dict()[task_suite]()
    task = task_suite_obj.get_task(0)
    init_states = task_suite_obj.get_task_init_states(0)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    print(f"[smoke] created env for task: {task_description!r}")

    env.reset()
    obs = env.set_init_state(init_states[0])
    obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
    img = get_libero_image(obs, 224)  # EGL render -> numpy image
    print(f"[smoke] EGL render OK, image shape={img.shape}, dtype={img.dtype}")
    print("[smoke] PASS — runtime looks good; safe to run the full eval.")


def _save_failures_locally(summary):
    """Write captured failures to libero_results/<ts>_<suite>_failures.jsonl
    (one record per line) so they land durably in the repo next to results."""
    import json
    import os
    import time

    failures = (summary or {}).get("failures", []) or []
    suite = (summary or {}).get("task_suite", "unknown")
    os.makedirs("libero_results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"libero_results/{ts}_{suite}_failures.jsonl"
    with open(path, "w") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")
    print(f"[erp-capture] wrote {len(failures)} failure(s) -> {path}")
    return path


@app.local_entrypoint()
def main(task_suite: str = "libero_spatial", num_trials_per_task: int = 50, num_tasks: int = 0,
         task_offset: int = 0):
    summary = run_eval.remote(
        task_suite=task_suite,
        num_trials_per_task=num_trials_per_task,
        num_tasks=num_tasks,
        task_offset=task_offset,
    )
    _save_failures_locally(summary)


def _save_rerun_locally(summary):
    """Write per-episode replay outcomes to libero_results/<ts>_<suite>_rerun.jsonl
    and print the headline recovery rate."""
    import json
    import os
    import time

    outcomes = (summary or {}).get("outcomes", []) or []
    suite = (summary or {}).get("task_suite", "unknown")
    os.makedirs("libero_results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = "rerun" if (summary or {}).get("injected", True) else "rerun_baseline"
    path = f"libero_results/{ts}_{suite}_{tag}.jsonl"
    with open(path, "w") as f:
        for rec in outcomes:
            f.write(json.dumps(rec) + "\n")
    print(f"[erp-rerun] recovery {summary.get('recovered')}/{summary.get('n')} "
          f"= {summary.get('recovery_rate', 0.0):.1%} -> {path}")
    return path


@app.local_entrypoint()
def rerun(task_suite: str = "libero_spatial", plan: str = "", inject: bool = True):
    """Stage 3 driver. `plan` is a local *_diagnosed.jsonl (one diagnosed failure
    per line). Runs only those failed episodes with the diagnosis injected and
    reports the recovery rate.

        modal run modal_libero_eval.py::rerun --plan libero_results/<f>_diagnosed.jsonl

    Add --no-inject for the C2 baseline (same episodes, no diagnosis).
    """
    import json

    assert plan, "pass --plan path/to/*_diagnosed.jsonl"
    records = []
    with open(plan) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("erp_diagnosis") or r.get("diagnosis_failed"):
                continue  # only replay successfully-diagnosed failures
            # Only the fields run_rerun needs (drop the heavy init_state/state).
            records.append({
                "task_id": r["task_id"],
                "episode_idx": r["episode_idx"],
                "task_description": r.get("task_description", ""),
                "erp_diagnosis": r["erp_diagnosis"],
                "diagnosis_failed": False,
            })
    assert records, f"no usable diagnosed records in {plan}"
    print(f"[erp-rerun] replaying {len(records)} diagnosed failure(s), inject={inject}")

    summary = run_rerun.remote(task_suite=task_suite, plan_records=records, inject=inject)
    _save_rerun_locally(summary)
