# VLA Self-Correction Harness

A GPU evaluation harness for studying **failure recovery in Vision-Language-Action (VLA) robot policies**.
It runs [OpenVLA-7B](https://github.com/openvla/openvla) on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
manipulation benchmark on [Modal](https://modal.com), captures every failed episode, and replays it
deterministically under controlled conditions. That makes it possible to measure whether language
feedback changes what the robot does.

Built during the **Algoverse AI Research** program (Summer 2026).

The research question: robot policies trained by behavior cloning only ever see successful
demonstrations. If a policy fails and a physics simulator tells it *why*, as a grounded
natural-language diagnosis in its prompt, does it recover more often than when it is simply given
more time?

---

## Pipeline

```
 Stage 1: Capture       Stage 2: Diagnose          Stage 3: Deterministic replay       Analysis
 ─────────────────      ──────────────────         ────────────────────────────        ───────────────
 OpenVLA on LIBERO  ->  grasp geometry + LLM   ->  same task / init state / seed   ->  matched pairs:
 save state + MuJoCo    one-line grounded           C1  diagnosis in prompt             McNemar exact,
 snapshot per failure   diagnosis                   C2  same budget, no text            Clopper-Pearson
```

1. **Capture**: run OpenVLA's official LIBERO eval and write a JSON record for every failed
   episode: end-effector, gripper, and object state, plus a byte-exact MuJoCo snapshot for replay.
2. **Diagnose**: turn each failure into a short corrective clause grounded in the measured grasp
   geometry, for example:
   > *the end-effector stalled ~0.27 m laterally from the bowl and never closed the gripper, so drive to the bowl's x,y first, then descend and close to grasp*
3. **Replay**: re-run only the failed `(task, episode)` pairs from the same initial state and RNG
   seed. The two arms (feedback vs. control) differ only in the prompt.

### Highlights

- **Deterministic replay.** Every episode reseeds NumPy *and* PyTorch (`set_seed_everywhere(episode_idx)`)
  before `env.reset()`. Without that, the no-feedback baseline reproduced only 44–56% of the original
  failures; with it, replay is essentially exact (0/5 and 3/33 divergences, the rest attributable to
  non-deterministic GPU float accumulation).
- **Inert-by-default hooks.** The changes to OpenVLA's eval are gated on `ERP_*` environment
  variables. With none set, the eval is identical to stock OpenVLA.
- **Auditable conditions.** Each run's condition is stamped in both the log filename and the log body,
  and the exact prompt is logged whenever a feedback clause is active.
- **LoRA fine-tuning on reflection data** (`reflect_finetune/`) checks whether a model *trained* to
  see the feedback prompt responds to it differently.
- **Second architecture** (`g05/`): Galaxea G0.5 served behind a batched WebSocket policy server with its
  own LIBERO and reflection loop.

### Baselines reproduced with this harness

| Model | LIBERO suite | Episodes | Success |
|---|---|---:|---:|
| OpenVLA-7B | Spatial | 50 | 82% |
| OpenVLA-7B | Object | 31 | 84% |
| OpenVLA-7B | Goal | 30 | 83% |
| OpenVLA-7B | Long | 30 | 67% |
| CogACT | Spatial | 20 | 100% |
| CogACT | Goal | 20 | 95% |
| MemVLA | Spatial | 20 | 100% |
| π0 | Spatial | 40 | 85% |
| G0.5 | Spatial | 50 | 93% |

---

## Repo layout

```
vla-self-correction-harness/
├── modal_libero_eval.py          # Modal app: image + entrypoints (smoke / main / rerun)
├── setup_libero.sh               # fetches upstream LIBERO into ./LIBERO
├── diagnose.py                   # Stage 2: LLM diagnosis per failure (Anthropic API)
├── .claude/skills/libero-diagnose/
│   ├── SKILL.md                  # Stage 2 as a Claude Code skill
│   └── diagnose.py               # geometry-based diagnosis (no API key needed)
├── openvla/                      # OpenVLA (MIT), with the capture / replay / prompt hooks
│   └── experiments/robot/
│       ├── libero/run_libero_eval.py   # official eval + env-gated hooks
│       ├── libero/libero_capture.py    # Stage 1: writes a JSON per failed episode
│       ├── libero/erp_rerun.py         # Stage 3: replays planned pairs, injects diagnosis
│       └── openvla_utils.py            # prompt construction (dedicated prefix slot)
├── reflect_finetune/             # synthetic reflection dataset + LoRA fine-tune + eval on Modal
└── g05/                          # Galaxea G0.5 integration (see g05/README.md)
```

---

## Quickstart

```bash
# one-time
pip install -r requirements-local.txt
modal token new
./setup_libero.sh

# 0. ~2 min runtime check (flash-attn on GPU, headless LIBERO env)
modal run modal_libero_eval.py::smoke --task-suite libero_spatial

# 1. capture failures (small run: 3 tasks x 4 trials)
modal run modal_libero_eval.py::main --task-suite libero_spatial --num-trials-per-task 4 --num-tasks 3
#    -> libero_results/<ts>_libero_spatial_failures.jsonl

# 2. diagnose
python diagnose.py libero_results/<ts>_libero_spatial_failures.jsonl          # LLM (needs ANTHROPIC_API_KEY in .env)
python .claude/skills/libero-diagnose/diagnose.py <failures.jsonl>            # geometry-only alternative
#    -> libero_results/<ts>_libero_spatial_diagnosed.jsonl

# 3. replay: C1 (feedback) and C2 (control, same episodes, no feedback)
modal run modal_libero_eval.py::rerun --plan libero_results/<name>_diagnosed.jsonl --task-suite libero_spatial
modal run modal_libero_eval.py::rerun --plan libero_results/<name>_diagnosed.jsonl --task-suite libero_spatial --no-inject
```

The first run builds the image (flash-attn compile, ~10 min) and downloads the 7B checkpoint
(~14 GB, cached in a Modal volume afterward). Add `--detach` to keep long runs going if your client
disconnects. The default GPU is an A10G; switch to an A100 in `modal_libero_eval.py` if you hit OOM.

Supported suites (those with a released OpenVLA checkpoint): `libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`.

### Failure record schema

```jsonc
{
  "task_suite_name": "libero_spatial",
  "task_id": 2, "episode_idx": 0,
  "task_description": "pick up the black bowl ...",
  "seed": 7,
  "checkpoint": "openvla/openvla-7b-finetuned-libero-spatial",
  "success": false,
  "state": {
    "robot_eef_pos": [x, y, z],
    "gripper_qpos": [a, b],
    "object_pos": [x, y, z],
    "object_label": "akita_black_bowl_1",
    "steps_taken": 220, "final_reward": 0.0
  },
  "init_state": [ ... ]   // byte-exact MuJoCo snapshot
}
```

---

## How the hooks work

OpenVLA's official `run_libero_eval.py` runs the rollout loop. Three env-gated hooks were added:

1. **Capture** (`libero_capture.maybe_record`): if `ERP_CAPTURE_DIR` is set and the episode failed,
   dump the state and replay snapshot.
2. **Select** (`erp_rerun`): in replay mode, run only the planned failed pairs and publish each
   episode's diagnosis to the prompt builder.
3. **Inject** (`openvla_utils`): place the feedback clause in a dedicated prompt-prefix slot:
   `In: {feedback} What action should the robot take to {task}?\nOut:`

| Variable | Purpose |
|---|---|
| `ERP_CAPTURE_DIR` | where failure JSONs are written |
| `ERP_MAX_TASKS` | cap on task ids run |
| `ERP_RERUN_PLAN` | diagnosed JSONL to replay |
| `ERP_RERUN_OUT` | where replay outcomes are written |
| `ERP_RERUN_NO_INJECT` | control arm (no feedback) |

---

## Acknowledgments

Developed as a team project in the Algoverse AI Research program. Builds on
[OpenVLA](https://github.com/openvla/openvla) (MIT License, see `openvla/LICENSE`),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[robosuite](https://github.com/ARISE-Initiative/robosuite), and
[GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA).
