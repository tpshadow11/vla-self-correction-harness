---
name: libero-diagnose
description: >-
  Diagnose failed LIBERO / OpenVLA pick-place rollouts. Given a failures
  run.jsonl (records with state.robot_eef_pos, gripper_qpos, object_pos), it
  computes the grasp geometry (lateral miss, vertical gap, gripper open/closed),
  classifies the failure mode, and writes a Stage-2 diagnosis.jsonl (erp_diagnosis
  + diagnosis_failed) ready for `modal_libero_eval.py::rerun`. Use whenever the
  user has a *_failures.jsonl / merged.jsonl of failed episodes and wants
  corrective diagnoses, or asks to "diagnose" a LIBERO run.
---

# LIBERO failure diagnosis

Turn a failures `run.jsonl` into a grounded `diagnosis.jsonl` for the ERP
recovery experiment (Stage 2 of the pipeline in `README.md`).

## When to use

The user points at a failures file — e.g.
`libero_results/<ts>_libero_spatial_failures.jsonl` or
`libero_failure_simulations/<suite>/merged.jsonl` — and wants a diagnosis per
failed episode, or wants the file that feeds
`modal run modal_libero_eval.py::rerun --plan <file>`.

## Input schema (one JSON object per line)

Only the final MuJoCo state is stored, so the diagnosis is reconstructed from it:

```jsonc
{
  "task_id": 2, "episode_idx": 0,
  "task_description": "pick up the black bowl ...",
  "state": {
    "robot_eef_pos": [x, y, z],      // end-effector at episode end
    "gripper_qpos":  [a, b],         // finger joints; ~0.04 open, ~0.00 closed
    "object_pos":    [x, y, z],      // the target bowl (may be "N/A"/missing)
    "steps_taken": 220, "final_reward": 0.0
  }
}
```

## How the diagnosis is grounded (not guessed)

From the final state, per record:

- **lateral miss** `horiz = hypot(eef.x - obj.x, eef.y - obj.y)`
- **vertical gap** `dz = eef.z - obj.z`  (positive = end-effector above the bowl)
- **gripper** `open` if `gripper_qpos[0] > 0.02`, else `closed`

Failure taxonomy (thresholds are constants at the top of `diagnose.py`):

| final state | failure mode | corrective clause |
|---|---|---|
| gripper **open**, `horiz` large (≥0.12) | `lateral_miss` + never closed | go to bowl x,y, then descend & close |
| gripper **open**, `dz` ≤ −0.05 (below) | `below_target` + never closed | raise to bowl height, then descend & close |
| gripper **open**, near in x,y | never closed (± too high) | recenter, descend to contact, close |
| gripper **closed**, `horiz` ≥ 0.08 | `empty_grasp` far away | align over x,y before closing |
| gripper **closed**, `dz` > 0.025 | `empty_grasp` + `too_high` | descend to contact before closing |
| gripper **closed**, aligned & low | `slipped` | regrasp firmly, lift slowly |

The two dominant clusters in practice: **(a)** the approach stalls with the
fingers still open (the close action never fires), and **(b)** the close fires
while ~3 cm too high or grossly misaligned, grasping nothing.

## Run it

```bash
# default: writes <stem>_diagnosed.jsonl next to the input
python .claude/skills/libero-diagnose/diagnose.py \
    libero_failure_simulations/libero_spatial/merged.jsonl

# choose the output path (or -o - to stream JSONL to stdout)
python .claude/skills/libero-diagnose/diagnose.py <run.jsonl> \
    -o libero_results/$(date +%Y%m%d_%H%M%S)_libero_spatial_diagnosed.jsonl
```

The script prints a per-episode geometry table and an `X/N Stage-3 usable`
count. Each output line is:

```jsonc
{
  "task_id": 2, "episode_idx": 0,
  "task_description": "pick up the black bowl ...",
  "erp_diagnosis": "the end-effector stalled ~0.27 m laterally from the bowl and never closed the gripper, so drive to the bowl's x,y first, then descend and close to grasp",
  "diagnosis_failed": false,
  "analysis": { "horizontal_dist_m": 0.274, "height_gap_m": 0.023,
                "gripper_open": true, "failure_modes": ["gripper_never_closed","lateral_miss"],
                "grounded": true }
}
```

`erp_diagnosis` + `diagnosis_failed` are the fields the rerun driver reads
(`modal_libero_eval.py`, `rerun()` filter). The `analysis` block is extra
context for review and is ignored downstream.

## Feed it to Stage 3

```bash
modal run modal_libero_eval.py::rerun \
    --plan libero_results/<name>_diagnosed.jsonl \
    --task-suite libero_spatial
# add --no-inject for the C2 baseline (same episodes, diagnosis NOT injected)
```

## When invoked interactively

1. Run `diagnose.py` on the file the user names (default output next to it, or a
   timestamped path under `libero_results/` if they'll rerun).
2. Read the printed geometry table; sanity-check that each `erp_diagnosis`
   matches its numbers. If any record is `diagnosis_failed: true` (e.g.
   `object_pos` was `"N/A"`), tell the user which episodes and why — do **not**
   invent a diagnosis for them.
3. If the user wants sharper, per-scene wording than the templates, refine the
   `erp_diagnosis` strings by hand using the `analysis` numbers as the anchor;
   keep the "<observation>, so <correction>" shape and the measured distances.

## Notes / gotchas

- **Field names matter.** Stage 3 filters on `erp_diagnosis` (non-empty) and
  `diagnosis_failed` (falsy). A `diagnosis` field is ignored and yields
  "no usable diagnosed records".
- The diagnosis reflects the **final** frame only; it explains where the grasp
  ended up, not the full trajectory. That is what the ERP prompt injection needs.
- Distances are in metres, rounded to millimetres in `analysis`.
