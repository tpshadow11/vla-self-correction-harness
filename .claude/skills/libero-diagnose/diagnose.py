#!/usr/bin/env python3
"""Diagnose failed LIBERO / OpenVLA pick-place rollouts.

Input  : a failures run.jsonl (one JSON record per line, the schema written by
         modal_libero_eval.py -> *_failures.jsonl / merged.jsonl). Each record
         carries the final MuJoCo state:

             state.robot_eef_pos  [x, y, z]
             state.gripper_qpos   [finger_a, finger_b]
             state.object_pos     [x, y, z]  (or "N/A" / missing)
             state.steps_taken, state.final_reward
             task_id, episode_idx, task_description

Output : a Stage-2 diagnosis.jsonl (one record per input line) with the fields
         the rerun driver requires -- `erp_diagnosis` (a one-clause corrective
         diagnosis) and `diagnosis_failed` -- plus an `analysis` block showing
         the geometry the diagnosis was grounded on. Feed it straight into:

             modal run modal_libero_eval.py::rerun --plan <this file> ...

The diagnosis is grounded, not guessed: it is derived from the grasp geometry
(lateral miss, vertical gap, gripper open/closed) of the final state. See the
thresholds below.

Usage:
    python diagnose.py <run.jsonl> [-o <out.jsonl>] [--quiet]
    python diagnose.py <run.jsonl> -o -          # write JSONL to stdout
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# --- grasp-geometry thresholds (metres; tuned to the Panda gripper in LIBERO) --
GRIP_OPEN_THR = 0.02   # finger qpos above this => fingers still open (never grasped)
ALIGNED_XY    = 0.03   # within this lateral distance => "centred over" the object
GROSS_XY      = 0.12   # beyond this => a gross lateral miss
CONTACT_DZ    = 0.02   # |eef_z - obj_z| within this => at grasp contact height
TOO_HIGH_DZ   = 0.025  # eef this far above the object => "closed too high"
BELOW_DZ      = -0.05  # eef this far below the object => needs to be raised


def _surface(task_description: str) -> str:
    """A short prepositional phrase locating the bowl, for a richer clause."""
    d = (task_description or "").lower()
    if "drawer" in d:
        return " inside the drawer"
    if "stove" in d:
        return " on the stove"
    if "cabinet" in d:
        return " on the cabinet"
    return ""


def _is_xyz(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(
        isinstance(c, (int, float)) for c in v
    )


def classify(horiz: float, dz: float, gripper_open: bool, surf: str):
    """Map grasp geometry -> (failure_modes, diagnosis clause).

    `dz` is eef_z - obj_z (positive => end-effector above the object).
    Returns (modes: list[str], diagnosis: str).
    """
    modes: list[str] = []

    if gripper_open:
        modes.append("gripper_never_closed")
        if horiz >= GROSS_XY:
            modes.append("lateral_miss")
            obs = (f"the end-effector stalled ~{horiz:.2f} m laterally from the "
                   f"bowl{surf} and never closed the gripper")
            corr = "drive to the bowl's x,y first, then descend and close to grasp"
        elif dz <= BELOW_DZ:
            modes.append("below_target")
            obs = (f"the end-effector was ~{horiz:.2f} m from and ~{abs(dz):.2f} m "
                   f"below the bowl{surf} and never closed the gripper")
            corr = ("raise to the bowl's height, move over its x,y, then descend "
                    "and close to grasp")
        else:
            if horiz > ALIGNED_XY:
                modes.append("lateral_miss")
            if dz > CONTACT_DZ:
                modes.append("too_high")
                obs = (f"the gripper ended ~{horiz:.2f} m to the side and "
                       f"~{dz:.2f} m above the bowl{surf} with the fingers still open")
            else:
                obs = (f"the gripper ended ~{horiz:.2f} m to the side of the "
                       f"bowl{surf} with the fingers still open")
            corr = ("recenter over the bowl's x,y, descend to contact height, then "
                    "close the gripper")
    else:
        modes.append("empty_grasp")
        if horiz >= 0.08:
            modes.append("lateral_miss")
            obs = (f"the gripper closed ~{horiz:.2f} m away from the bowl{surf} and "
                   f"came up empty")
            corr = "align over the bowl's x,y before closing the gripper"
        elif dz > TOO_HIGH_DZ:
            modes.append("too_high")
            obs = (f"the gripper was centred over the bowl{surf} but closed "
                   f"~{dz:.2f} m too high and grasped air")
            corr = ("descend the last few centimetres to contact before closing "
                    "the gripper")
        else:
            modes.append("slipped")
            obs = (f"the gripper closed on the bowl{surf} but lost it before "
                   f"reaching the plate")
            corr = ("regrasp firmly over the bowl's centre and lift slowly along "
                    "the path to the plate")

    return modes, f"{obs}, so {corr}"


def diagnose_record(rec: dict) -> dict:
    """Build one Stage-2 diagnosis record from one failures record."""
    task_id = rec.get("task_id")
    episode_idx = rec.get("episode_idx")
    task_desc = rec.get("task_description") or rec.get("task_instruction") or ""
    surf = _surface(task_desc)

    state = rec.get("state") or {}
    eef = state.get("robot_eef_pos")
    obj = state.get("object_pos")
    grip = state.get("gripper_qpos")

    out = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_desc,
        "erp_diagnosis": "",
        "diagnosis_failed": True,
        "analysis": {"grounded": False, "failure_modes": []},
    }

    gripper_open = None
    if _is_xyz(grip if isinstance(grip, list) and len(grip) == 3 else None):
        pass  # gripper is 2-dof, handled below
    if isinstance(grip, (list, tuple)) and len(grip) >= 1 and isinstance(grip[0], (int, float)):
        gripper_open = grip[0] > GRIP_OPEN_THR

    # Fully grounded: we have both the end-effector and the object position.
    if _is_xyz(eef) and _is_xyz(obj) and gripper_open is not None:
        horiz = math.hypot(eef[0] - obj[0], eef[1] - obj[1])
        dz = eef[2] - obj[2]
        modes, diagnosis = classify(horiz, dz, gripper_open, surf)
        out["erp_diagnosis"] = diagnosis
        out["diagnosis_failed"] = False
        out["analysis"] = {
            "grounded": True,
            "horizontal_dist_m": round(horiz, 3),
            "height_gap_m": round(dz, 3),
            "gripper_open": gripper_open,
            "object_label": state.get("object_label"),
            "steps_taken": state.get("steps_taken"),
            "final_reward": state.get("final_reward"),
            "failure_modes": modes,
        }
        return out

    # Partial: no object pose, but the gripper tells us it never even closed.
    if gripper_open is True:
        out["erp_diagnosis"] = (
            "the gripper never closed during the episode, so approach the bowl's "
            "x,y, descend to contact height, then close the gripper to grasp")
        out["diagnosis_failed"] = False
        out["analysis"] = {
            "grounded": False,
            "gripper_open": True,
            "failure_modes": ["gripper_never_closed"],
            "note": "object_pos unavailable; diagnosis grounded on gripper state only",
        }
        return out

    # Ungrounded: cannot produce a trustworthy geometric diagnosis.
    out["analysis"]["note"] = "missing robot_eef_pos / object_pos / gripper_qpos"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose failed LIBERO rollouts.")
    ap.add_argument("run", help="path to the failures run.jsonl / merged.jsonl")
    ap.add_argument("-o", "--out", default=None,
                    help="output path ('-' for stdout). "
                         "Default: <stem>_diagnosed.jsonl next to the input.")
    ap.add_argument("--quiet", action="store_true", help="suppress the summary table")
    args = ap.parse_args(argv)

    src = Path(args.run)
    if not src.exists():
        ap.error(f"input not found: {src}")

    records = []
    for ln, line in enumerate(src.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[libero-diagnose] skipping malformed line {ln}: {e}", file=sys.stderr)

    diagnosed = [diagnose_record(r) for r in records]

    if args.out == "-":
        for d in diagnosed:
            print(json.dumps(d))
    else:
        out = Path(args.out) if args.out else src.with_name(f"{src.stem}_diagnosed.jsonl")
        out.write_text("".join(json.dumps(d) + "\n" for d in diagnosed))
        if not args.quiet:
            print(f"[libero-diagnose] wrote {len(diagnosed)} diagnoses -> {out}")

    if not args.quiet:
        usable = sum(1 for d in diagnosed if d["erp_diagnosis"] and not d["diagnosis_failed"])
        print(f"[libero-diagnose] Stage-3 usable: {usable}/{len(diagnosed)}", file=sys.stderr)
        print(f"\n{'task':>4} {'ep':>3} | {'horiz':>6} {'dz':>7} {'grip':>6} | modes",
              file=sys.stderr)
        for d in diagnosed:
            a = d["analysis"]
            h = f"{a['horizontal_dist_m']:.3f}" if "horizontal_dist_m" in a else "  -  "
            z = f"{a['height_gap_m']:+.3f}" if "height_gap_m" in a else "   -  "
            g = ("open" if a.get("gripper_open") else "closed") if "gripper_open" in a else "-"
            print(f"{str(d['task_id']):>4} {str(d['episode_idx']):>3} | {h:>6} {z:>7} "
                  f"{g:>6} | {','.join(a.get('failure_modes', []))}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
