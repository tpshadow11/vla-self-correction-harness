"""diagnose.py — ERP Stage 2: get a real Claude diagnosis for each captured failure.

Reads a `*_failures.jsonl` (Stage 1 output) and writes a `*_diagnosed.jsonl`
with `erp_diagnosis` + `diagnosis_failed` added to each line, per the schema
`erp_rerun.py` (Stage 3) expects. Runs locally — no GPU / Modal needed.

Usage:
    python diagnose.py libero_results/<ts>_libero_spatial_failures.jsonl
    # -> writes libero_results/<ts>_libero_spatial_diagnosed.jsonl
"""

import json
import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You are diagnosing why a robot arm failed a pick-and-place task in the "
    "LIBERO simulator. You will be given the task instruction and the robot/"
    "object state at the end of the failed episode (end-effector position, "
    "gripper joint position, target object position and label, steps taken, "
    "final reward). Respond with ONE short clause, at most 25 words, in the "
    "exact form: '<what went wrong>, so <corrective action>'. No preamble, "
    "no quotes, no labels, lowercase start, no trailing period."
)


def diagnose_one(client, record):
    payload = {
        "task_description": record.get("task_description"),
        "state": record.get("state"),
    }
    resp = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    tokens = resp.usage.input_tokens + resp.usage.output_tokens
    if len(text) < 15:
        raise RuntimeError(f"diagnosis too short/empty ({len(text)} chars, {tokens} tokens) -- likely truncated: {text!r}")
    return text, tokens


def main():
    if len(sys.argv) != 2:
        print("usage: python diagnose.py <failures.jsonl>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = in_path.replace("_failures.jsonl", "_diagnosed.jsonl")
    if out_path == in_path:
        out_path = in_path.rsplit(".jsonl", 1)[0] + "_diagnosed.jsonl"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set (check .env)")
        sys.exit(1)
    client = Anthropic(api_key=api_key)

    with open(in_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[diagnose] {len(records)} failure(s) to diagnose using {MODEL}")

    out_records = []
    for i, rec in enumerate(records):
        tag = f"t{rec['task_id']}_e{rec['episode_idx']}"
        try:
            diagnosis, tokens = diagnose_one(client, rec)
            rec["erp_diagnosis"] = diagnosis
            rec["diagnosis_failed"] = False
            rec["failure_reason"] = None
            rec["tokens_used"] = tokens
            rec["erp_model"] = MODEL
            rec["simulated"] = False
            print(f"[diagnose] {i+1}/{len(records)} {tag}: {diagnosis}")
        except Exception as e:
            rec["erp_diagnosis"] = None
            rec["diagnosis_failed"] = True
            rec["failure_reason"] = str(e)
            rec["tokens_used"] = 0
            rec["erp_model"] = MODEL
            rec["simulated"] = False
            print(f"[diagnose] {i+1}/{len(records)} {tag}: FAILED -> {e}")
        out_records.append(rec)

    with open(out_path, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    ok = sum(1 for r in out_records if not r["diagnosis_failed"])
    print(f"[diagnose] wrote {ok}/{len(out_records)} diagnoses -> {out_path}")


if __name__ == "__main__":
    main()
