"""Build a synthetic 'reflection' RLDS dataset from openvla/modified_libero_rlds.

For each episode, takes the tail segment (from a random point ~50-70% through
to the end) and re-emits it as a new episode with language_instruction
replaced by the reflection/retry prompt (same text used at eval time). Images
and actions are untouched real data from the original successful demo -- this
teaches the model "when given the reflection prompt, keep executing toward
task completion from wherever you are," using real trajectory continuations.

Important, honest limitation: this is built from offline replay data, so it
cannot synthesize a genuinely perturbed/failed state (no live simulator here
-- an action is only meaningful relative to the real image that follows it).
It teaches response-to-the-prompt, not recovery-from-an-actual-induced-error.
"""
import random

import tensorflow as tf

RETRY_PROMPT = "You have exceeded the number of steps. Move a few steps back and retry the task"

SRC_NAME = "libero_spatial_no_noops"
DST_NAME = "libero_spatial_reflect_no_noops"
DATA_DIR = "/data"

# Keys and their tf.train.Feature type, as found in the real data.
BYTES_KEYS = [
    "steps/language_instruction",
    "steps/observation/image",
    "steps/observation/wrist_image",
]
FLOAT_KEYS = [
    "steps/action",
    "steps/discount",
    "steps/reward",
    "steps/observation/state",
    "steps/observation/joint_state",
]
INT_KEYS = ["steps/is_first", "steps/is_last", "steps/is_terminal"]
CONTEXT_ONLY_BYTES = ["episode_metadata/file_path"]


def _get_feature_list(ex: tf.train.SequenceExample, key: str):
    feat = ex.context.feature[key]
    kind = feat.WhichOneof("kind")
    if kind == "bytes_list":
        return list(feat.bytes_list.value), "bytes"
    elif kind == "float_list":
        return list(feat.float_list.value), "float"
    elif kind == "int64_list":
        return list(feat.int64_list.value), "int64"
    raise ValueError(f"Unknown feature kind for {key}: {kind}")


def _set_feature_list(ex: tf.train.SequenceExample, key: str, values, kind: str):
    feat = ex.context.feature[key]
    if kind == "bytes":
        feat.bytes_list.value.extend(values)
    elif kind == "float":
        feat.float_list.value.extend(values)
    elif kind == "int64":
        feat.int64_list.value.extend(values)


def make_reflection_episode(raw_record: bytes, rng: random.Random) -> bytes | None:
    ex = tf.train.SequenceExample()
    ex.ParseFromString(raw_record)

    # action is [N * 7] flat float list (7-dim action per step) -- infer N from it.
    action_flat, _ = _get_feature_list(ex, "steps/action")
    n_steps = len(action_flat) // 7
    if n_steps < 10:
        return None  # too short to meaningfully split

    split = rng.randint(int(n_steps * 0.5), max(int(n_steps * 0.5) + 1, int(n_steps * 0.7)))
    tail_len = n_steps - split
    if tail_len < 5:
        return None

    out = tf.train.SequenceExample()

    # Per-step scalar fields (is_first/is_last/is_terminal/discount/reward): 1 value per step.
    for key in INT_KEYS + ["steps/discount", "steps/reward"]:
        values, kind = _get_feature_list(ex, key)
        tail = list(values[split:n_steps])
        _set_feature_list(out, key, tail, kind)

    # action, state, joint_state: fixed-width vectors per step, flattened.
    for key, width in [
        ("steps/action", 7),
        ("steps/observation/state", 8),
        ("steps/observation/joint_state", 7),
    ]:
        values, kind = _get_feature_list(ex, key)
        tail = list(values[split * width : n_steps * width])
        _set_feature_list(out, key, tail, kind)

    # images: 1 bytes value per step.
    for key in ["steps/observation/image", "steps/observation/wrist_image"]:
        values, kind = _get_feature_list(ex, key)
        tail = list(values[split:n_steps])
        _set_feature_list(out, key, tail, kind)

    # language_instruction: overwrite every step in the tail with the retry prompt.
    _set_feature_list(
        out, "steps/language_instruction",
        [RETRY_PROMPT.encode("utf-8")] * tail_len, "bytes",
    )

    # Fix up boundary flags for the new (shorter) episode.
    out.context.feature["steps/is_first"].int64_list.value[:] = (
        [1] + [0] * (tail_len - 1)
    )
    out.context.feature["steps/is_last"].int64_list.value[:] = (
        [0] * (tail_len - 1) + [1]
    )
    # is_terminal: keep original tail values (only True if the original episode
    # actually terminated there, i.e. always at the true end -- unchanged).

    # episode_metadata: copy through.
    if "episode_metadata/file_path" in ex.context.feature:
        out.context.feature["episode_metadata/file_path"].CopyFrom(
            ex.context.feature["episode_metadata/file_path"]
        )

    return out.SerializeToString()
