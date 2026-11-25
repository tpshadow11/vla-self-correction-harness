# G0.5 (Galaxea) Integration

[G0.5](https://huggingface.co/OpenGalaxea) is a separate VLA (2B params, autoregressive
Qwen3.5 backbone) evaluated alongside OpenVLA as a comparison point — notably, it has a
real persistent-context mechanism (`ChunkedPolicyWrapper`, a `num_obs_steps`-length frame
buffer) that OpenVLA lacks. This directory is self-contained: own model loading, own
LIBERO eval loop, own reflection mechanism, separate from the OpenVLA/Modal harness in
the repo root.

## Scripts

```
g05/
├── modal_g05_loadtest.py       # verifies the model loads + runs on Modal
├── modal_g05_libero_eval.py    # Stage 1: baseline LIBERO eval (no reflection)
├── eval_libero_reflect.py      # Stage 2: reflection mechanism + eval loop
├── modal_g05_reflect_eval.py   # orchestrates the Stage 2 reflection runs
├── libero_eval_utils.py        # shared eval helpers
├── policy_ws_client.py         # WebSocket client for the served policy
└── serve_policy_batched.py     # policy server
```

## Results

**Baseline** (LIBERO-Spatial, 50 trials): **93–100% success**, noticeably higher than
OpenVLA's baseline on the same suite.

**Reflection.** At this baseline only 1 of 50 episodes exhausted its step budget, so a
reflection comparison on G0.5 needs a larger run or a harder suite (e.g. LIBERO-Long) to
gather enough triggered episodes.

## Known limitation

This evaluation ran G0.5 with `num_obs_steps=1`, i.e. its memory buffer effectively
disabled, to match OpenVLA's single-frame conditioning and keep the comparison fair. A
memory-enabled run (`num_obs_steps > 1`) has **not** been done — if you want to actually
exercise G0.5's context mechanism, that's the first thing to try.

## Setup notes (undocumented upstream, found by trial and error)

The checkpoint needs:
- `hf_processor/` (Qwen tokenizer/processor files) at `checkpoints/g05-libero/hf_processor/`
- `action_tokenizer.pt` at `checkpoints/g05-libero/action_tokenizer.pt`

Neither path is mentioned in G0.5's own docs — the checkpoint's `config.yaml`'s
`hf_processor_path` field is what points to the first one.
