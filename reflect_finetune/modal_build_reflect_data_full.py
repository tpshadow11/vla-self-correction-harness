from pathlib import Path

import modal

app = modal.App("build-reflect-dataset-full")
rlds_vol = modal.Volume.from_name("rlds-reflect-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("tensorflow==2.15.0", "tensorflow-datasets==4.9.3", "huggingface_hub")
    .pip_install("tensorflow-metadata==1.15.0", extra_options="--no-deps")
    .add_local_file(
        str(Path(__file__).parent / "build_reflect_dataset.py"),
        "/root/build_reflect_dataset.py",
        copy=True,
    )
)

DATA_DIR = "/data"
SRC_NAME = "libero_spatial_no_noops"
DST_NAME = "libero_spatial_reflect_no_noops"
NUM_SHARDS = 16


@app.function(image=image, timeout=3600, volumes={DATA_DIR: rlds_vol})
def build():
    import json
    import os
    import random
    import sys

    sys.path.insert(0, "/root")
    from build_reflect_dataset import make_reflection_episode

    from huggingface_hub import hf_hub_download
    import tensorflow as tf

    src_dir = f"{DATA_DIR}/{SRC_NAME}/1.0.0"
    os.makedirs(src_dir, exist_ok=True)

    for f in ["dataset_info.json", "features.json"]:
        hf_hub_download(
            "openvla/modified_libero_rlds", f"{SRC_NAME}/1.0.0/{f}",
            repo_type="dataset", local_dir=DATA_DIR,
        )

    with open(f"{src_dir}/dataset_info.json") as fh:
        src_info = json.load(fh)
    print("source shard lengths:", src_info["splits"][0]["shardLengths"])
    src_file_name = src_info["name"]  # actual on-disk shard prefix, e.g. "libero_spatial"

    rng = random.Random(0)
    dst_dir = f"{DATA_DIR}/{DST_NAME}/1.0.0"
    os.makedirs(dst_dir, exist_ok=True)

    n_written_total = 0
    n_skipped_total = 0
    dst_shard_lengths = []
    total_bytes = 0

    for shard_idx in range(NUM_SHARDS):
        shard_name = f"{src_file_name}-train.tfrecord-{shard_idx:05d}-of-{NUM_SHARDS:05d}"
        shard_path = hf_hub_download(
            "openvla/modified_libero_rlds", f"{SRC_NAME}/1.0.0/{shard_name}",
            repo_type="dataset", local_dir=DATA_DIR,
        )

        out_name = f"{DST_NAME}-train.tfrecord-{shard_idx:05d}-of-{NUM_SHARDS:05d}"
        out_path = f"{dst_dir}/{out_name}"
        n_written = 0
        n_skipped = 0
        with tf.io.TFRecordWriter(out_path) as writer:
            for raw_record in tf.data.TFRecordDataset([shard_path]):
                result = make_reflection_episode(raw_record.numpy(), rng)
                if result is None:
                    n_skipped += 1
                    continue
                writer.write(result)
                n_written += 1

        size = os.path.getsize(out_path)
        total_bytes += size
        dst_shard_lengths.append(str(n_written))
        n_written_total += n_written
        n_skipped_total += n_skipped
        print(f"[shard {shard_idx}] wrote {n_written}, skipped {n_skipped}, {size} bytes")

        # Remove the downloaded source shard to save volume space once processed.
        os.remove(shard_path)

    # Write derived dataset_info.json + copy features.json unchanged.
    import shutil
    shutil.copy(f"{src_dir}/features.json", f"{dst_dir}/features.json")

    dst_info = dict(src_info)
    dst_info["name"] = DST_NAME
    dst_info["moduleName"] = f"{DST_NAME}.{DST_NAME}_dataset_builder"
    dst_info["splits"] = [
        {
            "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            "name": "train",
            "numBytes": str(total_bytes),
            "shardLengths": dst_shard_lengths,
        }
    ]
    with open(f"{dst_dir}/dataset_info.json", "w") as fh:
        json.dump(dst_info, fh, indent=2)

    rlds_vol.commit()
    print(f"\nTOTAL: wrote {n_written_total} episodes, skipped {n_skipped_total} (too short)")

    # Verify: load it back with the real TFDS builder.
    import tensorflow_datasets as tfds

    builder = tfds.builder(DST_NAME, data_dir=DATA_DIR)
    print("verify: builder loaded, splits:", builder.info.splits)
    ds = builder.as_dataset(split="train[:2]")
    for i, episode in enumerate(ds):
        steps = list(episode["steps"])
        instrs = {s["language_instruction"].numpy() for s in steps}
        print(f"verify episode {i}: {len(steps)} steps, instructions used: {instrs}")


@app.local_entrypoint()
def main():
    build.remote()
