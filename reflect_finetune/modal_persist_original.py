import modal

app = modal.App("persist-original-rlds")
rlds_vol = modal.Volume.from_name("rlds-reflect-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub")

DATA_DIR = "/data"
SRC_NAME = "libero_spatial_no_noops"
NUM_SHARDS = 16


@app.function(image=image, timeout=1800, volumes={DATA_DIR: rlds_vol})
def persist():
    import json
    import os
    from huggingface_hub import hf_hub_download

    src_dir = f"{DATA_DIR}/{SRC_NAME}/1.0.0"
    os.makedirs(src_dir, exist_ok=True)
    with open(f"{src_dir}/dataset_info.json") as fh:
        info = json.load(fh)
    src_file_name = info["name"]

    for i in range(NUM_SHARDS):
        shard_name = f"{src_file_name}-train.tfrecord-{i:05d}-of-{NUM_SHARDS:05d}"
        dest = f"{src_dir}/{shard_name}"
        if os.path.exists(dest):
            print(f"[cached] {shard_name}")
            continue
        hf_hub_download(
            "openvla/modified_libero_rlds", f"{SRC_NAME}/1.0.0/{shard_name}",
            repo_type="dataset", local_dir=DATA_DIR,
        )
        print(f"[downloaded] {shard_name}")

    rlds_vol.commit()
    print("done, listing:", os.listdir(src_dir))


@app.local_entrypoint()
def main():
    persist.remote()
