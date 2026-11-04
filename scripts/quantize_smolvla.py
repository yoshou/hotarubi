import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from safetensors.torch import save_file

import mx

MODEL = "lerobot/smolvla_base"
MODEL_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
CAMERAS = {
    "observation.images.camera1": "observation.images.top",
    "observation.images.camera2": "observation.images.wrist",
}
W8_LAYERS = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")
PATCH_EMBEDDING = "patch_embedding"


class Dataset:
    def __init__(self, root):
        self.root = root
        self.info = json.loads((root / "meta/info.json").read_text())
        self.fps = self.info["fps"]
        episodes = []
        for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
            episodes += pq.read_table(path).to_pylist()
        self.episodes = {row["episode_index"]: row for row in episodes}
        tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
        self.tasks = {row["task_index"]: row["__index_level_0__"] for row in tasks}
        self.rows = {}

    def row(self, episode, frame):
        meta = self.episodes[episode]
        path = self.root / self.info["data_path"].format(
            chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
        if path not in self.rows:
            table = pq.read_table(path, columns=["index", "observation.state", "task_index"]).to_pylist()
            self.rows[path] = {row["index"]: row for row in table}
        return self.rows[path][meta["dataset_from_index"] + frame]

    def image(self, episode, frame, key):
        meta = self.episodes[episode]
        path = self.root / self.info["video_path"].format(
            video_key=key, chunk_index=meta[f"videos/{key}/chunk_index"],
            file_index=meta[f"videos/{key}/file_index"])
        time_s = max(0.0, meta[f"videos/{key}/from_timestamp"] + (frame - 0.5) / self.fps)
        height, width, channels = self.info["features"][key]["shape"]
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{time_s:.6f}", "-i", str(path),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True).stdout
        return np.frombuffer(out, np.uint8).reshape(height, width, channels)


def select_frames(dataset, count, seed):
    rng = np.random.default_rng(seed)
    episodes = sorted(e for e in dataset.episodes if e % 2 == 0)
    frames = []
    for _ in range(count):
        episode = int(rng.choice(episodes))
        length = dataset.episodes[episode]["length"]
        t = rng.uniform(0, length / dataset.fps - 1.0)
        frames.append((episode, min(int(round(t * dataset.fps)), length - 1)))
    return frames


class Collector:
    def __init__(self):
        self.hessians = {}
        self.last_input = None
        self.last_gram = None

    def hook(self, name):
        def collect(module, inputs):
            x = inputs[0]
            if self.last_input is not x:
                flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float64)
                self.last_input = x
                self.last_gram = (flat.T @ flat, flat.shape[0])
            if name not in self.hessians:
                self.hessians[name] = mx.Hessian(x.shape[-1])
            self.hessians[name].add(*self.last_gram)
        return collect


def relative_error(w, w_hat, h):
    d = mx.pad_k(w) - w_hat
    w = mx.pad_k(w)
    h = torch.nn.functional.pad(h, (0, w.shape[1] - h.shape[0], 0, w.shape[1] - h.shape[0]))
    return float(torch.trace(d @ h @ d.T) / torch.trace(w @ h @ w.T))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("build/data/svla_so100_pickplace"))
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=Path("build/quant/smolvla.safetensors"))
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    frames = select_frames(dataset, args.frames, args.seed)
    policy = SmolVLAPolicy.from_pretrained(MODEL, revision=MODEL_REVISION)
    policy.eval()
    policy.float()
    config = policy.config
    preprocess, _ = make_pre_post_processors(
        config, pretrained_path=MODEL, pretrained_revision=MODEL_REVISION,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})

    collector = Collector()
    hooks = [module.register_forward_pre_hook(collector.hook(name))
             for name, module in policy.named_modules() if isinstance(module, torch.nn.Linear)]
    start = time.time()
    for i, (episode, frame) in enumerate(frames):
        row = dataset.row(episode, frame)
        batch = {key: torch.from_numpy(dataset.image(episode, frame, source).copy()).permute(2, 0, 1)[None].float() / 255.0
                 for key, source in CAMERAS.items()}
        batch[OBS_STATE] = torch.tensor(row["observation.state"], dtype=torch.float32)[None]
        batch["task"] = dataset.tasks[row["task_index"]]
        batch = preprocess(batch)
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        torch.manual_seed(i)
        noise = torch.randn(1, config.chunk_size, config.max_action_dim)
        with torch.no_grad():
            policy.model.sample_actions(images, image_masks, batch[OBS_LANGUAGE_TOKENS],
                                        batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=noise)
        collector.last_input = None
        collector.last_gram = None
        print(f"frame {i}: episode {episode} frame {frame} ({time.time() - start:.0f} s)", flush=True)
    for hook in hooks:
        hook.remove()

    tensors = {}
    start = time.time()
    for name, module in policy.named_modules():
        weight = module.weight.detach() if hasattr(module, "weight") else None
        if isinstance(module, torch.nn.Conv2d) and name.endswith(PATCH_EMBEDDING):
            q, scale = mx.mxint8(weight.reshape(weight.shape[0], -1))
            tensors[f"{name}.weight.mxint8"] = q
            tensors[f"{name}.weight.scale"] = scale
            print(f"{name}: mxint8 {tuple(q.shape)}", flush=True)
        elif isinstance(module, torch.nn.Linear) and name in collector.hessians:
            if name.rsplit(".", 1)[-1] in W8_LAYERS:
                q, scale = mx.mxint8(weight)
                tensors[f"{name}.weight.mxint8"] = q
                tensors[f"{name}.weight.scale"] = scale
                print(f"{name}: mxint8 {tuple(q.shape)}", flush=True)
            else:
                h = collector.hessians[name].value()
                codes, scale = mx.gptq_mxfp4(weight, h)
                tensors[f"{name}.weight.mxfp4"] = codes
                tensors[f"{name}.weight.scale"] = scale
                rtn_codes, rtn_scale = mx.mxfp4(weight)
                gptq = relative_error(weight.float(), mx.mxfp4_value(codes, scale), h)
                rtn = relative_error(weight.float(), mx.mxfp4_value(rtn_codes, rtn_scale), h)
                print(f"{name}: mxfp4 {tuple(codes.shape)} error {gptq:.5f} (rtn {rtn:.5f}) "
                      f"({time.time() - start:.0f} s)", flush=True)
        elif isinstance(module, torch.nn.Linear):
            print(f"{name}: not run", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={
        "model": MODEL, "model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
        "frames": str(args.frames), "seed": str(args.seed)})
    print(f"wrote {len(tensors)} tensors to {args.output}", flush=True)


if __name__ == "__main__":
    main()
