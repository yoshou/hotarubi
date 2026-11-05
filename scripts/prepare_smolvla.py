import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from safetensors.numpy import save_file

from quantize_smolvla import CAMERAS, MODEL, MODEL_REVISION, Dataset


def select_frames(dataset, count, seed):
    rng = np.random.default_rng(seed)
    episodes = sorted(e for e in dataset.episodes if e % 2 == 1)
    frames = []
    for _ in range(count):
        episode = int(rng.choice(episodes))
        length = dataset.episodes[episode]["length"]
        t = rng.uniform(0, length / dataset.fps - 1.0)
        frames.append((episode, min(int(round(t * dataset.fps)), length - 1)))
    return frames


def language_rows(policy, batch):
    tokens = batch[OBS_LANGUAGE_TOKENS]
    mask = batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()
    with torch.no_grad():
        rows = policy.model.vlm_with_expert.embed_language_tokens(tokens)
        rows = rows * math.sqrt(rows.shape[-1])
    return rows[0][mask].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("build/data/svla_so100_pickplace"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--lang-tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("build/board/smolvla"))
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    policy = SmolVLAPolicy.from_pretrained(MODEL, revision=MODEL_REVISION)
    policy.eval()
    policy.float()
    config = policy.config
    preprocess, _ = make_pre_post_processors(
        config, pretrained_path=MODEL, pretrained_revision=MODEL_REVISION,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})

    frames_dir = args.output / "frames"
    reference_dir = args.output / "reference"
    frames_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    language = {}
    for i, (episode, frame) in enumerate(select_frames(dataset, args.frames, args.seed)):
        row = dataset.row(episode, frame)
        images = {key: np.ascontiguousarray(dataset.image(episode, frame, source)) for key, source in CAMERAS.items()}
        batch = {key: torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 255.0
                 for key, image in images.items()}
        batch[OBS_STATE] = torch.tensor(row["observation.state"], dtype=torch.float32)[None]
        task = dataset.tasks[row["task_index"]]
        batch["task"] = task
        batch = preprocess(batch)
        rows = language_rows(policy, batch)
        if rows.shape[0] != args.lang_tokens:
            raise SystemExit(f"task {task!r} has {rows.shape[0]} tokens, not {args.lang_tokens}")
        if task in language and not np.array_equal(language[task], rows):
            raise SystemExit(f"task {task!r} gives different language rows")
        language[task] = rows
        prepared, masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        torch.manual_seed(i)
        noise = torch.randn(1, config.chunk_size, config.max_action_dim)
        with torch.no_grad():
            actions = policy.model.sample_actions(prepared, masks, batch[OBS_LANGUAGE_TOKENS],
                                                  batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=noise)
        save_file({"camera1": images["observation.images.camera1"],
                   "camera2": images["observation.images.camera2"],
                   "state": batch[OBS_STATE][0].numpy().astype(np.float32),
                   "noise": noise[0].numpy()},
                  frames_dir / f"{i:04d}.safetensors",
                  metadata={"task": task, "episode": str(episode), "frame": str(frame)})
        save_file({"actions": actions[0].numpy().astype(np.float32)}, reference_dir / f"{i:04d}.safetensors")
        print(f"frame {i}: episode {episode} frame {frame}", flush=True)
    save_file(language, args.output / "language.safetensors")
    print(f"wrote {args.frames} frames and {len(language)} tasks to {args.output}", flush=True)


if __name__ == "__main__":
    main()
