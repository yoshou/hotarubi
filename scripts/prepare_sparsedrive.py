import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

import sparsedrive

MINI_VAL = ("scene-0103", "scene-0916")


def last(outputs, key):
    return outputs[key][-1][0].numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes", type=Path, default=Path("build/data/nuscenes"))
    parser.add_argument("--checkpoint", type=Path, default=Path("build/sparsedrive/sparsedrive_stage2.pth"))
    parser.add_argument("--scenes", nargs="+", default=list(MINI_VAL))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("build/board/sparsedrive"))
    args = parser.parse_args()

    model = sparsedrive.load_model(args.checkpoint)
    model.eval()
    nusc = sparsedrive.NuScenes(args.nuscenes)
    frames = [(scene, token) for scene in args.scenes for token in nusc.scene_samples(scene)][:args.frames]
    frames_dir = args.output / "frames"
    reference_dir = args.output / "reference"
    frames_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    scene = None
    for i, (frame_scene, token) in enumerate(frames):
        if frame_scene != scene:
            model.reset()
            scene = frame_scene
        frame = nusc.frame(token)
        img, metas = sparsedrive.model_inputs(frame)
        with torch.no_grad():
            det, map_output, motion, planning = model(img, metas)
        save_file({"img": frame["img"].astype(np.float32),
                   "projection_mat": frame["projection_mat"].astype(np.float32),
                   "image_wh": frame["image_wh"].astype(np.float32),
                   "T_global": frame["T_global"].astype(np.float32),
                   "T_global_inv": frame["T_global_inv"].astype(np.float32),
                   "timestamp_us": np.array([nusc.samples[token]["timestamp"]], dtype=np.int64),
                   "cmd": frame["gt_ego_fut_cmd"].astype(np.float32)},
                  frames_dir / f"{i:04d}.safetensors",
                  metadata={"token": token, "scene_token": scene})
        save_file({"det_cls": last(det, "classification"),
                   "det_anchor": last(det, "prediction"),
                   "map_cls": last(map_output, "classification"),
                   "map_anchor": last(map_output, "prediction"),
                   "motion_cls": last(motion, "classification"),
                   "motion_reg": last(motion, "prediction"),
                   "plan_cls": last(planning, "classification"),
                   "plan_reg": last(planning, "prediction"),
                   "plan_status": last(planning, "status"),
                   "gt_ego_fut_trajs": frame["gt_ego_fut_trajs"].astype(np.float32)},
                  reference_dir / f"{i:04d}.safetensors",
                  metadata={"token": token, "scene_token": scene})
        print(f"frame {i}: {scene} sample {token}", flush=True)
    print(f"wrote {len(frames)} frames to {args.output}", flush=True)


if __name__ == "__main__":
    main()
