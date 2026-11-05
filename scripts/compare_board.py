import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def compare_smolvla(outputs, references):
    values = []
    for path in sorted(outputs.glob("*.safetensors")):
        npu = load_file(path)["actions"]
        reference = load_file(references / path.name)["actions"][:npu.shape[0], :npu.shape[1]]
        values.append(cosine(npu, reference))
        print(f"{path.stem}: cos {values[-1]:.4f}, max |diff| {np.abs(npu - reference).max():.4f}")
    print(f"{len(values)} frames: cos mean {np.mean(values):.4f}, min {np.min(values):.4f}")


def matched(npu_boxes, npu_classes, reference_boxes, reference_classes, distance):
    count = 0
    for box, cls in zip(reference_boxes, reference_classes):
        near = np.linalg.norm(npu_boxes[:, :2] - box[:2], axis=1) < distance
        count += bool(np.any(near & (npu_classes == cls)))
    return count


def confident(cls, anchor):
    score = sigmoid(cls).max(axis=1)
    keep = score >= 0.5
    return anchor[keep], cls[keep].argmax(axis=1)


def compare_sparsedrive(outputs, references):
    plan_l2 = []
    final_l2 = []
    det = [0, 0, 0]
    agree = 0
    paths = sorted(outputs.glob("*.safetensors"))
    for path in paths:
        npu = load_file(path)
        reference = load_file(references / path.name)
        trajectories = np.cumsum(reference["plan_reg"].reshape(3, 6, 6, 2), axis=2)
        plan_l2.append(float(np.linalg.norm(npu["planning"] - trajectories, axis=-1).mean()))
        command = int(npu["cmd"].argmax())
        mode = int(npu["mode"])
        final_l2.append(float(np.linalg.norm(npu["final_planning"] - trajectories[command, mode], axis=-1).mean()))
        agree += int(npu["plan_cls"].reshape(3, 6)[command].argmax() ==
                     reference["plan_cls"].reshape(3, 6)[command].argmax())
        npu_boxes, npu_classes = confident(npu["det_cls"], npu["det_anchor"])
        reference_boxes, reference_classes = confident(reference["det_cls"], reference["det_anchor"])
        det[0] += len(reference_boxes)
        det[1] += len(npu_boxes)
        det[2] += matched(npu_boxes, npu_classes, reference_boxes, reference_classes, 1.0)
        print(f"{path.stem}: planning L2 {plan_l2[-1]:.3f} m, final L2 {final_l2[-1]:.3f} m, "
              f"detections {len(npu_boxes)} (reference {len(reference_boxes)})")
    print(f"{len(paths)} frames: planning L2 mean {np.mean(plan_l2):.3f} m, final L2 mean {np.mean(final_l2):.3f} m, "
          f"best mode agrees {agree}/{len(paths)}, detections {det[1]} (reference {det[0]}, "
          f"{det[2]} of them matched within 1 m with the same class)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("smolvla", "sparsedrive"))
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    if args.model == "smolvla":
        compare_smolvla(args.outputs, args.references)
    else:
        compare_sparsedrive(args.outputs, args.references)


if __name__ == "__main__":
    main()
