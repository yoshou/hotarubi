import argparse
import hashlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import mx
import sparsedrive

CHECKPOINT_SHA256 = "a9786bd3398907666ef436b287b465d6de8c424467413648e4614e0b884db7ad"
FIRST_CONV = "img_backbone.conv1"
CACHE_SIZE = 4


def w8_linear(module):
    return module.in_features < 32 or module.out_features <= 64


def conv_columns(x, conv):
    kh, kw = conv.kernel_size
    cin = x.shape[0]
    if conv.kernel_size == (1, 1) and conv.padding == (0, 0):
        sh, sw = conv.stride
        return x[:, ::sh, ::sw].reshape(cin, -1).T
    cols = F.unfold(x[None], conv.kernel_size, dilation=conv.dilation, padding=conv.padding, stride=conv.stride)[0]
    return cols.reshape(cin, kh * kw, -1).permute(2, 1, 0).reshape(-1, kh * kw * cin)


def conv_matrix(weight):
    return weight.permute(0, 2, 3, 1).reshape(weight.shape[0], -1)


def first_conv_matrix(weight):
    return F.pad(weight, (0, 1)).reshape(weight.shape[0], -1)


class Collector:
    def __init__(self):
        self.hessians = {}
        self.rows = {}
        self.cache = []
        self.seconds = 0.0

    def gram(self, x, geometry, chunks):
        for tensor, version, key, value in self.cache:
            if tensor is x and version == x._version and key == geometry:
                return value
        gram = None
        rows = 0
        for chunk in chunks():
            chunk = chunk.to(torch.float64)
            product = chunk.T @ chunk
            gram = product if gram is None else gram + product
            rows += chunk.shape[0]
        value = (gram, rows)
        self.cache = [(x, x._version, geometry, value)] + self.cache[:CACHE_SIZE - 1]
        return value

    def add(self, name, k, gram, rows):
        if name not in self.hessians:
            self.hessians[name] = mx.Hessian(k)
        self.hessians[name].add(gram, rows)

    def linear_hook(self, name, gptq):
        def collect(module, inputs):
            start = time.perf_counter()
            x = inputs[0]
            k = x.shape[-1]
            self.rows[name] = self.rows.get(name, 0) + x.numel() // k
            if gptq:
                self.add(name, k, *self.gram(x, ("linear",), lambda: [x.reshape(-1, k)]))
            self.seconds += time.perf_counter() - start
        return collect

    def conv_hook(self, name, gptq):
        def collect(module, inputs):
            start = time.perf_counter()
            x = inputs[0]
            geometry = (module.kernel_size, module.stride, module.padding, module.dilation)
            if gptq:
                gram, rows = self.gram(x, geometry, lambda: (conv_columns(image, module) for image in x))
                self.add(name, gram.shape[0], gram, rows)
            else:
                rows = conv_columns(x[0], module).shape[0] * x.shape[0]
            self.rows[name] = self.rows.get(name, 0) + rows
            self.seconds += time.perf_counter() - start
        return collect


def relative_error(w, w_hat, h):
    d = mx.pad_k(w) - w_hat
    w = mx.pad_k(w)
    h = F.pad(h, (0, w.shape[1] - h.shape[0], 0, w.shape[1] - h.shape[0]))
    return float(torch.trace(d @ h @ d.T) / torch.trace(w @ h @ w.T))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes", type=Path, default=Path("build/data/nuscenes"))
    parser.add_argument("--checkpoint", type=Path, default=Path("build/sparsedrive/sparsedrive_stage2.pth"))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("build/quant/sparsedrive.safetensors"))
    args = parser.parse_args()

    digest = sha256(args.checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"{args.checkpoint}: sha256 {digest}, expected {CHECKPOINT_SHA256}")
    model = sparsedrive.load_model(args.checkpoint)
    folded = set(sparsedrive.fold_batchnorm(model))
    nusc = sparsedrive.NuScenes(args.nuscenes)
    frames = [(scene, token) for scene in sparsedrive.MINI_TRAIN for token in nusc.scene_samples(scene)]
    frames = frames[:args.frames]

    collector = Collector()
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_pre_hook(collector.conv_hook(name, name != FIRST_CONV)))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_pre_hook(collector.linear_hook(name, not w8_linear(module))))
    start = time.perf_counter()
    scene = None
    for i, (frame_scene, token) in enumerate(frames):
        if frame_scene != scene:
            model.reset()
            scene = frame_scene
        img, metas = sparsedrive.model_inputs(nusc.frame(token))
        hessian_start = collector.seconds
        frame_start = time.perf_counter()
        with torch.no_grad():
            model(img, metas)
        collector.cache = []
        hessian = collector.seconds - hessian_start
        print(f"frame {i}: {scene} sample {token} (forward {time.perf_counter() - frame_start - hessian:.1f} s, "
              f"hessian {hessian:.1f} s) ({time.perf_counter() - start:.0f} s)", flush=True)
    for hook in hooks:
        hook.remove()

    tensors = {}
    start = time.perf_counter()
    for name, module in model.named_modules():
        if isinstance(module, sparsedrive.Attention):
            q, scale = mx.mxint8(module.in_proj_weight.detach())
            tensors[f"{name}.in_proj_weight.mxint8"] = q
            tensors[f"{name}.in_proj_weight.scale"] = scale
            print(f"{name}.in_proj_weight: mxint8 {tuple(q.shape)}", flush=True)
            continue
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        if collector.rows.get(name, 0) == 0:
            print(f"{name}: not run", flush=True)
            continue
        weight = module.weight.detach()
        if isinstance(module, torch.nn.Conv2d):
            weight = first_conv_matrix(weight) if name == FIRST_CONV else conv_matrix(weight)
        if name in collector.hessians:
            h = collector.hessians[name].value()
            codes, scale = mx.gptq_mxfp4(weight, h)
            tensors[f"{name}.weight.mxfp4"] = codes
            tensors[f"{name}.weight.scale"] = scale
            rtn_codes, rtn_scale = mx.mxfp4(weight)
            gptq = relative_error(weight.float(), mx.mxfp4_value(codes, scale), h)
            rtn = relative_error(weight.float(), mx.mxfp4_value(rtn_codes, rtn_scale), h)
            print(f"{name}: mxfp4 {tuple(codes.shape)} error {gptq:.5f} (rtn {rtn:.5f}) "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
        else:
            q, scale = mx.mxint8(weight)
            tensors[f"{name}.weight.mxint8"] = q
            tensors[f"{name}.weight.scale"] = scale
            print(f"{name}: mxint8 {tuple(q.shape)}", flush=True)
        if name in folded:
            tensors[f"{name}.bias"] = module.bias.detach().clone()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={
        "model": "SparseDrive-S", "code": sparsedrive.CODE, "checkpoint_sha256": digest,
        "dataset": "nuScenes v1.0-mini mini_train", "scenes": ",".join(sparsedrive.MINI_TRAIN),
        "frames": str(len(frames))})
    print(f"wrote {len(tensors)} tensors to {args.output}", flush=True)


if __name__ == "__main__":
    main()
