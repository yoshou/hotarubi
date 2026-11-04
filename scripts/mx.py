import torch

BLOCK = 32
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def pad_k(w):
    k = w.shape[-1]
    padded = -(-k // BLOCK) * BLOCK
    return torch.nn.functional.pad(w, (0, padded - k)) if padded != k else w


def exponent_field(amax):
    bits = amax.to(torch.float32).contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) - 127


def block_amax(w):
    n, k = w.shape
    return w.abs().reshape(n, k // BLOCK, BLOCK).amax(dim=2)


def e2m1_code(v):
    magnitude = v.abs()
    lower = torch.searchsorted(E2M1_MIDPOINTS, magnitude.contiguous(), right=False)
    tie = (lower < len(E2M1_MIDPOINTS)) & (
        E2M1_MIDPOINTS[lower.clamp(max=len(E2M1_MIDPOINTS) - 1)] == magnitude)
    code = torch.where(tie & (lower % 2 == 1), lower + 1, lower)
    sign = (v < 0) & (code != 0)
    return (code + torch.where(sign, 8, 0)).to(torch.uint8)


def e2m1_value(code):
    code = code.to(torch.int64)
    value = E2M1_VALUES[code & 7]
    return torch.where((code & 8) != 0, -value, value)


def mxint8(w):
    w = pad_k(w.to(torch.float32))
    n, k = w.shape
    amax = block_amax(w)
    e = torch.where(amax > 0, exponent_field(amax), torch.zeros_like(amax, dtype=torch.int32))
    shift = torch.exp2((6 - e).to(torch.float32)).repeat_interleave(BLOCK, dim=1)
    q = torch.round(w * shift).clamp(-128, 127).to(torch.int8)
    return q, (e + 127).to(torch.uint8)


def mxfp4(w):
    w = pad_k(w.to(torch.float32))
    amax = block_amax(w)
    e = torch.where(amax > 0, exponent_field(amax) - 2, torch.zeros_like(amax, dtype=torch.int32))
    scale = torch.exp2(e.to(torch.float32)).repeat_interleave(BLOCK, dim=1)
    return e2m1_code(w / scale), (e + 127).to(torch.uint8)


def mxint8_value(q, scale):
    e = scale.to(torch.int32) - 127
    return q.to(torch.float32) * torch.exp2((e - 6).to(torch.float32)).repeat_interleave(BLOCK, dim=1)


def mxfp4_value(code, scale):
    e = scale.to(torch.int32) - 127
    return e2m1_value(code) * torch.exp2(e.to(torch.float32)).repeat_interleave(BLOCK, dim=1)


class Hessian:
    def __init__(self, k):
        self.sum = torch.zeros(k, k, dtype=torch.float64)
        self.rows = 0

    def add(self, gram, rows):
        self.sum += gram
        self.rows += rows

    def value(self):
        return (self.sum / self.rows).to(torch.float32)


def gptq_mxfp4(w, h, blocksize=128, percdamp=0.01):
    w = pad_k(w.to(torch.float32)).clone()
    n, k = w.shape
    h = torch.nn.functional.pad(h.to(torch.float32), (0, k - h.shape[0], 0, k - h.shape[0])).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += torch.eye(k) * (percdamp * torch.mean(torch.diag(h)))
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = torch.zeros(n, k, dtype=torch.uint8)
    scales = torch.zeros(n, k // BLOCK, dtype=torch.uint8)
    scale = None
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % BLOCK == 0:
                amax = w1[:, i:i + BLOCK].abs().amax(dim=1)
                e = torch.where(amax > 0, exponent_field(amax) - 2, torch.zeros_like(amax, dtype=torch.int32))
                scales[:, col // BLOCK] = (e + 127).to(torch.uint8)
                scale = torch.exp2(e.to(torch.float32))
            code = e2m1_code(w1[:, i] / scale)
            codes[:, col] = code
            q = e2m1_value(code) * scale
            err = (w1[:, i] - q) / hinv1[i, i]
            w1[:, i:] -= err[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = err
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes, scales
