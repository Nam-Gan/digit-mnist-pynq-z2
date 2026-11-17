# qat.py
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

def ste_round(x): return (x.round() - x).detach() + x
def ste_floor(x): return (x.floor() - x).detach() + x

class QLinear(nn.Module):
    def __init__(s, n_in, n_out, last=False):
        super().__init__()
        s.fc, s.last = nn.Linear(n_in, n_out), last
        s.ew = s.ea_in = s.ea_out = 0          # integer exponents, frozen after calibration

    @property
    def shift(s): return s.ea_out - (s.ea_in + s.ew)

    def int_params(s):
        sw = 2.0 ** s.ew
        Wq = torch.clamp(ste_round(s.fc.weight / sw), -128, 127)
        bq = ste_round(s.fc.bias / 2.0 ** (s.ea_in + s.ew))
        return Wq, bq

    def forward(s, a):                          # a = dequantized input (int * 2^ea_in)
        Wq, bq = s.int_params()
        z = F.linear(a, Wq * 2.0 ** s.ew, bq * 2.0 ** (s.ea_in + s.ew))
        if s.last: return z
        sa = 2.0 ** s.ea_out
        return torch.clamp(ste_floor(z / sa), 0, 255) * sa   # floor + clamp == RTL >>> and saturate

class MLP(nn.Module):
    def __init__(s, dims=(784, 64, 32, 10)):
        super().__init__()
        s.layers = nn.ModuleList(
            QLinear(i, o, last=(k == len(dims) - 2))
            for k, (i, o) in enumerate(zip(dims[:-1], dims[1:])))
        s.quant = False                          # float warm-up first

    def forward(s, raw):                         # raw = 0..255 pixels, (N,784)
        a = raw / 256.0
        for L in s.layers:
            if s.quant: a = L(a)
            else:
                a = L.fc(a)
                if not L.last: a = F.relu(a)
        return a

@torch.no_grad()
def calibrate(model, X):                        # fix all exponents from float model stats
    a, ea_in = X / 256.0, -8
    for L in model.layers:
        L.ea_in = ea_in
        L.ew = int(np.ceil(np.log2(L.fc.weight.abs().max().item() / 127)))
        z = L.fc(a)
        if not L.last:
            a = F.relu(z)
            ea = int(np.ceil(np.log2(a.max().item() / 255)))
            L.ea_out = max(ea, L.ea_in + L.ew)   # forces shift >= 0
            ea_in = L.ea_out
    model.quant = True

def get_data():
    tr = datasets.MNIST('data', train=True,  download=True)
    te = datasets.MNIST('data', train=False, download=True)
    f = lambda d: (d.data.float().view(-1, 784), d.targets)
    return f(tr), f(te)

if __name__ == "__main__":
    (Xtr, ytr), (Xte, yte) = get_data()
    m = MLP(); opt = torch.optim.Adam(m.parameters(), 1e-3)

    def run(epochs, lr):
        for g in opt.param_groups: g['lr'] = lr
        for ep in range(epochs):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 128):
                idx = perm[i:i+128]
                loss = F.cross_entropy(m(Xtr[idx]), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            acc = (m(Xte).argmax(1) == yte).float().mean().item()
            print(f"ep {ep} acc {acc:.4f}")

    run(3, 1e-3)                 # 1) float warm-up
    calibrate(m, Xtr[:5000])     # 2) freeze exponents
    run(8, 3e-4)                 # 3) QAT fine-tune
    torch.save(m.state_dict(), "mlp_qat.pt")
    print("shifts:", [L.shift for L in m.layers])
    torch.save({"sd": m.state_dict(),
            "exp": [(L.ew, L.ea_in, L.ea_out) for L in m.layers]},
           "mlp_qat.pt")
    print("exponents (ew, ea_in, ea_out):", [(L.ew, L.ea_in, L.ea_out) for L in m.layers])
