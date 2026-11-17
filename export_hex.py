# export_hex.py
import torch, numpy as np, os
from qat import MLP, calibrate, get_data
(Xtr, ytr), (Xte, yte) = get_data()          # must come first: defines Xte, yte


ck = torch.load("mlp_qat.pt")
m = MLP()
m.load_state_dict(ck["sd"])
for L, (ew, ai, ao) in zip(m.layers, ck["exp"]):
    L.ew, L.ea_in, L.ea_out = ew, ai, ao
m.quant = True

# sanity check: float QAT model must match the integer model
with torch.no_grad():
    md = m.double()
    qat_acc = (md(Xte.double()).argmax(1) == yte).float().mean().item()
print("QAT (float sim) accuracy:", qat_acc)

P = []
for i, L in enumerate(m.layers, 1):
    Wq, bq = [t.detach().numpy().astype(np.int64) for t in L.int_params()]
    P.append((Wq, bq, L.shift, L.last))
    with open(f"mem/l{i}_w.mem", "w") as f:       # 1 line per input; neuron 0 = lowest byte
        for j in range(Wq.shape[1]):
            f.write("".join(f"{int(Wq[n, j]) & 0xFF:02x}"
                            for n in reversed(range(Wq.shape[0]))) + "\n")
    with open(f"mem/l{i}_b.mem", "w") as f:
        f.writelines(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in bq)
    print(f"L{i}: shift={L.shift}, W range [{Wq.min()},{Wq.max()}]")

def golden(x):                              # x: (784,) ints 0..255; mirrors the RTL exactly
    a = x.astype(np.int64)
    for Wq, bq, sh, last in P:
        acc = Wq @ a + bq
        a = acc if last else np.clip(acc >> sh, 0, 255)   # numpy >> on int64 = floor
    return a

X = Xte.numpy().astype(np.int64); y = yte.numpy()
pred = np.array([golden(x).argmax() for x in X])
print("integer-model accuracy:", (pred == y).mean())      # should match the QAT float accuracy

# test vectors for simulation and the board
N = 100
np.save("test_images.npy", X[:N].astype(np.uint8)); np.save("test_labels.npy", y[:N])
with open("mem/test_images.mem", "w") as f:
    for x in X[:N]: f.writelines(f"{int(v):02x}\n" for v in x)
with open("mem/test_expected.mem", "w") as f:
    for x in X[:N]: f.write(f"{int(golden(x).argmax()):02x}\n")
def forward_all(x):
    a = x.astype(np.int64)
    ins = []
    for Wq, bq, sh, last in P:
        ins.append(a)                              # this layer's input
        acc = Wq @ a + bq
        a = acc if last else np.clip(acc >> sh, 0, 255)
    return ins

allins = [forward_all(x) for x in X[:N]]
np.save("l2_in.npy", np.array([t[1] for t in allins]).astype(np.uint8))   # layer-1 outputs
np.save("l3_in.npy", np.array([t[2] for t in allins]).astype(np.uint8))   # layer-2 outputs
print("saved l2_in.npy", np.load("l2_in.npy").shape, "l3_in.npy", np.load("l3_in.npy").shape)
