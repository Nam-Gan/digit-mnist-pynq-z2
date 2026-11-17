import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N_IN   = int(os.environ.get("N_IN", 784))
N_OUT  = int(os.environ.get("N_OUT", 64))
SHIFT  = int(os.environ.get("SHIFT", 12))
LAST   = int(os.environ.get("LAST", 0))
WFILE  = os.environ.get("W_FILE", "mem/l1_w.mem")
BFILE  = os.environ.get("B_FILE", "mem/l1_b.mem")
IMAGES = os.environ.get("IMAGES", "test_images.npy")
N_TEST = int(os.environ.get("N_TEST", 20))
OW     = 32 if LAST else 8


def load_weights():
    W = np.zeros((N_OUT, N_IN), dtype=np.int64)
    with open(WFILE) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == N_IN, f"{WFILE}: {len(lines)} lines, expected {N_IN}"
    for j, line in enumerate(lines):
        assert len(line) == N_OUT * 2, f"line {j} has {len(line)} digits"
        v = int(line, 16)
        for n in range(N_OUT):
            byte = (v >> (8 * n)) & 0xFF
            W[n, j] = byte - 256 if byte > 127 else byte
    return W


def load_bias():
    with open(BFILE) as f:
        vals = [int(l.strip(), 16) for l in f if l.strip()]
    assert len(vals) == N_OUT
    return np.array([v - (1 << 32) if v >= (1 << 31) else v for v in vals], dtype=np.int64)


def golden(W, b, x):
    acc = W @ x.astype(np.int64) + b
    return acc if LAST else np.clip(acc >> SHIFT, 0, 255)


async def input_buffer(dut, cur):
    """Registered-read model: in_data <= x[in_addr] each clock edge."""
    while True:
        await RisingEdge(dut.clk)
        addr = int(dut.in_addr.value)
        dut.in_data.value = int(cur[0][addr]) if addr < N_IN else 0


async def reset(dut):
    dut.nrst.value = 0
    dut.start.value = 0
    dut.in_data.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.nrst.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


def read_acc(dut):
    # cocotb 2.x: .to_signed(); on 1.x use .signed_integer
    return np.array([dut.neuron_gen[n].neuron_i.acc.value.to_signed()
                     for n in range(N_OUT)], dtype=np.int64)


def read_outputs(dut):
    flat = int(dut.activation_flat.value)
    mask = (1 << OW) - 1
    out = [(flat >> (OW * n)) & mask for n in range(N_OUT)]
    if LAST:
        out = [v - (1 << 32) if v >= (1 << 31) else v for v in out]
    return np.array(out, dtype=np.int64)


async def run_image(dut, timeout=N_IN + 100):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    acc = None
    for cycle in range(timeout):
        await RisingEdge(dut.clk)
        if acc is None and int(dut.state.value) == 3:     # STORE
            acc = read_acc(dut)
        if int(dut.ready_q.value) == 1:
            return cycle, acc
    raise AssertionError("timeout waiting for ready_q")


@cocotb.test()
async def test_layer(dut):
    W, b = load_weights(), load_bias()
    imgs = np.load(IMAGES).astype(np.int64)[:N_TEST]
    assert imgs.shape[1] == N_IN, f"{IMAGES}: input width {imgs.shape[1]}, expected {N_IN}"

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())   # cocotb 1.x: units="ns"
    cur = [imgs[0]]
    cocotb.start_soon(input_buffer(dut, cur))
    await reset(dut)

    bad = bad_acc = 0
    for k, x in enumerate(imgs):
        cur[0] = x
        await RisingEdge(dut.clk)

        lat, acc = await run_image(dut)
        got = read_outputs(dut)
        exp = golden(W, b, x)
        exp_acc = W @ x + b

        if acc is None or not np.array_equal(acc, exp_acc):
            bad_acc += 1
            idx = int(np.nonzero(acc != exp_acc)[0][0]) if acc is not None else -1
            dut._log.error(f"img {k}: raw acc mismatch, first n={idx}")
        if not np.array_equal(got, exp):
            bad += 1
            d = np.nonzero(got != exp)[0]
            dut._log.error(f"img {k}: {len(d)}/{N_OUT} outputs differ, "
                           f"first n={d[0]} got={got[d[0]]} exp={exp[d[0]]}")
        else:
            dut._log.info(f"img {k}: OK (latency {lat} cycles)")

        for _ in range(5):
            await RisingEdge(dut.clk)
        assert np.array_equal(read_outputs(dut), got), "outputs changed after ready"

    if LAST:
        dut._log.info(f"argmax matches golden on all images: "
                      f"{all(np.argmax(golden(W, b, x)) == np.argmax(golden(W, b, x)) for x in imgs)}")
    assert bad_acc == 0, f"{bad_acc}/{len(imgs)} images had wrong raw accumulators"
    assert bad == 0, f"{bad}/{len(imgs)} images had wrong outputs"
