import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N_TEST = int(os.environ.get("N_TEST", 100))
TIMEOUT = 3000                                   # a classification takes ~900 cycles


# ---------- golden data ----------
def load_expected():
    with open("mem/test_expected.mem") as f:
        return np.array([int(l.strip(), 16) for l in f if l.strip()], dtype=np.int64)


def load_layer3():
    """Layer 3 weights/bias parsed back from the hex files (10 neurons, 32 inputs)."""
    n_out, n_in = 10, 32
    W = np.zeros((n_out, n_in), dtype=np.int64)
    with open("mem/l3_w.mem") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == n_in
    for j, line in enumerate(lines):
        v = int(line, 16)
        for n in range(n_out):
            byte = (v >> (8 * n)) & 0xFF
            W[n, j] = byte - 256 if byte > 127 else byte
    with open("mem/l3_b.mem") as f:
        vals = [int(l.strip(), 16) for l in f if l.strip()]
    b = np.array([v - (1 << 32) if v >= (1 << 31) else v for v in vals], dtype=np.int64)
    return W, b


# ---------- DUT helpers ----------
async def reset(dut):
    dut.nrst.value = 0
    dut.start.value = 0
    dut.wr_en.value = 0
    dut.wr_addr.value = 0
    dut.wr_data.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.nrst.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def write_image(dut, img):
    for i in range(784):
        dut.wr_en.value = 1
        dut.wr_addr.value = i
        dut.wr_data.value = int(img[i])
        await RisingEdge(dut.clk)               # RAM samples these values on this edge
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)                   # let the last write land before start


async def pulse_start(dut):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0


async def wait_done(dut):
    for cycle in range(TIMEOUT):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            return cycle
    raise AssertionError(f"timeout: done not seen within {TIMEOUT} cycles")


async def classify(dut, img):
    await write_image(dut, img)
    await pulse_start(dut)
    lat = await wait_done(dut)
    return int(dut.digit.value), lat


async def check_done_is_pulse(dut, cycles=20):
    """After a result, done must go low and stay low (no stuck or repeated pulses)."""
    for c in range(cycles):
        await RisingEdge(dut.clk)
        assert int(dut.done.value) == 0, f"done high again {c + 1} cycles after the result"


def flat_to_array(flat, n, width, signed):
    mask = (1 << width) - 1
    out = [(flat >> (width * k)) & mask for k in range(n)]
    if signed:
        out = [v - (1 << width) if v >= (1 << (width - 1)) else v for v in out]
    return np.array(out, dtype=np.int64)


# ---------- tests ----------
@cocotb.test()
async def test_full_classification(dut):
    imgs = np.load("test_images.npy").astype(np.int64)[:N_TEST]
    labels = np.load("test_labels.npy")[:N_TEST]
    expected = load_expected()[:N_TEST]
    l2_in = np.load("l2_in.npy").astype(np.int64)[:N_TEST]    # layer-1 outputs
    l3_in = np.load("l3_in.npy").astype(np.int64)[:N_TEST]    # layer-2 outputs
    W3, b3 = load_layer3()

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())   # cocotb 1.x: units="ns"
    await reset(dut)

    bad = bad_l1 = bad_l2 = bad_l3 = 0
    correct_vs_label = 0
    for k in range(len(imgs)):
        got, lat = await classify(dut, imgs[k])

        # layer outputs hold until the next start, so they can be read after done
        l1 = flat_to_array(int(dut.layer1.activation_flat.value), 64, 8, False)
        l2 = flat_to_array(int(dut.layer2.activation_flat.value), 32, 8, False)
        l3 = flat_to_array(int(dut.layer3.activation_flat.value), 10, 32, True)
        exp_l3 = W3 @ l3_in[k] + b3

        if not np.array_equal(l1, l2_in[k]):
            bad_l1 += 1
            dut._log.error(f"img {k}: layer 1 output differs ({int((l1 != l2_in[k]).sum())}/64 neurons)")
        if not np.array_equal(l2, l3_in[k]):
            bad_l2 += 1
            dut._log.error(f"img {k}: layer 2 output differs ({int((l2 != l3_in[k]).sum())}/32 neurons)")
        if not np.array_equal(l3, exp_l3):
            bad_l3 += 1
            dut._log.error(f"img {k}: layer 3 logits differ: got {l3.tolist()} exp {exp_l3.tolist()}")

        if got != expected[k]:
            bad += 1
            dut._log.error(f"img {k}: digit {got}, expected {expected[k]}")
        else:
            dut._log.info(f"img {k}: digit {got} OK (latency {lat} cycles)")
        correct_vs_label += int(got == labels[k])

        await check_done_is_pulse(dut)

    n = len(imgs)
    dut._log.info(f"RTL vs golden: {n - bad}/{n} match; vs true labels: {correct_vs_label}/{n}")
    assert bad_l1 == 0, f"{bad_l1} images with wrong layer 1 output"
    assert bad_l2 == 0, f"{bad_l2} images with wrong layer 2 output"
    assert bad_l3 == 0, f"{bad_l3} images with wrong layer 3 logits"
    assert bad == 0, f"{bad}/{n} digits differ from the golden model"


@cocotb.test()
async def test_back_to_back_and_repeat(dut):
    """Different images in a row, then the same image twice: no state may leak between runs."""
    imgs = np.load("test_images.npy").astype(np.int64)
    expected = load_expected()

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    order = [0, 1, 2, 3, 3, 2, 1, 0]
    results = []
    for idx in order:
        got, _ = await classify(dut, imgs[idx])
        assert got == expected[idx], f"image {idx}: got {got}, expected {expected[idx]}"
        results.append(got)
    assert results[3] == results[4] and results[0] == results[7]


@cocotb.test()
async def test_write_port_gating(dut):
    """Garbage on wr_addr/wr_data while wr_en=0 must not change the stored image."""
    imgs = np.load("test_images.npy").astype(np.int64)
    expected = load_expected()

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    await write_image(dut, imgs[5])
    rng = np.random.default_rng(1)
    for _ in range(50):                          # wiggle the buses with wr_en low
        dut.wr_addr.value = int(rng.integers(0, 784))
        dut.wr_data.value = int(rng.integers(0, 256))
        await RisingEdge(dut.clk)
    await pulse_start(dut)
    await wait_done(dut)
    assert int(dut.digit.value) == expected[5], "image was corrupted by writes with wr_en low"


@cocotb.test()
async def test_reset_mid_run(dut):
    """Resetting during a run must return to idle, and the next image must still be right."""
    imgs = np.load("test_images.npy").astype(np.int64)
    expected = load_expected()

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    await write_image(dut, imgs[7])
    await pulse_start(dut)
    for _ in range(300):                         # part-way through layer 1
        await RisingEdge(dut.clk)
    await reset(dut)

    got, _ = await classify(dut, imgs[8])
    assert got == expected[8], f"after reset: got {got}, expected {expected[8]}"
