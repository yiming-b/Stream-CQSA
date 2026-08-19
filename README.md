# Stream-CQSA

**Exact attention that still returns an answer where FlashAttention-2 runs out of memory.**

Stream-CQSA decomposes one attention call into a set of smaller ones using a
*cyclic quorum set* (CQS) over the sequence, runs them one at a time, and
recomposes their local softmax statistics into the exact global result. The
decomposition is a partition of the query–key pairs, so nothing is dropped,
sampled, or approximated: the output is the attention function, not a surrogate
for it.

The practical consequence is a knob that trades wall-clock time for peak device
memory, and keeps turning after the monolithic kernel has already failed.

📄 **Paper:** [arXiv:2604.20819](https://doi.org/10.48550/arXiv.2604.20819) —
*Stream-CQSA: Avoiding Out-of-Memory in Attention Computation via Flexible
Workload Scheduling*

---

## Results

Causal attention, `B=1 H=8 D=64`, fp16, NVIDIA A100 80GB. Each cell is
**wall-clock / peak device memory**, where peak is
`torch.cuda.max_memory_allocated()`.

Three Stream-CQSA configurations appear, and the distinction matters:

| column | decomposition depth | fp32 accumulator | what it is for |
|---|---|---|---|
| `itr=1` | **fixed** at 1 | **on GPU** | tracing the trade curve at a pinned depth |
| `itr=2` | **fixed** at 2 | **on GPU** | as above, one step deeper |
| `itr=auto`, acc=CPU | **automatic** | **on host** | **the configuration you should actually use** |

The two fixed-depth columns are measurements, not recommendations — they exist
so the time/memory trade is visible as a curve. The third column is the shipped
default (`stream_cqsa_auto`): it picks the depth itself and moves the fp32
accumulator off the device, which removes the one O(N) device term that no
amount of decomposition can shrink. **Those runs are still in the queue**; the
cells are marked *pending* rather than estimated.

In normal use you never pick a depth yourself — see
[Let it choose the depth](#let-it-choose-the-depth).

### Backward

| N | SDPA | SDPA (mem-eff.) | FlashAttention-2 | Stream-CQSA `itr=1` acc=GPU | Stream-CQSA `itr=2` acc=GPU | Stream-CQSA `itr=auto` acc=CPU |
|---|---|---|---|---|---|---|
| **1M**  | 26 s / 12.1 GiB | 78 s / 11.1 GiB | 25 s / 10.1 GiB | 57 s / 7.6 GiB | 68 s / **4.4 GiB** | *pending* |
| **2M**  | 106 s / 24.1 GiB | 3 860 s / 22.1 GiB | 101 s / 20.1 GiB | 218 s / 15.3 GiB | 262 s / **8.9 GiB** | *pending* |
| **4M**  | 429 s / 48.3 GiB | 15 421 s / 44.3 GiB | 407 s / 40.3 GiB | 854 s / 30.5 GiB | 998 s / **17.7 GiB** | *pending* |
| **8M**  | **OOM** | **OOM** | **OOM** | 3 372 s / 61.1 GiB | 3 895 s / **35.5 GiB** | *pending* |
| **16M** | **OOM** | **OOM** | **OOM** | **OOM** | *not run* | *pending* |

**This is where the method pays off.** At 8M every baseline is out of memory and
Stream-CQSA finishes, on the same card with the same numerics. And it is not
only a fallback: at 4M it does the backward in 17.7 GiB against
FlashAttention-2's 40.3 GiB — **2.3× less peak memory** — because raising the
depth shrinks the in-flight working set while the inputs stream from host
memory. The price is **2.1–2.7×** the time.

At 16M even the acc=GPU configuration runs out: the fp32 accumulator is itself
O(N) on the device. That is exactly the term the acc=CPU column removes, and
why it is the last column standing.

### Forward

| N | SDPA | SDPA (mem-eff.) | FlashAttention-2 | Stream-CQSA `itr=1` acc=GPU | Stream-CQSA `itr=2` acc=GPU | Stream-CQSA `itr=auto` acc=CPU |
|---|---|---|---|---|---|---|
| **1M**  | 8 s / 5.0 GiB | 15 s / 5.0 GiB | 8 s / 5.0 GiB | 14 s / 6.5 GiB | 15 s / **4.2 GiB** | *pending* |
| **2M**  | 32 s / 10.1 GiB | 61 s / 10.0 GiB | 32 s / 10.1 GiB | 53 s / 13.0 GiB | 64 s / **8.3 GiB** | *pending* |
| **4M**  | 131 s / 20.1 GiB | 245 s / 20.0 GiB | 130 s / 20.1 GiB | 203 s / 25.9 GiB | 237 s / **16.7 GiB** | *pending* |
| **8M**  | 520 s / 40.3 GiB | 966 s / 40.0 GiB | 532 s / 40.3 GiB | 796 s / 51.8 GiB | 907 s / **33.3 GiB** | *pending* |
| **16M** | **OOM** | **OOM** | **OOM** | **OOM** | *not run* | *pending* |

**The forward story is more modest, and worth stating plainly.** In the acc=GPU
configuration Stream-CQSA does *not* extend the forward's OOM boundary: every
method here reaches 8M and fails at 16M. What it buys is headroom at a given N
— 33.3 GiB against FlashAttention-2's 40.3 GiB at 8M (**0.83×**) — for
**1.5–2.0×** the time. At `itr=1` it actually uses *more* memory than the
baselines (1.29×), because the fp32 accumulator is an extra O(N) device term.

The reason is structural: the forward's peak is dominated by the inputs and the
accumulator, both O(N) on the device, and neither shrinks with depth. Moving the
accumulator to the host is what should carry the forward past 16M — the pending
column.

![memory and wall-clock across N](docs/figures/fig_mem_time_fp16.jpg)

### What it costs

Below the OOM boundary, Stream-CQSA is **slower and, in the forward, uses more
device memory** than the baselines it is meant to rescue. That is the trade, and
the repository does not hide it:

Ratios against FlashAttention-2 over N = 1M–8M, fp16 (min–max across N):

| | | time | peak memory |
|---|---|--:|--:|
| forward | `itr=1` | 1.49–1.81× | 1.29× |
| forward | `itr=2` | 1.70–1.99× | **0.83×** |
| backward | `itr=1` | 2.10–2.26× | 0.76× |
| backward | `itr=2` | 2.45–2.70× | **0.44×** |

The memory ratios are strikingly constant across N — the decomposition shrinks
the working set by a fixed factor set by `itr`, not by anything N-dependent.

**Use FlashAttention-2 when it fits.** Reach for Stream-CQSA when it does not,
or when you need the backward's memory headroom more than you need the 2×.

One incidental finding worth flagging: PyTorch's **memory-efficient SDPA backend
is a severe outlier in the backward** — 15 421 s at 4M against FlashAttention-2's
407 s, a factor of 38 — while using *more* memory than Stream-CQSA `itr=2`. If
you are reaching for that backend for long-context training, measure it.

### Accuracy

Every number is measured **against a float64 reference**, not against SDPA —
SDPA is one of the methods under test, not the yardstick. N=8192, 10 seeds.

**Forward** — relative error vs float64:

| method | fp16 | bf16 |
|---|--:|--:|
| SDPA | 2.689e-04 | 2.168e-03 |
| FlashAttention-2 | 2.689e-04 | 2.168e-03 |
| **Stream-CQSA `itr=1`** | **1.705e-04** | **1.375e-03** |
| **Stream-CQSA `itr=2`** | **1.681e-04** | **1.356e-03** |

**Backward** — relative error vs a *dense* float64 reference:

| method | fp16 | bf16 |
|---|--:|--:|
| SDPA | 3.075e-04 | 2.458e-03 |
| FlashAttention-2 | 3.075e-04 | 2.458e-03 |
| Stream-CQSA `itr=1` | 3.080e-04 | 2.462e-03 |
| Stream-CQSA `itr=2` | 3.085e-04 | 2.465e-03 |

Three claims this supports:

- **Decomposition costs no accuracy.** Backward error is flat in `itr` to three
  significant figures; error does **not** compound as subproblems are merged.
- **The forward is ~1.6× *more* accurate than the baselines.** Not a rounding
  artefact: Stream-CQSA's statistics and output path are fp32 while the
  baselines round the output to fp16.
- **The floor is the input dtype, not the method.** bf16/fp16 = 8.0–8.1×
  everywhere, exactly the mantissa ratio (10 vs 7 explicit bits, 2³ = 8).

![accuracy across precision](docs/figures/fig_accuracy.jpg)

> **One caveat, stated plainly.** When attention is close to uniform, the
> backward's `dS = P(dP − D)` suffers catastrophic cancellation, and
> decomposition amplifies it (each subproblem's `dP` is further from the global
> `D` it is differenced against). At an input scale of 0.05 the `dQ` error
> drifts 3.5e-03 → 8.5e-03 from `itr=0` to `itr=2`. At realistic score
> magnitudes (scale ≥ 0.5) the drift vanishes entirely, and the undecomposed
> baseline is *already* 11× degraded in that regime — it is a property of the
> input, not of the method. Details in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## Layout

```
stream_cqsa/            the package
  stable_stream.py        production entry points: forward, backward, scheduler,
                          chunk pool -- this is where the fast path lives
  oom_fallback.py         stream_cqsa_auto and the 7-rung escalation ladder
  cqs_mask.py             CQS mask construction and pair-coverage validation
  reference.py            readable ground-truth implementation, used by the tests
  backends/exact/         pure-PyTorch reference kernels (slow, inspectable)
  baselines/longnet/      LongNet dilated-attention pattern, decomposed the same way

csrc/                   CUDA extension
  flash_attn/             derived from FlashAttention-2; 11 files carry the CQS
                          masking and block-skipping changes
  cutlass/                vendored NVIDIA CUTLASS headers (4.3.4)

tests/                  correctness suite (243 tests)
examples/quickstart.py  smallest end-to-end example
notebooks/
  reference_kernels_demo.ipynb   how the decomposition works, from first principles
  tutorial_stream_cqsa.ipynb     production API, profiling, simulating a smaller card
  accuracy_demo.ipynb            accuracy vs FlashAttention-2 and SDPA, fp16/bf16
benchmarks/             experiment harness, report and figure generators
results/paper/          raw JSONL backing every published number
docs/
  METHODOLOGY.md          the numerical argument and the CUDA-level work
  RESULTS.md              full tables, generated from results/ -- do not hand-edit
  figures/                generated figures (jpg + svg)
third_party/            upstream BSD-3 license texts
```

## Installation


There are **no prebuilt wheels**: the CUDA extension is compiled for the GPU you
actually have, which keeps compile time and binary size sane.

### Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability **≥ 8.0** (Ampere or newer: A100, A6000, L40S, H100, …) |
| CUDA toolkit | `nvcc` must be present. It does **not** need to be on `PATH` — PyTorch looks in `/usr/local/cuda` — but its major version should match your PyTorch build. |
| PyTorch | any recent version, built for **your** CUDA. Install it *before* this package. |
| Python | ≥ 3.9 |
| build | `ninja` (installed below); a C++17 host compiler |

Check what you have:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # need compute_cap >= 8.0
nvcc --version || ls /usr/local/cuda/bin/nvcc          # need one of these
```

### Steps

```bash
# 1. PyTorch first, matching your CUDA (cu126 / cu128 / cu130 / ...)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install ninja

# 2. this package
git clone https://github.com/yiming-b/Stream-CQSA.git
cd Stream-CQSA
pip install -e . --no-build-isolation
```

**`--no-build-isolation` is not optional.** The extension compiles against the
PyTorch you already have. With build isolation pip downloads a *second* torch
into a throwaway environment, compiles against that ABI, and the result fails to
import against yours. This is the same constraint FlashAttention ships under,
for the same reason.

Expect **roughly half an hour** — measured at **33 minutes** with `MAX_JOBS=16`
on a 80-core node. The last few backward kernels dominate and are largely
serial in `ptxas`, so more cores help less than you would hope. `MAX_JOBS` caps
parallel compilation; lower it if the build is killed for memory (each `nvcc`
can take ~2 GB):

```bash
MAX_JOBS=16 pip install -e . --no-build-isolation
```

### Verify

```bash
python examples/quickstart.py     # end-to-end against SDPA, ~10 s

pip install pytest                # not a runtime dependency
pytest tests/ -q                  # 243 tests, ~1 min
```

`quickstart.py` prints the escalation rung that was chosen and the relative
error against SDPA; anything near `1e-04` in fp16 is correct.

<details>
<summary>Known-good configuration</summary>

These exact steps were rehearsed end-to-end in a clean virtualenv against a
fresh clone of this repository:

| | |
|---|---|
| GPU / driver | A100-PCIE-40GB, driver 610.57.04 |
| CUDA toolkit | nvcc 13.3 (at `/usr/local/cuda`, not on `PATH`) |
| PyTorch | 2.13.0+cu130, installed from the cu130 index |
| Python | 3.11 |
| build | `MAX_JOBS=16`, 33 min, exit 0 |
| result | `quickstart.py` rel. err 1.447e-04; **243 tests passed** |

The extension also builds and runs against torch 2.10.0+cu130, so it is not
pinned to one PyTorch release.

</details>

### Building where the GPU is not visible

`setup.py` reads the target architecture from
`torch.cuda.get_device_capability()`. **On a cluster login node, or in a
container with no GPU attached, there is nothing to read** — it prints a warning
and falls back to `sm_80`, which will produce a build that does not run on, say,
an H100. Set the architecture explicitly whenever you build somewhere other than
the machine you will run on:

```bash
FLASH_ATTN_CUDA_ARCHS="90" pip install -e . --no-build-isolation   # H100
FLASH_ATTN_CUDA_ARCHS="80;90" pip install -e . --no-build-isolation  # fat binary
```

`80` = A100, `86` = A6000/3090, `89` = L40S/4090, `90` = H100, `100` = B200.

### Choosing the kernel set

The default builds fp16 + bf16 × head-dim 64 and 128 × causal and non-causal,
which covers most transformers. Override with `CQSA_KERNEL_SET`:

| value | forward head dims | backward head dims | when |
|---|---|---|---|
| `common` *(default)* | 64, 128 | 64, 128 | almost always |
| `full` | 32, 64, 96, 128, 192, 256 | 64, 128 | you need a wide-head **forward** |
| `a100_fp16_hdim128` | 128 (fp16 only) | 128 | fast iteration while developing |

Both sets carry fp16 **and** bf16, causal and non-causal.

```bash
CQSA_KERNEL_SET=full pip install -e . --no-build-isolation
```

> **The backward is head-dim 64 and 128 only, in every kernel set** — those are
> the only backward kernels in the tree. `full` widens the forward, not the
> backward. If you need to *train* at head-dim 96 or 256, this package cannot do
> it yet.

(`a100_fp16_hdim64_128` and `a100_fp16_hdim64_noncau` also exist; both are
narrow development builds, and the latter is the one to avoid — it is fp16,
head-dim 64, **non-causal only**.)

### If it goes wrong

| symptom | cause |
|---|---|
| `undefined symbol` / `ImportError` on `import stream_cqsa` | built against a different torch — rebuild with `--no-build-isolation`, or you upgraded torch after building |
| `no kernel image is available for execution` | built for the wrong arch; set `FLASH_ATTN_CUDA_ARCHS` and rebuild |
| `this CQSA build only supports head_dim=64 or 128` | rebuild with `CQSA_KERNEL_SET=full` |
| `nvcc: not found` / `CUDA_HOME` unset | install the CUDA toolkit, or `export CUDA_HOME=/usr/local/cuda` |
| build killed | lower `MAX_JOBS` |

---

## Usage


One function. It takes the place of `scaled_dot_product_attention` and returns
the same thing.

```python
import torch
from stream_cqsa import stream_cqsa_auto

q, k, v = (torch.randn(1, 8, 1_000_000, 64, device="cuda", dtype=torch.float16)
           for _ in range(3))

out = stream_cqsa_auto(q, k, v, causal=True)
```

`stream_cqsa_auto` is the *"just run it"* path: it walks an escalation ladder
cheapest-first and returns the first configuration that fits.

It tries these seven configurations **in order**, top to bottom, and returns the
first one that completes without running out of memory:

| # | `itr` (depth) | Q/K/V live on | fp32 accumulator lives on |
|:--:|:--:|---|---|
| 1 | 1 | device | device |
| 2 | 2 | device | device |
| 3 | 1 | **host** | device |
| 4 | 2 | **host** | device |
| 5 | 2 | **host** | **host** |
| 6 | 3 | **host** | **host** |
| 7 | 4 | **host** | **host** |

Reading down the table, each step relaxes exactly one constraint and buys device
memory at the cost of time:

- **raising `itr`** splits the work into more, smaller subproblems (`c^itr`
  of them, so 7 → 49 → 343), shrinking the in-flight working set;
- **moving Q/K/V to the host** removes the largest O(N) device term, paging each
  subsequence in on demand;
- **moving the accumulator to the host** removes the *other* O(N) device term —
  the one that no amount of decomposition can shrink.

Because it stops at the first rung that works, you pay only for the headroom you
actually need. `info["config"]` (or `return_info=True`) reports which rung was
used.

### Let it choose the depth

**Do not set `itr` yourself.** It defaults to `"auto"`, and automatic depth is
the intended way to use this library — the whole point is that you should not
have to know what a decomposition depth is to get a correct result.

There are two levels of automatic, and they fail differently:

| | how it decides | when it is right |
|---|---|---|
| `stream_cqsa_auto(...)` | **runs, catches the OOM, escalates, retries** | the robust default — use this |
| `stream_cqsa_forward(..., itr="auto")` | plans once from free device memory | you want a single call with no retry |

The difference matters. The planner reads *device-wide* free memory, so if
something else on the card takes memory after it plans — or if you have capped
your process with `set_per_process_memory_fraction` — its estimate can be
optimistic. `stream_cqsa_auto` does not care, because it recovers from the
actual failure rather than predicting it.

Measured, N=262144 on a 40 GiB A100 with the same call each time, varying only
the memory the process is allowed:

| budget | what `stream_cqsa_auto` did | result |
|---|---|---|
| full card | `itr=1` | correct |
| 1.98 GiB | escalated to `itr=2` | correct |
| 1.58 GiB | escalated to `itr=2` + host-resident inputs | correct |
| 1.19 GiB | exhausted the ladder | **clean error**, not a crash |

When a monolithic call fits, `auto` detects that and does not decompose at all
(`plan_reason: "monolithic fits ...: not decomposing is both faster and more
accurate"`) — so it costs you nothing to leave it on.

**Fixed `itr` is still supported**, and is the right choice for exactly three
things: reproducing a published measurement, pinning peak memory to a known
value, and tracing the time/memory trade curve (as the tables above do).

### Explicit control, and the backward

```python
from stream_cqsa import stream_cqsa_forward, stream_cqsa_backward

out, info = stream_cqsa_forward(q, k, v, causal=True,     # itr="auto" by default
                                stream_from_host=True,    # Q/K/V live in host memory
                                accumulate_on_gpu=False)  # fp32 accumulator on host too

dq, dk, dv = stream_cqsa_backward(q, k, v, dout,
                                  out.cpu(), info["lse"].cpu(),
                                  itr=info["itr"],        # <- feed back what auto chose
                                  causal=True, stream_from_host=True)
```

`stream_cqsa_backward` takes an **`int`**, not `"auto"` — it must use the same
depth the forward used, and `info["itr"]` is where you read it.

> The two `.cpu()` calls are currently required: `stream_from_host=True` needs
> every operand host-resident, and `out`/`lse` come back wherever the forward's
> accumulator happened to live — which varies with the depth `auto` picked. This
> is a known rough edge, not a deep one; the backward raises a message naming
> the offending tensor if you forget.

`out` comes back **fp32**, and `info` carries the plan the scheduler actually
chose (`itr`, `n_subproblems`, `n_parallel`, `stage_totals_ms`, …) — useful for
profiling.

`info["lse"]` is the **global** log-sum-exp. The backward needs it and cannot
reconstruct it: that is exactly what makes the decomposed backward exact, since
it gives `P = exp(s − lse_global) ≤ 1` for every subproblem (see
[Why it does not overflow](#why-it-does-not-overflow)).

**New to the method?** [`notebooks/reference_kernels_demo.ipynb`](notebooks/reference_kernels_demo.ipynb)
walks through the decomposition using the pure-PyTorch reference kernels in
`stream_cqsa.backends.exact` — it shows the CQS mask partitioning the pair set,
swaps inner kernels, and demonstrates the overflow that motivates the production
design. Slow by construction, but every step is inspectable.

Runnable versions: [`examples/quickstart.py`](examples/quickstart.py),
[`notebooks/tutorial_stream_cqsa.ipynb`](notebooks/tutorial_stream_cqsa.ipynb)
(usage, profiling, and simulating a smaller card with
`torch.cuda.set_per_process_memory_fraction`), and
[`notebooks/accuracy_demo.ipynb`](notebooks/accuracy_demo.ipynb).

### What the knobs mean

| argument | effect |
|---|---|
| `itr` | decomposition depth. Higher = smaller subproblems = less peak memory, more time. **Leave it at the `"auto"` default**; set an int only to pin a specific point (see above). |
| `stream_from_host` | keep Q/K/V in host memory, page each subsequence in on demand. Removes the largest O(N) *device* term. |
| `accumulate_on_gpu` | `False` moves the fp32 accumulator to the host. Slower per subproblem, but removes an O(N) device term that **no depth of `itr` can shrink**. |
| `c`, `interest_set` | CQS parameters. Defaults `c=7`, `(0,1,3)` — a λ=1 Singer difference set. |

### Why it does not overflow

The obvious way to recompose subproblems is through unnormalised numerators and
denominators, reconstructing `Den_i = exp(lse_i)` from each local kernel's
log-sum-exp. **This is algebraically correct and numerically unusable.** `exp`
overflows fp32 at `lse > 88.72` (`ln FLT_MAX`), and `lse` grows with both score
magnitude and sequence length — precisely the regime this method exists for.
Past the threshold you get `inf/inf`: silently zeros or NaNs, no exception, no
warning. In the backward the same substitution yields **silently wrong
gradients**, which is worse, because training degrades slowly and the model gets
the blame.

Stream-CQSA never forms that quantity. The forward carries `(acc, l, m)` and
merges by re-basing both sides onto `m' = max(m, m_i)`, so every exponent is
`≤ 0` and every intermediate lies in `(0, 1]`. The backward uses the *global*
log-sum-exp already produced by the forward, giving
`P = exp(s − lse_global) ≤ 1` by construction. `exp(lse)` is never computed, for
any subproblem, at any point.

The full account — including the CQS block-summary masking that gives O(1) tile
verdicts, and the register-budget bug that cost a 7× slowdown before it was
found — is in **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

---

## Reproduce the experiments


All numbers above come from the raw JSONL committed under `results/`, and the
report and figures are generated from it — they cannot drift from the data.

```bash
python benchmarks/make_report.py                  # regenerates docs/RESULTS.md
python benchmarks/make_figures.py results/paper/* --out-dir docs/figures

# re-run the sweep yourself
python benchmarks/run_paper_experiment.py --preset smoke --out-dir /tmp/smoke
python benchmarks/run_paper_experiment.py \
    --n 1048576,2097152,4194304 --dtypes fp16 --directions fwd,bwd \
    --methods sdpa,sdpa_mem,flash,cqsa_host --itr-list 1,2 \
    --out-dir /tmp/sweep
```

Full tables, provenance, and the measurement caveats that matter (allocated vs
reserved memory, warm-up contamination at small N, why stage timings over-count)
are in **[docs/RESULTS.md](docs/RESULTS.md)**.

---

## Citation


> Yiming Bian and Joshua M. Akey. *Stream-CQSA: Avoiding Out-of-Memory in
> Attention Computation via Flexible Workload Scheduling.* arXiv:2604.20819, 2026.
> <https://doi.org/10.48550/arXiv.2604.20819>

```bibtex
@article{bian2026streamcqsa,
  title   = {Stream-CQSA: Avoiding Out-of-Memory in Attention Computation
             via Flexible Workload Scheduling},
  author  = {Bian, Yiming and Akey, Joshua M.},
  journal = {arXiv preprint arXiv:2604.20819},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.20819},
  url     = {https://doi.org/10.48550/arXiv.2604.20819}
}
```

The arXiv entry is versioned and the DOI above always resolves to the latest
version. This repository tracks the revision in preparation, so some numbers
here are newer than those in v1.

---

## License


BSD 3-Clause — see [LICENSE](LICENSE).

This project vendors and derives from two BSD-3-Clause codebases, whose license
texts are reproduced in [`third_party/`](third_party/):

- **FlashAttention-2** (Copyright © 2023, Tri Dao) — `csrc/flash_attn/` is a
  derivative work; the CQS masking and block-skipping changes are ours.
- **NVIDIA CUTLASS** 4.3.4 (Copyright © 2017–2025 NVIDIA CORPORATION &
  AFFILIATES) — vendored headers under `csrc/cutlass/`.
