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

---

## The result in one table

Causal attention, `B=1 H=8 D=64`, fp16, **backward pass**, NVIDIA A100 80GB.
Peak is `torch.cuda.max_memory_allocated()`.

| N | SDPA | SDPA (mem-eff.) | FlashAttention-2 | Stream-CQSA `itr=1` | Stream-CQSA `itr=2` |
|---|---|---|---|---|---|
| **1M**  | 26 s / 12.1 GiB | 78 s / 11.1 GiB | 25 s / 10.1 GiB | 57 s / 7.6 GiB | 68 s / **4.4 GiB** |
| **2M**  | 106 s / 24.1 GiB | 3 860 s / 22.1 GiB | 101 s / 20.1 GiB | 218 s / 15.3 GiB | 262 s / **8.9 GiB** |
| **4M**  | 429 s / 48.3 GiB | 15 421 s / 44.3 GiB | 407 s / 40.3 GiB | 854 s / 30.5 GiB | 998 s / **17.7 GiB** |
| **8M**  | **OOM** | **OOM** | **OOM** | 3 372 s / 61.1 GiB | 3 895 s / **35.5 GiB** |

Two things to read off it:

1. **At 8M every baseline is out of memory and Stream-CQSA finishes** — on the
   same card, with the same numerics.
2. **Even below the wall it is not just a fallback.** At 4M it does the
   backward in 17.7 GiB against FlashAttention-2's 40.3 GiB — **2.3× less peak
   memory** — because raising `itr` shrinks the in-flight working set while the
   inputs stream from host memory.

The price is time: **2.1–2.7× FlashAttention-2** in the backward and
**1.5–2.0×** in the forward, depending on depth. This is a deliberate trade,
and it is the honest summary of the method — see
[What it costs](#what-it-costs).

![memory and wall-clock across N](docs/figures/fig_mem_time_fp16.jpg)

---

## Install

Requires a CUDA GPU of compute capability ≥ 8.0 (Ampere or newer), PyTorch with
CUDA, and `ninja`. The CUDA extension is **compiled for your own GPU** — there
are no prebuilt wheels, because the kernel set is large and building only your
architecture keeps compile time and binary size sane.

```bash
git clone https://github.com/yiming-b/Stream-CQSA.git
cd Stream-CQSA
pip install -e .
```

`setup.py` detects your architecture via `torch.cuda.get_device_capability()`
and builds the fp16 + bf16 × head-dim 64/128 × causal/non-causal kernel set.
Expect **20–60 minutes** on a multi-core node; `MAX_JOBS=8 pip install -e .`
controls parallelism if you are memory-limited while compiling.

```bash
pytest tests/ -q          # verify the build
```

---

## Use it

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

| rung | `itr` | inputs | accumulator |
|---|---|---|---|
| 1–2 | 1, 2 | device | device |
| 3–4 | 1, 2 | **host** | device |
| 5–7 | 2, 3, 4 | **host** | **host** |

Each rung relaxes exactly one constraint, so you pay only for the headroom you
actually need. When a monolithic call fits, **call the monolithic kernel** —
this entry point is deliberately not the fastest path.

For explicit control, and to run the backward:

```python
from stream_cqsa import stream_cqsa_forward, stream_cqsa_backward

out, info = stream_cqsa_forward(q, k, v, causal=True, itr=2,
                                stream_from_host=True,    # Q/K/V live in host memory
                                accumulate_on_gpu=False)  # fp32 accumulator on host too

dq, dk, dv = stream_cqsa_backward(q, k, v, dout, out.cpu(), info["lse"],
                                  itr=2, causal=True, stream_from_host=True)
```

> The `.cpu()` is currently required: `stream_from_host=True` needs every
> operand host-resident, and the forward still returns `out` on the device even
> when the accumulator is not. This is a known rough edge, not a deep one — the
> backward raises a message telling you exactly this if you forget.

`out` comes back **fp32**, and `info` carries the plan the scheduler actually
chose (`itr`, `n_subproblems`, `n_parallel`, `stage_totals_ms`, …) — useful for
profiling.

`info["lse"]` is the **global** log-sum-exp. The backward needs it and cannot
reconstruct it: that is exactly what makes the decomposed backward exact, since
it gives `P = exp(s − lse_global) ≤ 1` for every subproblem (see
[Why it does not overflow](#why-it-does-not-overflow)).

Runnable versions: [`examples/quickstart.py`](examples/quickstart.py),
[`notebooks/tutorial_stream_cqsa.ipynb`](notebooks/tutorial_stream_cqsa.ipynb)
(usage, profiling, and simulating a smaller card with
`torch.cuda.set_per_process_memory_fraction`), and
[`notebooks/accuracy_demo.ipynb`](notebooks/accuracy_demo.ipynb).

### What the knobs mean

| argument | effect |
|---|---|
| `itr` | decomposition depth. Higher = smaller subproblems = less peak memory, more time. `"auto"` (default) picks from free memory. |
| `stream_from_host` | keep Q/K/V in host memory, page each subsequence in on demand. Removes the largest O(N) *device* term. |
| `accumulate_on_gpu` | `False` moves the fp32 accumulator to the host. Slower per subproblem, but removes an O(N) device term that **no depth of `itr` can shrink**. |
| `c`, `interest_set` | CQS parameters. Defaults `c=7`, `(0,1,3)` — a λ=1 Singer difference set. |

---

## Accuracy

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

## What it costs

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

---

## Why it does not overflow

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

## Reproduce

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

## Layout

```
stream_cqsa/          Python package
  stable_stream.py      main entry points: forward, backward, scheduler, chunk pool
  oom_fallback.py       stream_cqsa_auto and the escalation ladder
  reference.py          readable ground-truth implementation, used by the tests
  cqs_mask.py           CQS mask construction
csrc/                 CUDA extension (FlashAttention-2 derived, + CUTLASS headers)
  flash_attn/src/       11 of these files carry the CQS modifications
tests/                correctness suite
benchmarks/           the experiment harness, report and figure generators
notebooks/            usage tutorial and accuracy demo
results/paper/        raw JSONL backing every published number
docs/                 METHODOLOGY.md (implementation), RESULTS.md (generated)
```

---

## Citation

Paper in preparation. Please open an issue if you would like to cite this before
it appears.

## License

BSD 3-Clause — see [LICENSE](LICENSE).

This project vendors and derives from two BSD-3-Clause codebases, whose license
texts are reproduced in [`third_party/`](third_party/):

- **FlashAttention-2** (Copyright © 2023, Tri Dao) — `csrc/flash_attn/` is a
  derivative work; the CQS masking and block-skipping changes are ours.
- **NVIDIA CUTLASS** 4.3.4 (Copyright © 2017–2025 NVIDIA CORPORATION &
  AFFILIATES) — vendored headers under `csrc/cutlass/`.
