# Implementation methodology of the native Stream-CQSA kernel

*Drafted for insertion as a paper section. Every number below is measured on a
single NVIDIA A100, causal attention, `B=1, H=8, D=64`, `c=7`,
`I=(0,1,3)`; provenance for each is given inline so claims can be re-checked.*

Stream-CQSA decomposes one attention call into many smaller ones and recomposes
their local statistics. That is a statement about arithmetic. Turning it into a
kernel that is *worth using* required a separate set of decisions, and this
section records them. They fall into three groups: keeping the recomposition
numerically representable (§1), making the CQS mask cheap enough that
decomposition is not swamped by masking overhead (§2), and keeping the host side
off the critical path (§3). §4 lists what was measured and rejected, and §5 the
measurement practices without which several of these numbers would have been
wrong.

The single most important lesson is §2.2, and it is not about attention at all:
**in a register-saturated kernel, state that is merely *held* is as expensive as
state that is computed.** It cost a 7x slowdown before it was found.

---

## 1. Numerical stability: never form `exp(lse)`

### 1.1 The overflow that motivates the design

The natural way to write the recomposition is in terms of an unnormalised
numerator and denominator. For a subproblem `i` with local scores `R_i`,

```
Num_i = exp(R_i) V_i ,    Den_i = rowsum(exp(R_i)) ,    O = (sum_i Num_i) / (sum_i Den_i)
```

and since a local kernel typically returns the *normalised* output `out_i` and a
log-sum-exp `lse_i`, one reconstructs `Den_i = exp(lse_i)` and
`Num_i = out_i * Den_i`.

This is algebraically correct and numerically unusable. `exp` overflows fp32 at
`lse > 88.72` (`ln FLT_MAX`), and `lse` is a *log-sum* over the retained keys:
it grows with both the score magnitude and the sequence length, which is
precisely the regime this method exists for. Past the threshold `Den_i` becomes
`inf`, `Num_i/Den_i` becomes `inf/inf`, and the result is silently zeros or
NaNs — not an exception, not a warning. In the backward the same substitution
appears as `dNum = dO/Den` and `dDen = -rowsum(dO . Num)/Den^2`, so the failure
mode there is **silently wrong gradients**, which is worse than wrong
activations because training degrades slowly and the model gets the blame.

Every decision below follows from refusing to materialise that quantity.

### 1.2 Forward: a max-shifted merge over `(acc, l, m)`

Each subproblem returns three statistics rather than a normalised output:

| symbol | shape | meaning |
|---|---|---|
| `m` | `[B,H,N]` | running row maximum of the scores seen so far |
| `l` | `[B,H,N]` | running sum of `exp(score - m)` |
| `acc` | `[B,H,N,D]` | running sum of `exp(score - m) * V` |

Merging a new contribution `(acc_i, l_i, m_i)` into the accumulator
`(acc, l, m)` re-bases both sides onto `m' = max(m, m_i)`:

```
acc <- acc * exp(m - m') + acc_i * exp(m_i - m')
l   <- l   * exp(m - m') + l_i   * exp(m_i - m')
m   <- m'
```

Both exponents are `<= 0` by construction, so every intermediate lies in
`(0, 1]`. The output `O = acc / l` and the global log-sum-exp
`lse = m + log(l)` are formed once, at the end. **`exp(lse)` is never
computed**, at any point, for any subproblem. This is the standard online-softmax
recurrence; the contribution here is recognising that the *decomposition
framework*, not just the inner kernel, has to be written in these terms — a
framework that consumes `out_i` and `lse_i` has already lost the property.

The merge is order-independent (addition commutes and the re-basing is exact),
which is what allows subproblems to complete in any order.

### 1.3 Backward: the global-lse reformulation

The backward needs the attention weights `P`, and the `Num`/`Den` form
reconstructs them as `exp(R_i)/Den_i` — overflowing numerator over overflowing
denominator. Substituting `Num = Den . O` and folding `Den` into `P` gives

```
P[q,k] = exp(s[q,k] - lse_global[q])  <=  1        for all q, k
```

where `lse_global` is the *global* log-sum-exp already produced by the forward at
no extra cost. Every subproblem receives it, so the standard softmax backward
applies unchanged on the retained pair set:

```
D[q]     = sum_d dO[q,d] O[q,d]
dV[k]   += sum_q P[q,k] dO[q]
dP[q,k]  = dO[q] . V[k]
dS[q,k]  = P[q,k] (dP[q,k] - D[q])
dQ[q]   += scale * sum_k dS[q,k] K[k]
dK[k]   += scale * sum_q dS[q,k] Q[q]
```

Each term is a sum over retained pairs, and CQS partitions the pairs exactly
once, so per-subproblem contributions simply add. Nothing exceeds 1 relative to
the softmax: the formulation is overflow-free **by construction** rather than by
range assumption.

> **Note for the algorithm block.** The paper's `algo:bwd` is written in the
> `Num`/`Den` form because that is the clearest way to *show* additivity. The
> two are the same gradient — the substitution is an identity — but a reader who
> transcribes the algorithm literally will get the silent failure of §1.1. The
> implemented form is the one above, and the difference is worth stating
> explicitly rather than leaving as an exercise.

### 1.4 fp32 statistics: why decomposition costs no accuracy

`(acc, l, m)` and the merge are computed in fp32 regardless of the input dtype,
and the kernel writes its accumulator out in fp32 rather than rounding to the
input type. The consequence is the central accuracy result: **error does not
compound across merged subproblems**, so a decomposed run is as accurate as a
monolithic one. Measured against a float64 reference (forward, sampled rows):

| N | SDPA | Stream-CQSA | ratio |
|---|--:|--:|--:|
| 16384 | 2.77e-04 | 1.81e-04 | 0.65 |
| 65536 | 2.87e-04 | 1.95e-04 | 0.68 |
| 262144 | 3.20e-04 | 2.11e-04 | 0.66 |

Stream-CQSA is consistently **~1.5x more accurate than the unwrapped baseline**,
not merely equal: decomposing buys accuracy here, because the statistics and
output path are fp32 while the baseline's output is fp16.

Across precisions the error is set by the input dtype alone. bf16 is 8.0x fp16
in both directions — exactly the mantissa ratio (10 vs 7 explicit bits,
`2^3 = 8`) — confirming the floor is dtype rounding and not anything the
decomposition adds.

### 1.5 A conditioning caveat: near-uniform attention

One regime deserves explicit mention because it is easy to mistake for a
decomposition defect, and because the project's own input convention sits in it.

The backward forms `dS = P (dP - D)` with `D[q] = sum_d dO[q,d] O[q,d]`. When
attention is close to uniform, `dP` and `D` are close and the subtraction is a
**catastrophic cancellation**: the leading digits agree and cancel, so the result
is dominated by rounding. Decomposition amplifies it, because each subproblem's
`dP` covers only its retained keys and is therefore further from the global `D`
it is differenced against.

Measured (`dQ` relative error against float64, N=4096, varying the input scale
that controls how peaked the softmax is):

| input scale | dtype | itr=0 | itr=1 | itr=2 |
|---|---|--:|--:|--:|
| 0.05 | fp16 | 3.46e-03 | 5.42e-03 | **8.51e-03** |
| 0.05 | bf16 | 2.37e-03 | 2.37e-03 | 2.37e-03 |
| 0.5 | fp16 | 3.04e-04 | 3.05e-04 | 3.05e-04 |
| 1.0 | fp16 | 3.07e-04 | 3.07e-04 | 3.08e-04 |

Three observations, in order of importance:

1. **The undecomposed baseline is already degraded at scale 0.05** — 3.46e-03
   against 3.04e-04 at scale 0.5, 11x worse. The cancellation hurts every
   implementation; it is a property of the regime, not of the method.
2. **At realistic score magnitudes the drift disappears entirely.** At scale 0.5
   and 1.0, `itr` has no measurable effect.
3. bf16 never drifts, because its own rounding floor (~2.4e-03) already sits
   above the cancellation noise and masks it.

**Consequence for experiment design.** Q/K/V scaled by 0.05 — the convention used
in this project's earlier experiments and stated in the paper's setup — puts the
softmax in the near-uniform regime. Forward numbers are unaffected, but *backward*
numbers measured there are reporting the conditioning of `dP - D` rather than the
accuracy of the implementation, and they will make any decomposed method look
worse than it is at realistic magnitudes. Gradient comparisons should use
unit-variance activations. This is also the most likely explanation for the
outsized bf16 `dV` errors previously reported in the appendix.

### 1.5 The empty-row sentinel

A subsequence can legitimately retain no keys for some query row. The natural
sentinel is `lse = -inf` (or `m = -inf`), but a merge that re-bases onto
`m' = max(m, m_i)` then evaluates `exp(-inf - (-inf))`, which is NaN, and one
NaN destroys the row. Contributions whose statistics are non-finite are
therefore normalised to "no contribution" before merging. This was not
hypothetical: an `+inf` sentinel silently erased 586 of 2048 token rows before it
was caught.

---

## 2. Making the CQS mask cheap

Adding a mask to a FlashAttention-style kernel is not free, and the naive cost is
severe: the generic per-element masking path reads the per-token group-bit vector
from global memory *twice per element* in the kernel's innermost loop. Merely
enabling CQS with an all-zero mask — a mask that excludes nothing — measured
**4.3x slower than the unmodified kernel** in the forward and **9.1x** in the
backward. Three techniques bring it back.

### 2.1 Block summaries and an O(1) tile verdict

Group bits are summarised per aligned block of tokens: `blk_or[b]` is the OR of
the bits in block `b`, `blk_and[b]` the AND. From these, two range queries decide
a whole tile in constant time, before any per-element work:

- **clear** — `OR(rows) & OR(cols) == 0`: no pair in the tile is masked, so the
  masking loop is skipped entirely and the tile takes the unmodified
  FlashAttention path.
- **fully masked** — `AND(rows) & AND(cols) != 0`: every pair shares a group bit,
  so the tile contributes exactly zero and its GEMMs can be skipped (§2.3).

The AND test requires block-aligned ranges — a partial edge block's AND describes
tokens outside the tile and could claim a mask that does not hold — while the OR
test is safe on any range because `blk_range_or` rounds outward and can only fail
to skip, never skip wrongly. Both are conservative in the safe direction.

This works because **the CQS mask is block-structured**: a subsequence is the
union of whole chunks, and chunks are contiguous token ranges. Measured census of
the tiles that causal masking already keeps (128x128 tiles):

| N | itr | fully masked | fully clear | partial |
|---|--:|--:|--:|--:|
| 65536 | 1 | 22.0% | 77.2% | 0.8% |
| 65536 | 2 | 38.8% | 58.9% | 2.3% |
| 262144 | 2 | 39.4% | 60.1% | 0.6% |

Over 97% of tiles are resolved in O(1). Better, the *partial* fraction — the only
tiles that pay per-element masking — shrinks as N grows, because a tile is
partial only where it straddles a chunk boundary: partial tiles are `O(nb)` of
`O(nb^2)`, i.e. `~l/nb`. Measured 2.08% / 0.53% / 0.13% at `nb` = 95 / 377 /
1505, matching the prediction. **At N=1M only 0.13% of tiles run the general
path**, which is why no further masking optimisation was pursued.

### 2.2 Register budget: the dominant effect

This was the largest single performance factor and the least expected.

The mask needs nine values (enable flag, the group-bit pointer, the two block
summaries, their block size and count, and three chunk-arithmetic fields).
Holding them as members of the `Mask` object is the obvious design, and it costs
nothing in an ordinary kernel. FlashAttention kernels are not ordinary: they run
at the register ceiling. Nine extra live values — several of them pointers at two
registers each — pushed them over it, and the additions went straight into
local-memory spill inside the innermost loop.

Register usage, ours versus the upstream `flash-attn` binary (obtained with
`cuobjdump -res-usage`; upstream compiles many more variants, hence the larger
`n`):

| build | REG | spilling | mean spill |
|---|---|--:|--:|
| ours, forward | 220–255 | 206/230 (90%) | 229 B |
| upstream, forward | 129–255 | 1816/3150 (58%) | 107 B |
| ours, backward *(before fix)* | 254–255 | 242/264 | 274 B |
| ours, backward *(after fix)* | 241–255 | 154/264 (58%) | 123 B |
| upstream, backward | 231–255 | 1980/3680 (54%) | 56 B |

The fix is to stop *holding* the state: the block summaries and their geometry
are read once per tile, and the chunk-arithmetic fields are unused in the hot
path, so they were moved out of the object and read from the kernel's `params`
struct — which already lives in constant memory, making a use-site read a cheap
broadcast instead of a register held for the kernel's lifetime. The mask logic
itself was not changed.

Effect on the backward kernel, same shape, CQS mask disabled so the comparison
isolates the cost of *carrying* the feature:

| L | before | after | vs SDPA |
|---|--:|--:|--:|
| 8192 | 20.3 ms | **4.8 ms** | 10.59x -> **1.42x** |
| 12039 | 25.9 ms | **9.0 ms** | 7.22x -> **1.37x** |
| 28088 | 123.8 ms | **24.5 ms** | 6.27x -> **1.38x** |

The generalisable statement: **when a kernel is at the register ceiling, the cost
of a feature is dominated by its live state, not by its arithmetic.** The same
mask code costs the forward only 1.4–2.0x — because the forward had slack — and
cost the backward 7x, because it had none. Diagnosing this required a baseline
build with the feature compiled out; without one, the natural conclusion is
"the masking logic is slow," which is false and would have led to optimising the
wrong thing.

### 2.3 Skipping fully-masked tiles without breaking the pipeline

A tile the AND test proves fully masked contributes exactly zero, so its five
GEMMs are wasted. It cannot simply be `continue`d: the backward's `m_block` loop
carries state across iterations — the `dq_accum` load/modify/store round trip,
the sQ/sdO double-buffer swaps, the `cp_async` prefetches for the next
iteration, the `gLSE`/`gdPsum` pointer advances, and several `__syncthreads()`.
Skipping the iteration corrupts all of it.

Instead the GEMMs, the mask application and the `exp2` are individually guarded
while every stateful step continues to run. The verdict depends only on
`(m_block, n_block)` and the block summaries — never on `tidx` — so it is
CTA-uniform and the barriers remain collectively executed.

The design property that makes this safe: `acc_s` and `acc_dp` are both cleared
at the top of the loop, so skipping the first GEMM and the `exp2` leaves
`scores == 0`, hence `dS == 0`, hence a zero contribution. **The tile is
algebraically inert whether or not the later guards fire**, so correctness does
not depend on getting each guard exactly right — skipping the remaining GEMMs is
purely an optimisation. (`apply_mask` must be guarded for the same reason: it
writes `-INFINITY` into `acc_s`, which would take `scores` off zero.)

Measured in isolation — same kernel, same shape, only the mask differing:

| N | itr | L | zero mask | real mask | gain | skippable |
|---|--:|--:|--:|--:|--:|--:|
| 65536 | 1 | 28086 | 28.1 ms | 23.4 ms | 1.20x | 21.9% |
| 262144 | 2 | 48148 | 72.2 ms | 51.0 ms | **1.42x** | 55.3% |

A real CQS mask is now *faster* than an all-zero one despite doing strictly more
masking work, which is the cleanest available evidence that the skip pays. The
gain trails the naive `1/(1-skippable)` bound because the shared-memory writes,
the elementwise `dS` loop and the per-iteration bookkeeping still run, and
diagonal tiles are only partially masked.

### 2.4 Sorted gather makes the stock causal mask exact

Subsequences are gathered in ascending chunk order. Consequently local token
order equals global token order, and a query at local position `i` precedes a key
at local position `j` globally exactly when `i >= j` locally. The unmodified
FlashAttention causal mask is therefore exactly correct on the gathered
subsequence, with no index translation and no custom causal logic. This is a
correctness simplification rather than a speed one, but it removes a whole class
of off-by-one bug and lets the CQS mask compose with causality by a plain AND.

---

## 3. Keeping the host off the critical path

In the streamed configuration the host performs a gather per subproblem and a
scatter of the gradients. Both are large memory operations and both were
bottlenecks at some point.

**Token-major layout, once.** Gathering from `[B,H,N,D]` requires a
`transpose().contiguous()` per subproblem, which is cache-hostile: 166.6 ms
versus 3.0 ms at N=32768, a **56x** difference. Transposing once up front makes
every subsequent gather a contiguous `index_select` on the token dimension.

**Contiguous runs instead of index vectors.** With sorted gather, a subsequence
is the ascending union of a few whole chunks, so its index vector is only 2–3
contiguous runs — never L scattered positions. Gathering and scattering by slice
is the same bytes with none of the indirection: **22x** faster on the host at
L=12039 (5.5 ms -> 0.2 ms). The code detects the run structure and falls back to
`index_select`/`index_add_` when the ids are not ascending, so the optimisation
cannot silently produce a wrong answer on an unsorted path.

**Mixed-dtype accumulation.** `fp32_acc.add_(fp16_src)` promotes to the
accumulator dtype and is **bit-identical** to `add_(src.float())`, but avoids
materialising a full-size fp32 temporary: 1.86x on the add at L=12036.

---

## 4. Measured and rejected

Recording these matters as much as the accepted techniques, because each is an
optimisation that *sounds* obviously correct.

| technique | expectation | measured outcome |
|---|---|---|
| Async host scatter on a worker thread | hides scatter behind GPU work | **16.2% slower.** The host is bandwidth-bound; overlapping scatter with gather makes two bandwidth-bound streams contend. |
| Pinned inputs + direct run-wise DMA | removes the staging copy entirely | **2x device workspace, no time win.** Needs per-slot device landing buffers — the wrong trade for a path whose purpose is minimal device memory. |
| Shared-memory column bits for partial tiles | speeds the general masking path | **Not built.** Partial tiles are 0.13–2.3% of tiles and shrink as `~l/nb`; ceiling ~0.1% at N=1M, against added state in a register-critical kernel. |
| Drop `O` from transfers via precomputed `dP_sum` | ~4.6% (O is 1/5 of gather and H2D) | **1.8–2.4x slower.** See §5.3. |
| Chunk-residency dedup across concurrent subproblems | lower peak memory | Peak was flat in `n_par` on the device path; shipped as an opt-in flag for the streamed path only. |
| Raising scheduler parallelism `n_par` | more overlap | Auto policy already optimal; workspace flat in `n_par`. |

---

## 5. Measurement practices

Several results above would have been wrong under naive measurement.

### 5.1 Peak GPU memory is not "the memory attention used"

`max_memory_allocated()` includes the caller's Q/K/V/O and allocator cache, and
across residency models it is actively misleading: a host-streaming method keeps
Q/K/V off the device, so its peak falls for a reason unrelated to how much
scratch attention needs. Reporting

```
dev_total = dev_inputs + dev_workspace
```

separately corrected a claim in our own earlier draft. Host streaming had been
reported as a "5.4x reduction". Split, at N=262144: the **backward's** reduction
is real workspace (3696 -> 626 MiB, because the fp32 `dQ/dK/dV` accumulators
genuinely relocate), while the **forward's** workspace is *unchanged* at 2173 MiB
— streaming removes exactly the input residency, a 26% saving, not 5.4x. Raw
peak conflated the two cases.

### 5.2 Accuracy has to be measurable at the N that matters

A dense float64 reference is O(N^2) and cannot be formed past ~16k — below every
sequence length this method exists for. The reference instead samples R query
rows and computes exact float64 attention for those rows against all N keys,
streaming keys in tiles: O(R.N) memory, usable at any N, and itself written with
a running-max softmax so the reference is not an overflow risk at long context.

### 5.3 A stage total is a ceiling, not an estimate

Per-stage timings attribute *overlapped* time. Predicting a ~4.6% gain from
removing `O` from the transfers (it is 1/5 of the gather and 1/5 of the H2D) was
wrong by more than an order of magnitude: `O` rides the same pinned staging
pipeline as Q/K/V/dO and hides behind compute, so its marginal wall cost is far
below its stage total, while computing the global `dP_sum` must read `dO` and `O`
once from pageable host memory as serial, unoverlapped work. The implementation
is exact and complete, and it is **1.8–2.4x slower**; it ships behind a
default-off flag.

**A stage total bounds what removing that stage can save; it does not estimate
it.** Confirm with an end-to-end A/B before implementing, not after.

### 5.4 Miscellaneous traps encountered

- **Under-warming.** One warmup did not cover first-call setup on the device
  path, inflating a measurement 10x (2725 ms against a true, rock-steady 255 ms).
  Caught by internal consistency: N=65536 and N=262144 reported nearly equal
  times despite 16x the work.
- **Profiler ordering.** Whichever case is profiled first pays warmup; a kernel
  read 91 ms vs 72 ms purely from ordering. Re-measured head to head, identical.
- **A reference that OOMs is not a method that OOMs.** The float64 backward
  reference is far more memory-hungry than any method under test; letting its
  OOM propagate marked the *method* as OOM and retired it from larger N —
  manufacturing exactly the boundary the experiment was meant to measure.

---

## 6. Summary

| technique | where | measured effect |
|---|---|---|
| Max-shifted `(acc,l,m)` merge; never form `exp(lse)` | framework | removes silent zeros/NaNs above `lse > 88.7` |
| Global-lse backward reformulation | framework | overflow-free gradients by construction |
| fp32 statistics and output | framework | ~1.5x **better** accuracy than the baseline; error does not compound |
| Block-summary O(1) tile verdicts | kernel | resolves >97% of tiles without per-element work |
| CQS state read from constant memory, not held in registers | kernel | **backward 10.6x -> 1.4x** of SDPA |
| Fully-masked tile skipping (guarded, not `continue`) | kernel | further **1.20–1.42x** |
| Sorted gather | framework | stock causal mask exact; no index translation |
| Token-major once; contiguous-run slicing | host | 56x; 22x |
| Mixed-dtype accumulate | host | 1.86x, bit-identical |

Net effect against the unwrapped baselines at N=262144: forward **1.57x** SDPA,
backward **1.8x** — against a decomposition cost of `(l^2/c)^itr = 1.29x` at
`itr=1`. The wrapper itself accounts for 1.01x, so the method runs at
essentially its algorithmic cost and the residual is our kernel build rather than
the decomposition.
