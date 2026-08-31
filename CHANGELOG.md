# Changelog

## 0.3.1

Measurement only. No change to any computed result.

### Fixed: the streamed forward mis-timed its device-to-host stage

`stream_cqsa_forward` recorded the `d2h` stage with a host timer and no CUDA
event, so `TraceRecorder.stage_totals_ms()` fell back to the host interval for
it. That copy is blocking, so the interval also contained the wait for the
asynchronous kernel before it, and the stage reported kernel + copy rather than
copy. Per subsequence at N=1M: 1732 ms recorded against 1566 ms of kernel, a
166 ms difference that is the actual transfer. In aggregate the host-accumulating
forward's stage totals came to 1.95x its measured wall clock, where every other
configuration came to 1.00x.

The copy is now bracketed by a CUDA event pair, as the backward always did for
all three of its device stages. Measured directly, returning the accumulator to
the host costs 23.2 s at N=16M -- 0.7% of the call -- and doubles at 2.00 per
doubling of N, which is what an O(N) transfer should do.

Only `stage_totals_ms()` output changes. Wall clock, peak memory and every
numerical result are unaffected: re-measuring the configuration on the same
hardware reproduced wall clock to 0.9% and kernel time to 0.1%.

## 0.3.0

The planner now accounts for where tensors actually live, which changes the
depth it selects at the top of the range. Everything else is additive.


### Fixed: the depth planner counted host-resident terms as device memory

`estimate_peak_bytes` computed its O(N) floor as `(3*itemsize + 4 + 4) * N*H*D`
-- Q/K/V, the fp32 accumulator and the fp32 output -- regardless of where those
tensors actually live. But `stream_from_host` puts Q/K/V on the host and
`accumulate_on_gpu=False` puts the accumulator there, which is precisely the
configuration a caller asking for them has requested.

The failure is not conservatism. An estimator that overstates a floor no depth
can satisfy does not pick a safer depth, it concludes that nothing fits and falls
back to the deepest one it has. At N=16M the planner reported an O(N*H*D) floor
of 112 GiB against a 59 GiB budget and selected itr=3. The configuration it was
estimating then ran in 32.0 GiB, and at itr=2 completes the forward in 3896 s
against 6228 s -- strictly faster at an identical peak. The fallback was a
dominated choice rather than a trade.

`estimate_peak_bytes` and `plan_decomposition` now take `stream_from_host` and
`accumulate_on_gpu` and drop the terms that are not device-resident, and the
forward passes its own configuration through. At the 16M budget the planner
returns itr=2 where it previously returned itr=3. Below 16M nothing changes: a
monolithic call fits, so the planner returns itr=0 by a path that never consulted
the floor.

### Fixed: a fixed depth could silently become a deeper one

`stream_cqsa_forward` and `stream_cqsa_backward` take `allow_escalation`, and the
experiment runner sets it False whenever `itr` is not `"auto"`. A column labelled
itr=1 has to report what itr=1 costs, including when itr=1 does not fit; with
recovery enabled a failing configuration is refined and the rescue is recorded
under the depth that failed. A 16M forward reported success at 76.0 GiB this way,
having escalated seven of its subproblems.

### Added: the backward reports its own escalations

`stream_cqsa_backward` takes an optional `bwd_info` dict and records
`itr_requested`, `itr_reached`, `depth_escalations` and `oom_retries` from both
its recovery paths. It previously returned only gradients, so a row's counters
came from the forward and a backward that deepened itself was indistinguishable
from one that did not.

### Added: the backward can place its accumulators

`stream_cqsa_backward` takes `accumulate_on_gpu`, the analogue of the forward's.
dQ, dK and dV are three fp32 [B,N,H,D] buffers -- 12 bytes per element, measured
at 12.01 against a predicted 12.00 -- and where they live decides whether 8M runs
at all. Previously the placement was implied by `stream_from_host` with no way to
choose, so the acc=GPU and acc=CPU backward columns were the same measurement.

## 0.2.0

### Autograd

`stream_cqsa_attn` and `StreamCQSAAttention` make the native path an autograd
operator, so `.backward()` works and Stream-CQSA drops into a model like any
other attention call. The explicit `stream_cqsa_forward` / `stream_cqsa_backward`
pair is unchanged and remains the way to pin every scheduling choice.

Both APIs live in the same package and call the same kernels: `native_autograd`
is a thin layer over `stable_stream`, not a second implementation. See
[Autograd and `.backward()`](README.md#autograd-and-backward) for what the
convenience costs, measured.

Operands must be fp16 or bf16. Passing fp32 now raises a message saying so,
rather than failing inside the kernel.

### Depth escalation

A subproblem that does not fit is now decomposed a further level and its children
run in its place -- the recovery the method describes, in the forward and in both
backward paths. Previously the forward could only lower its stream count, and the
device backward had no out-of-memory handling at all.

Escalation is bounded by structure rather than by a constant: `c**itr <= N` is the
deepest decomposition a sequence admits. `attention_oom_safe(max_itr=...)` now
defaults to that bound instead of 4.

Refining one task at a time, rather than the whole problem, is what produces the
hybrid schedules of mixed depth that a changing memory budget calls for.

### Fixed: a race on any call issued with GPU work in flight

The scheduler creates CUDA side streams per call and never ordered them after the
caller's stream. A new stream carries no such dependency, so a call made while the
caller still had work in flight ran alongside it. This corrupted entire gradient
tensors -- 8 of 8 trials at N=32768 returned NaN, and 1 of 8 at N=8192.

It stayed hidden because any synchronising read between the caller's work and the
call orders the two by accident and the corruption disappears, which is the shape
every existing test happened to have. `.backward()` made it easy to hit, since a
backward naturally follows the rest of a model.

Fixed by making the worker streams, and the merge stream, wait on the caller's
stream. The exit side was already covered by the synchronize after the scheduling
loop. Affects the forward and the streamed backward at every sequence length.

### Also

* The output of `stream_cqsa_attn` follows its inputs' device. The forward returns
  its result wherever the accumulator lived, which varies with the schedule, and
  letting a scheduling decision pick the output's device made `grad_output` arrive
  somewhere the caller had not put it.
* `notebooks/tutorial_stream_cqsa_v2.ipynb`, covering the above with outputs saved.

## 0.1.0

First release.
