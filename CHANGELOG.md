# Changelog

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
