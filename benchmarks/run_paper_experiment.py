#!/usr/bin/env python
"""
Publication experiment: Stream-CQSA native kernel versus strong exact-attention
baselines, across sequence length, precision, and direction.

One sweep produces every number the paper needs:

  axes      N x dtype{fp16,bf16} x direction{fwd,bwd} x method
  records   output accuracy (vs float64), peak device memory, wall-clock time,
            and for Stream-CQSA a per-stage breakdown

Three claims it is built to support or refute:

  C1  accuracy    Stream-CQSA reproduces the attention output to within the
                  input dtype's own rounding, in both fp16 and bf16, forward
                  and backward.
  C2  overhead    Below the OOM boundary it costs more time and more device
                  memory than the baselines. It is not a replacement.
  C3  recovery    Past the point where every baseline OOMs, it still returns a
                  result, with the decomposition depth `itr` rising as needed.

Design notes that matter for the numbers being trustworthy
----------------------------------------------------------

*Memory.* `torch.cuda.max_memory_allocated()` counts live tensors, which is
reproducible but excludes allocator cache and CUDA context; `nvidia-smi` counts
everything but is noisy and includes the context. Both are recorded, plus the
baseline-subtracted `workspace` (peak during the call minus allocated
immediately before it) and the caller-side `inputs` resident on the device. The
split matters when comparing residency models: a host-streaming method keeps
Q/K/V off the device, so its raw peak falls for a reason unrelated to how much
scratch attention needs. Between every measurement the allocator is drained
(`gc.collect()` + `empty_cache()`) so a previous method cannot inflate or mask
the next one.

*OOM.* A method that OOMs at some N is retired from all larger N (it cannot
recover) but the sweep continues, which is the entire point of C3. OOM is
caught, the allocator drained, and the failure recorded as a result rather than
crashing the run. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so a
fragmented heap does not produce a spurious OOM that would flatter the claim.

*Accuracy at large N.* A dense float64 reference is O(N^2) and cannot be formed
past ~16k -- below every N of interest. Forward accuracy therefore samples R
query rows and computes exact float64 attention for those rows against all N
keys, streaming keys in tiles: O(R*N) memory, usable at any N. Backward uses the
same trick: for sampled rows, dQ needs all k, and dK/dV need all q, each
computable in tiles. The reference uses a running-max softmax, so it is not
itself an overflow risk at long context.

*Tensor generation.* Random tensors are generated once per (N, dtype) and reused
by every method and repeat at that point -- at N=4M a Q/K/V set is 12 GB, and
regenerating per method would dominate the run. Generation is chunked on the GPU
(fast, and bounded device residency) into the destination, so a host-resident
method never needs the full set on the device even transiently.

*Resumability.* Each result is appended to the JSONL output as it completes, so
a wall-clock timeout loses only the config in flight.

Usage
-----
    # smoke (small, fast)
    python run_paper_experiment.py --preset smoke --out-dir outputs/paper/smoke

    # full sweep
    python run_paper_experiment.py --preset full --out-dir outputs/paper/run1

    # explicit
    python run_paper_experiment.py --n 65536,262144 --dtypes fp16,bf16 \
        --directions fwd,bwd --methods sdpa,sdpa_mem,flash,cqsa_auto,cqsa_host
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

MIB = 1024.0 ** 2
GIB = 1024.0 ** 3


# ---------------------------------------------------------------------------
# Inputs: generated once per (N, dtype), reused by every method
# ---------------------------------------------------------------------------

class InputCache:
    """
    Q/K/V (+dO for backward) for one (N, dtype), generated once.

    Generation is chunked on the GPU: the RNG is orders of magnitude faster
    there than on the host, but a full set at large N will not fit alongside the
    method under test, so chunks are produced on the device and copied straight
    into the destination. `device_hint="cpu"` therefore never places the whole
    set on the GPU even transiently, which is what lets the host-streaming
    configurations be measured at N where the baselines cannot run at all.

    Scale 0.05 matches the paper's existing convention (a zero-mean Gaussian
    scaled down), which keeps softmax logits in a range where fp16 does not
    saturate and the comparison is about the method, not about overflow.
    """

    def __init__(self, B, H, D, scale=1.0, chunk=1 << 19):
        self.B, self.H, self.D, self.scale, self.chunk = B, H, D, scale, chunk
        self._key = None
        self._val = None

    def _gen(self, N, dtype, seed, dest_device):
        out = torch.empty((self.B, self.H, N, self.D), dtype=dtype, device=dest_device)
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        for s in range(0, N, self.chunk):
            e = min(s + self.chunk, N)
            blk = torch.randn((self.B, self.H, e - s, self.D), generator=g,
                              device="cuda", dtype=torch.float32) * self.scale
            out[:, :, s:e].copy_(blk.to(dtype))
            del blk
        torch.cuda.empty_cache()
        return out

    def get(self, N, dtype, seed, dest_device):
        key = (N, str(dtype), seed, str(dest_device))
        if self._key == key:
            return self._val
        self.drop()
        vals = tuple(self._gen(N, dtype, seed + i, dest_device) for i in range(4))
        self._key, self._val = key, vals            # q, k, v, dout
        return vals

    def drop(self):
        self._key, self._val = None, None
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# float64 reference on a sample of rows -- usable at any N
# ---------------------------------------------------------------------------

def _ref_forward_rows(q, k, v, rows, *, causal, scale, tile=4096):
    """Exact float64 attention for `rows`, streaming keys. O(len(rows) * tile)."""
    dev = "cuda"
    B, H, N, D = q.shape
    qs = q.index_select(2, rows.to(q.device)).to(dev, torch.float64)
    R = qs.shape[2]
    acc = torch.zeros(B, H, R, D, dtype=torch.float64, device=dev)
    m = torch.full((B, H, R), float("-inf"), dtype=torch.float64, device=dev)
    l = torch.zeros(B, H, R, dtype=torch.float64, device=dev)
    rd = rows.to(dev)
    for s in range(0, N, tile):
        e = min(s + tile, N)
        ks = k[:, :, s:e].to(dev, torch.float64)
        vs = v[:, :, s:e].to(dev, torch.float64)
        sc = (qs @ ks.transpose(-1, -2)) * scale
        if causal:
            cols = torch.arange(s, e, device=dev)
            sc = sc.masked_fill(cols[None, None, None, :] > rd[None, None, :, None],
                                float("-inf"))
        mn = torch.maximum(m, sc.amax(-1))
        fin = torch.isfinite(mn)
        ms = torch.where(fin, mn, torch.zeros_like(mn))
        corr = torch.where(fin & torch.isfinite(m), torch.exp(m - ms), torch.zeros_like(m))
        p = torch.where(fin.unsqueeze(-1), torch.exp(sc - ms.unsqueeze(-1)),
                        torch.zeros_like(sc))
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        acc = acc * corr.unsqueeze(-1) + p @ vs
        l = l * corr + p.sum(-1)
        m = torch.where(fin, mn, m)
        del ks, vs, sc, p
    return acc / l.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(-1), l, m


def _ref_backward_dense(q, k, v, dout, *, causal, scale):
    """Dense float64 gradients. O(N^2) -- only for the small-N accuracy table."""
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    N = q.shape[2]
    s = qd @ kd.transpose(-1, -2) * scale
    if causal:
        s = s.masked_fill(torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), 1),
                          float("-inf"))
    o = s.softmax(-1) @ vd
    return torch.autograd.grad(o, [qd, kd, vd], dout.double())


def _relerr(got, ref):
    g = got.to(ref.device).double()
    r = ref.double()
    n = r.norm()
    return (float((g - r).norm() / n.clamp_min(1e-300)),
            float((g - r).abs().max()))


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

@dataclass
class Method:
    key: str
    label: str                 # for plots / tables
    residency: str = "device"  # "device" | "host"
    family: str = "baseline"   # "baseline" | "cqsa"
    note: str = ""
    available: Callable[[], bool] = lambda: True


def _sdpa_backend(name):
    from torch.nn.attention import SDPBackend
    return {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
            "mem": SDPBackend.EFFICIENT_ATTENTION}[name]


def _has_flash():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


METHODS: dict[str, Method] = {
    "sdpa": Method("sdpa", "SDPA (default)",
                   note="PyTorch dispatcher; what a user gets by default"),
    "sdpa_flash": Method("sdpa_flash", "SDPA (flash)",
                         note="SDPA pinned to its FlashAttention backend"),
    "sdpa_mem": Method("sdpa_mem", "SDPA (mem-efficient)",
                       note="SDPA pinned to the memory-efficient backend"),
    "flash": Method("flash", "FlashAttention-2", available=_has_flash,
                    note="reference flash-attn package"),
    # --- Stream-CQSA family -------------------------------------------------
    # Three orthogonal axes. Streaming is NOT one of them: the method is
    # Stream-CQSA, host-resident Q/K/V is definitional, and the all-on-GPU
    # variant is strictly dominated -- it needs the whole input set resident AND
    # adds an accumulator, so it OOMs earlier than the baselines it is meant to
    # rescue. It is kept here only for reproducing the earlier sweeps and is not
    # in DEFAULT_SERIES.
    #
    #   residency   : host (always, for the shipped variants)
    #   accumulator : GPU (fast) | CPU (survives)   <- `acc` in the name
    #   depth       : fixed itr | automatic         <- `--itr-list ...,auto`
    "cqsa_accgpu": Method("cqsa_accgpu", "Stream-CQSA acc=GPU", residency="host",
                          family="cqsa",
                          note="streamed Q/K/V, fp32 accumulator on the GPU -- "
                               "faster, but the accumulator is an O(N) device "
                               "term that no depth of decomposition shrinks"),
    "cqsa_acccpu": Method("cqsa_acccpu", "Stream-CQSA acc=CPU", residency="host",
                          family="cqsa",
                          note="streamed Q/K/V AND fp32 accumulator on the host "
                               "-- the configuration that survives the longest"),
    # Retained for reproducibility only; excluded from the shipped comparison.
    "cqsa_allgpu": Method("cqsa_allgpu", "Stream-CQSA (all on GPU)",
                          residency="device", family="cqsa",
                          note="DEPRECATED: everything device-resident; dominated "
                               "by the baselines, kept only to reproduce old runs"),
}


def run_method(key, q, k, v, dout, *, causal, scale, direction, itr, trace_on):
    """Returns (outputs_tuple, stages_dict, info_dict)."""
    import torch.nn.functional as F

    if key.startswith("cqsa"):
        from stream_cqsa.stable_stream import (stream_cqsa_forward,
                                               stream_cqsa_backward, TraceRecorder)
        host = METHODS[key].residency == "host"
        kw = dict(itr=itr, causal=causal)
        # A column labelled itr=1 must report what itr=1 costs, including when
        # itr=1 does not fit. Two separate mechanisms would otherwise rescue it
        # and file the rescue under the requested depth: the library refines a
        # subproblem that will not fit, and the runner retries the whole backward
        # a level deeper. Both are the right behavior for a library and wrong for
        # a measurement, so a fixed depth turns both off and an out-of-memory
        # result is reported as one.
        fixed_depth = str(itr) != "auto"
        if fixed_depth:
            kw["allow_escalation"] = False
        if host:
            kw["stream_from_host"] = True
        if key.startswith(("cqsa_hostmin", "cqsa_acccpu")):
            kw["low_memory"] = True          # fp32 accumulator to the host too
        # itr="auto" asks the planner, which sizes from *device* free memory and
        # has no host-residency or backward term. That is fine going in, but its
        # estimate can be optimistic -- e.g. at N=8M on an 80 GiB card a
        # monolithic forward (37 GiB) looks affordable, so it returns itr=0, and
        # the *backward* at that depth then OOMs even though itr=1 completes.
        # Escalate on failure rather than recording a defeat the method does not
        # actually suffer: this is also the behavior the paper claims (`itr`
        # rises as the budget tightens), so measuring anything else would be
        # measuring the planner's blind spot instead of the method.
        def _forward_escalating(kw):
            first = kw.get("itr", "auto")
            attempts, depth = [], None
            for step in range(0, 5):
                k2 = dict(kw)
                if step:
                    depth = (1 if depth is None else depth + 1)
                    k2["itr"] = depth
                tr_ = TraceRecorder(enabled=trace_on)
                try:
                    o_, i_ = stream_cqsa_forward(q, k, v, trace=tr_, **k2)
                    i_ = dict(i_); i_["itr_escalations"] = attempts
                    return o_, i_, tr_
                except Exception as exc:                        # noqa: BLE001
                    if "out of memory" not in str(exc).lower():
                        raise
                    if depth is None:
                        # planner's own pick failed; start climbing from its value
                        depth = 0
                    attempts.append(k2.get("itr", first))
                    _drain()
            raise torch.cuda.OutOfMemoryError(
                f"no depth up to {depth} fit (tried {attempts})")

        out, info, tr = _forward_escalating(kw)
        meta = {kk: vv for kk, vv in info.items() if not torch.is_tensor(vv)}
        if direction == "fwd":
            return (out,), (info.get("stage_totals_ms") or {}), meta
        lse = info["lse"]
        o16 = out.to(q.dtype)
        if host:
            o16, lse = o16.cpu(), lse.cpu()
        # The backward takes an int; itr="auto" is resolved by the forward and
        # reported in info["itr"]. Passing the string through would make the
        # backward parse "auto" as an integer.
        kw["itr"] = int(info["itr"])
        # `low_memory` is the forward's spelling of accumulate_on_gpu=False, and
        # the backward has its own parameter for the same choice, so translate
        # rather than drop.
        #
        # This used to drop it, because the streamed backward kept dQ/dK/dV on
        # the host unconditionally and there was nothing to move. That made the
        # acc=GPU and acc=CPU backward columns the SAME measurement -- both host
        # accumulating -- which is why they agreed to within run-to-run noise at
        # every N. The backward now places its three fp32 [B,N,H,D] buffers where
        # it is told, so the columns differ by what their names say: on the device
        # they cost 12 bytes per element that no depth of decomposition reduces.
        bkw = {k2: v2 for k2, v2 in kw.items() if k2 != "low_memory"}
        bkw["accumulate_on_gpu"] = not kw.get("low_memory", False)
        # The backward escalates INDEPENDENTLY of the forward. It carries dO and
        # three gradient tensors on top of what the forward held, so a depth that
        # fits the forward routinely does not fit the backward. This is safe
        # because `lse` is the *global* log-sum-exp: it does not depend on how
        # either pass was decomposed, so the two may run at different depths.
        depth = int(info["itr"]); esc = []
        for _ in range(1 if fixed_depth else 5):
            tr2 = TraceRecorder(enabled=trace_on)
            try:
                binfo = {}
                g = stream_cqsa_backward(q, k, v, dout, o16, lse, trace=tr2,
                                         bwd_info=binfo,
                                         **{**bkw, "itr": depth})
                # The backward's own counters, not the forward's. Without these
                # a silent refinement inside the backward is recorded as the
                # depth that was asked for.
                meta.update({f"bwd_{k3}": v3 for k3, v3 in binfo.items()})
                meta["itr_bwd"] = depth
                meta["itr_bwd_escalations"] = esc
                return g, (tr2.stage_totals_ms() if trace_on else {}), meta
            except Exception as exc:                            # noqa: BLE001
                if "out of memory" not in str(exc).lower():
                    raise
                esc.append(depth); depth += 1
                _drain()
        raise torch.cuda.OutOfMemoryError(
            f"backward: no depth up to {depth} fit (tried {esc})")

    # --- baselines -------------------------------------------------------
    def fwd(qq, kk, vv):
        if key == "flash":
            from flash_attn import flash_attn_func
            return flash_attn_func(qq.transpose(1, 2), kk.transpose(1, 2),
                                   vv.transpose(1, 2), causal=causal,
                                   softmax_scale=scale).transpose(1, 2)
        if key == "sdpa":
            return F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, scale=scale)
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(_sdpa_backend(key.split("_", 1)[1])):
            return F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, scale=scale)

    if direction == "fwd":
        return (fwd(q, k, v),), {}, {}
    qq, kk, vv = (t.detach().requires_grad_() for t in (q, k, v))
    o = fwd(qq, kk, vv)
    return torch.autograd.grad(o, [qq, kk, vv], dout), {}, {}


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------

@dataclass
class Result:
    method: str = ""
    label: str = ""
    family: str = ""
    N: int = 0
    B: int = 0
    H: int = 0
    D: int = 0
    dtype: str = ""
    direction: str = ""
    causal: bool = True
    seed: int = 0
    status: str = "ok"
    error: str = ""
    # time
    ms: float = float("nan")
    ms_all: list[float] = field(default_factory=list)
    stage_ms: dict = field(default_factory=dict)
    # memory (MiB)
    mem_alloc_peak: float = float("nan")
    mem_reserved_peak: float = float("nan")
    mem_workspace: float = float("nan")
    mem_inputs_dev: float = float("nan")
    mem_host_bytes: float = float("nan")
    mem_nvsmi_peak: float = float("nan")
    # accuracy
    acc_rel: float = float("nan")
    acc_max: float = float("nan")
    acc_rel_dq: float = float("nan")
    acc_rel_dk: float = float("nan")
    acc_rel_dv: float = float("nan")
    acc_mode: str = ""
    acc_rows: int = 0
    out_norm: float = float("nan")
    info: dict = field(default_factory=dict)


def _drain():
    """Best-effort allocator drain.

    Never raises. A *hard* CUDA error (as opposed to the allocator's
    OutOfMemoryError) can leave the context unusable, and this runs in a
    `finally`, so an exception here would escape the per-config handler and kill
    a multi-hour sweep at the point where it should merely have recorded a
    failure and moved on.
    """
    try:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except Exception:                                          # noqa: BLE001
        pass


def _dev_bytes(*ts):
    seen, tot = set(), 0
    for t in ts:
        if not torch.is_tensor(t) or t.device.type != "cuda":
            continue
        kk = (t.data_ptr(), t.numel())
        if kk in seen:
            continue
        seen.add(kk)
        tot += t.numel() * t.element_size()
    return tot / MIB


def _host_bytes(*ts):
    seen, tot = set(), 0
    for t in ts:
        if not torch.is_tensor(t) or t.device.type != "cpu":
            continue
        kk = (t.data_ptr(), t.numel())
        if kk in seen:
            continue
        seen.add(kk)
        tot += t.numel() * t.element_size()
    return tot / MIB



def _accuracy(r, key, outs, q, k, v, dout, *, N, causal, scale, acc_rows,
              dense_acc_max_n, direction):
    """Fill the accuracy fields of `r`. May raise; the caller isolates it."""
    if acc_rows <= 0:
        return
    gcpu = torch.Generator().manual_seed(1234)
    idx = torch.randperm(N, generator=gcpu)[:min(acc_rows, N)].sort().values
    if direction == "fwd":
        ref, _, _ = _ref_forward_rows(q, k, v, idx, causal=causal, scale=scale)
        got = outs[0].index_select(2, idx.to(outs[0].device))
        r.acc_rel, r.acc_max = _relerr(got, ref)
        r.acc_mode, r.acc_rows = "fp64_sampled_rows", int(idx.numel())
        del ref, got
        return
    if N > dense_acc_max_n:
        # A dense float64 backward reference is O(N^2); past this N it cannot be
        # formed at all. Backward accuracy is therefore established at small N
        # with many seeds (where it is a property of dtype, not of N) rather
        # than being claimed everywhere.
        r.acc_mode = "skipped_large_N"
        return
    qd = q.cuda() if q.device.type == "cpu" else q
    kd = k.cuda() if k.device.type == "cpu" else k
    vd = v.cuda() if v.device.type == "cpu" else v
    dd = dout.cuda() if dout.device.type == "cpu" else dout
    rq, rk, rv = _ref_backward_dense(qd, kd, vd, dd, causal=causal, scale=scale)
    r.acc_rel_dq, _ = _relerr(outs[0], rq)
    r.acc_rel_dk, _ = _relerr(outs[1], rk)
    r.acc_rel_dv, mx = _relerr(outs[2], rv)
    r.acc_rel = max(r.acc_rel_dq, r.acc_rel_dk, r.acc_rel_dv)
    r.acc_max, r.acc_mode = mx, "fp64_dense"
    del rq, rk, rv


def measure(key, cache, *, N, B, H, D, dtype, direction, causal, seed,
            reps, warmup, acc_rows, dense_acc_max_n, itr) -> Result:
    m = METHODS[key]
    r = Result(method=key, label=m.label, family=m.family, N=N, B=B, H=H, D=D,
               dtype=str(dtype).split(".")[-1], direction=direction,
               causal=causal, seed=seed)
    scale = float(D) ** -0.5
    dest = "cpu" if m.residency == "host" else "cuda"
    _drain()
    try:
        q, k, v, dout = cache.get(N, dtype, seed, dest)
        _drain()
        base = torch.cuda.memory_allocated()

        def call():
            return run_method(key, q, k, v, dout, causal=causal, scale=scale,
                              direction=direction, itr=itr, trace_on=True)

        for _ in range(warmup):
            outs, _, _ = call()
            del outs
        _drain()
        base = torch.cuda.memory_allocated()

        times, outs, stages, info = [], None, {}, {}
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs, stages, info = call()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        r.mem_alloc_peak = torch.cuda.max_memory_allocated() / MIB
        r.mem_reserved_peak = torch.cuda.max_memory_reserved() / MIB
        r.mem_workspace = max(0.0, (torch.cuda.max_memory_allocated() - base) / MIB)
        r.mem_inputs_dev = _dev_bytes(q, k, v, dout)
        r.mem_host_bytes = _host_bytes(q, k, v, dout)
        times.sort()
        r.ms = times[len(times) // 2]
        r.ms_all = [round(t, 3) for t in times]
        r.stage_ms = {kk: round(vv, 2) for kk, vv in (stages or {}).items()}
        r.info = {kk: vv for kk, vv in (info or {}).items()
                  if isinstance(vv, (int, float, str, bool))}
        r.out_norm = float(outs[0].float().norm())

        # ---- accuracy -------------------------------------------------
        # Guarded separately and deliberately. The float64 reference is far more
        # memory-hungry than any method under test (a dense backward reference
        # is O(N^2) in fp64), so an OOM raised here says nothing about whether
        # the *method* fits. Letting it fall through to the outer handler would
        # mark the method OOM and retire it from all larger N -- manufacturing
        # exactly the OOM boundary this experiment is supposed to measure.
        try:
            _accuracy(r, key, outs, q, k, v, dout, N=N, causal=causal,
                      scale=scale, acc_rows=acc_rows,
                      dense_acc_max_n=dense_acc_max_n, direction=direction)
        except torch.cuda.OutOfMemoryError:
            r.acc_mode = "reference_oom"
        except Exception as e:                                 # noqa: BLE001
            r.acc_mode = f"reference_error:{type(e).__name__}"
        finally:
            _drain()
        del outs
    except torch.cuda.OutOfMemoryError as e:
        r.status, r.error = "oom", str(e).split("\n")[0][:200]
    except RuntimeError as e:
        msg = str(e)
        r.status = "oom" if "out of memory" in msg.lower() else "error"
        r.error = msg.split("\n")[0][:200]
    except Exception as e:                                     # noqa: BLE001
        r.status, r.error = "error", f"{type(e).__name__}: {e}".split("\n")[0][:200]
    finally:
        _drain()
    return r


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

PRESETS = {
    # fits an A100-40GB interactive session; verifies the machinery only
    "smoke": dict(n=[4096, 16384], dtypes=["fp16", "bf16"], directions=["fwd", "bwd"],
                  methods=["sdpa", "sdpa_mem", "flash", "cqsa_auto", "cqsa_host"],
                  reps=2, warmup=1, acc_rows=128),
    # accuracy: one N, both precisions, many seeds, dense fp64 backward reference
    "accuracy": dict(n=[8192], dtypes=["fp16", "bf16"], directions=["fwd", "bwd"],
                     methods=["sdpa", "sdpa_flash", "sdpa_mem", "flash",
                              "cqsa_auto", "cqsa_host"],
                     reps=2, warmup=1, acc_rows=256),
    # the paper sweep; sized for A100-80GB. One continuous ladder from where
    # every method fits to where only Stream-CQSA does -- the curve and the OOM
    # stress are the same experiment, so they belong on one axis.
    "full": dict(n=[8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
                    2097152, 4194304],
                 n_fwd=[8192, 16384, 32768, 65536, 131072, 262144, 524288,
                        1048576, 2097152, 4194304, 8388608, 16777216],
                 n_bwd=[8192, 16384, 32768, 65536, 131072, 262144, 524288,
                        1048576, 2097152, 4194304, 8388608],
                 dtypes=["fp16", "bf16"], directions=["fwd", "bwd"],
                 methods=["sdpa", "sdpa_flash", "sdpa_mem", "flash",
                          "cqsa_auto", "cqsa_host"],
                 reps=3, warmup=1, acc_rows=256),
}


def env_note():
    p = torch.cuda.get_device_properties(0)
    d = {"gpu": p.name, "gpu_gib": round(p.total_memory / GIB, 1),
         "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
         "python": platform.python_version(),
         "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
         "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    try:
        import flash_attn
        d["flash_attn"] = flash_attn.__version__
    except Exception:
        d["flash_attn"] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        d["cotenants"] = [x for x in out.split("\n") if x]
    except Exception:
        d["cotenants"] = ["<unavailable>"]
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--n", default="")
    # The two directions hit the memory wall at different N: the backward also
    # holds dO and three gradient tensors, so on an 80 GiB card it OOMs around
    # 8M while the forward still fits (~52 GiB) and needs ~16M to fail. A single
    # ceiling would therefore demonstrate recovery in only one of the two
    # panels, so each direction gets its own ladder.
    ap.add_argument("--n-fwd", default="", help="overrides --n for the forward")
    ap.add_argument("--n-bwd", default="", help="overrides --n for the backward")
    ap.add_argument("--dtypes", default="")
    ap.add_argument("--directions", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("-B", type=int, default=1)
    ap.add_argument("-H", type=int, default=8)
    ap.add_argument("-D", type=int, default=64)
    ap.add_argument("--no-causal", dest="causal", action="store_false", default=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seeds", default="0",
                    help="comma list; >1 gives mean/std for the accuracy table")
    ap.add_argument("--acc-rows", type=int, default=256,
                    help="query rows sampled for the float64 forward reference; 0 disables")
    ap.add_argument("--dense-acc-max-n", type=int, default=4096,
                    help="largest N for which a dense float64 *backward* reference is formed")
    ap.add_argument("--itr", default="auto")
    ap.add_argument("--method-itr", default="",
                    help="Per-method itr override, e.g. 'cqsa_accgpu=2' or "
                         "'cqsa_accgpu=2,cqsa_acccpu=auto'. Without it every cqsa "
                         "method runs every value in --itr-list, which is a cross "
                         "product; this pins one method to one depth so a single "
                         "process can measure two configurations that differ in "
                         "exactly one axis while sharing the generated tensors.")
    ap.add_argument("--itr-list", default="",
                    help="comma list; runs each depth as its own row for the "
                         "Stream-CQSA methods. Needed because itr='auto' picks 0 "
                         "whenever a monolithic call fits, which at small N means "
                         "the decomposition is never exercised.")
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="stddev of the Gaussian inputs. 1.0 (unit variance) is "
                         "the realistic setting. 0.05 makes the softmax nearly "
                         "uniform, which turns the backward's dS = P(dP - D) into "
                         "a catastrophic cancellation and degrades EVERY method "
                         "(see docs/METHODOLOGY.md 1.5) -- do not use it for "
                         "gradient comparisons.")
    ap.add_argument("--cooldown-s", type=float, default=2.0)
    ap.add_argument("--out-dir", default="outputs/paper/run")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    cfg = dict(PRESETS[a.preset]) if a.preset else {}
    ns = [int(x) for x in a.n.split(",") if x] or cfg.get("n", [8192])
    per_dir = {}
    for d, raw in (("fwd", a.n_fwd), ("bwd", a.n_bwd)):
        vals = [int(x) for x in raw.split(",") if x] or cfg.get(f"n_{d}")
        per_dir[d] = sorted(set(vals)) if vals else None
    all_ns = sorted(set(ns) | {n for v in per_dir.values() if v for n in v})
    dts = [x for x in a.dtypes.split(",") if x] or cfg.get("dtypes", ["fp16"])
    dirs = [x for x in a.directions.split(",") if x] or cfg.get("directions", ["fwd"])
    mts = [x for x in a.methods.split(",") if x] or cfg.get("methods", list(METHODS))
    reps = cfg.get("reps", a.reps) if a.preset else a.reps
    warm = cfg.get("warmup", a.warmup) if a.preset else a.warmup
    arows = cfg.get("acc_rows", a.acc_rows) if a.preset else a.acc_rows
    seeds = [int(x) for x in a.seeds.split(",") if x]
    itr = a.itr if a.itr == "auto" else int(a.itr)
    # "auto" is a legal entry: automatic depth is the shipped default, so the
    # sweep has to be able to measure it alongside pinned depths.
    itr_list = [(x.strip() if x.strip() == "auto" else int(x))
                for x in a.itr_list.split(",") if x.strip()] or [itr]
    method_itr: dict[str, list] = {}
    for spec in (x.strip() for x in a.method_itr.split(",") if x.strip()):
        mname, _, vals = spec.partition("=")
        method_itr[mname.strip()] = [(y.strip() if y.strip() == "auto" else int(y))
                                     for y in vals.split("|") if y.strip()]
    DT = {"fp16": torch.float16, "bf16": torch.bfloat16}

    for k in mts:
        if k not in METHODS:
            print(f"unknown method {k!r}; known: {list(METHODS)}", file=sys.stderr)
            return 2

    os.makedirs(a.out_dir, exist_ok=True)
    meta = env_note()
    meta.update(dict(B=a.B, H=a.H, D=a.D, causal=a.causal, reps=reps, warmup=warm,
                     acc_rows=arows, seeds=seeds, itr=str(itr), n=ns,
                     n_per_direction=per_dir, dtypes=dts,
                     input_scale=a.input_scale, itr_list=itr_list,
                     method_itr={k: [str(x) for x in v] for k, v in method_itr.items()},
                     directions=dirs, methods=mts, tag=a.tag,
                     started=time.strftime("%Y-%m-%d %H:%M:%S")))
    with open(os.path.join(a.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    jsonl = os.path.join(a.out_dir, "results.jsonl")

    print(f"# {meta['gpu']} ({meta['gpu_gib']} GiB)  torch {meta['torch']}  "
          f"flash-attn {meta['flash_attn']}")
    print(f"# alloc_conf={meta['alloc_conf'] or '<unset>'}  cotenants={meta['cotenants']}")
    print(f"# B={a.B} H={a.H} D={a.D} causal={a.causal} reps={reps} seeds={seeds}")
    print(f"# out={a.out_dir}\n")
    hdr = (f"{'N':>9} {'dtype':>5} {'dir':>4} {'method':>22} {'ms':>10} "
           f"{'peak MiB':>9} {'wkspc':>8} {'host MiB':>9} {'rel err':>9}  status")
    print(hdr)
    print("-" * len(hdr))

    cache = InputCache(a.B, a.H, a.D, scale=a.input_scale)
    n_rows = 0
    # N is the OUTER loop on purpose. The sweep runs for hours and can be cut off
    # by a wall-clock limit, so it walks N upward completing every
    # (dtype, direction, method) at each step: a truncated run then yields a
    # complete sweep up to some N rather than all of fp16 and none of bf16.
    # A method that OOMs cannot recover at larger N, so it is retired -- per
    # (dtype, direction), since the boundary differs between them.
    dead: dict[tuple[str, str], set[str]] = {(dt, d): set() for dt in dts for d in dirs}
    for N in all_ns:
        for dt in dts:
            for direction in dirs:
                if per_dir.get(direction) is not None:
                    if N not in per_dir[direction]:
                        continue
                elif N not in ns:
                    continue
                for seed in seeds:
                    for key, itr_use in [(k, i) for k in mts
                                         for i in (method_itr.get(k, itr_list)
                                                   if k.startswith("cqsa")
                                                   else [itr])]:
                        if not METHODS[key].available():
                            continue
                        # Retire on the (method, itr) pair, not the bare method:
                        # a deeper itr splits into smaller subproblems and can
                        # succeed exactly where a shallower one OOMed. Keying on
                        # `key` alone made an itr=1 OOM suppress itr=2, which
                        # discards the very point that shows deeper decomposition
                        # recovering.
                        if (key, itr_use) in dead[(dt, direction)]:
                            print(f"{N:>9} {dt:>5} {direction:>4} "
                                  f"{METHODS[key].label:>22} {'--':>10} {'':>9} "
                                  f"{'':>8} {'':>9} {'':>9}  skipped (OOM at smaller N)",
                                  flush=True)
                            continue
                        r = measure(key, cache, N=N, B=a.B, H=a.H, D=a.D,
                                    dtype=DT[dt], direction=direction,
                                    causal=a.causal, seed=seed, reps=reps,
                                    warmup=warm, acc_rows=arows,
                                    dense_acc_max_n=a.dense_acc_max_n, itr=itr_use)
                        r.info["itr_requested"] = str(itr_use)
                        if len(itr_list) > 1 and key.startswith("cqsa"):
                            r.method = f"{key}_itr{itr_use}"
                            r.label = f"{METHODS[key].label} itr={itr_use}"
                        with open(jsonl, "a") as fh:
                            fh.write(json.dumps(asdict(r)) + "\n")
                        n_rows += 1
                        if r.status == "ok":
                            err = "-" if r.acc_rel != r.acc_rel else f"{r.acc_rel:.2e}"
                            extra = ""
                            if r.family == "cqsa" and "itr" in r.info:
                                extra = f"  itr={r.info['itr']}"
                            print(f"{N:>9} {dt:>5} {direction:>4} "
                                  f"{METHODS[key].label:>22} {r.ms:10.1f} "
                                  f"{r.mem_alloc_peak:9.0f} {r.mem_workspace:8.0f} "
                                  f"{r.mem_host_bytes:9.0f} {err:>9}  ok{extra}",
                                  flush=True)
                            if r.stage_ms:
                                top = ", ".join(f"{kk} {vv:.0f}"
                                                for kk, vv in list(r.stage_ms.items())[:6])
                                print(f"{'':>9} {'':>5} {'':>4} {'':>22}  stages: {top}",
                                      flush=True)
                        else:
                            print(f"{N:>9} {dt:>5} {direction:>4} "
                                  f"{METHODS[key].label:>22} {'--':>10} {'':>9} "
                                  f"{'':>8} {'':>9} {'':>9}  {r.status.upper()}: "
                                  f"{r.error[:40]}", flush=True)
                            if r.status == "oom":
                                dead[(dt, direction)].add((key, itr_use))
                        if a.cooldown_s:
                            time.sleep(a.cooldown_s)
                    cache.drop()
        print(flush=True)
    print(f"wrote {jsonl}  ({n_rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
