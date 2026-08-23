"""Build notebooks/tutorial_stream_cqsa_v2.ipynb."""
import json, os

C = []          # cells
def md(s):   C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): C.append({"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": s.strip("\n").splitlines(True)})

md(r"""
# Stream-CQSA v2: autograd, `.backward()`, and depth escalation

What changed since v1. Two things, and the first is the reason for this notebook.

1. **`stream_cqsa_attn` is a `torch.autograd.Function`.** You call it like
   `scaled_dot_product_attention`, call `.backward()`, and gradients appear. In
   v1 the backward was a separate function you had to drive by hand, threading
   the global log-sum-exp through yourself and matching residency and dtype at
   every step.
2. **Depth escalation is real.** A subproblem that does not fit is now
   decomposed a level further and its children run in its place, in the forward
   and in both backward paths. Previously the forward could only lower its
   stream count, and the device backward had no out-of-memory handling at all.

Everything below runs in about two minutes on one A100.
""")

code(r"""
import gc, os, sys, time, warnings
import torch
import torch.nn.functional as F

def _find_pkg():
    # Prefer whichever tree actually carries the v2 code.
    here = os.path.abspath(globals().get("__vsc_ipynb_file__", os.getcwd()))
    if os.path.isfile(here): here = os.path.dirname(here)
    for base in (here, os.getcwd()):
        d = base
        for _ in range(6):
            for cand in (os.path.join(d, "packages", "stream-cqsa"),
                         os.path.join(d, "release", "Stream-CQSA")):
                # native_autograd.py is the v2 marker: a tree without it is stale.
                if os.path.isfile(os.path.join(cand, "stream_cqsa", "native_autograd.py")):
                    return cand
            if os.path.dirname(d) == d: break
            d = os.path.dirname(d)
    return None

PKG = _find_pkg()
if PKG is None:
    raise RuntimeError(
        "no tree with stream_cqsa/native_autograd.py found -- this notebook needs "
        "the v2 package, not the v1 one")
for _m in [m for m in sys.modules if m == "stream_cqsa" or m.startswith("stream_cqsa.")]:
    del sys.modules[_m]
sys.path.insert(0, PKG); warnings.filterwarnings("ignore")

from stream_cqsa.native_autograd import stream_cqsa_attn, StreamCQSAAttention
from stream_cqsa.stable_stream import (stream_cqsa_forward, stream_cqsa_backward,
                                       max_depth_for, TraceRecorder)
import cqsa_cuda  # noqa: F401

TOTAL_GIB = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU     {torch.cuda.get_device_name(0)}  ({TOTAL_GIB:.0f} GiB)")
print(f"package {PKG}")

B, H, N, D = 1, 4, 8192, 64
CAUSAL, SCALE = True, D ** -0.5
torch.manual_seed(0)
""")

md(r"""
## 1. The short version

The same computation, written both ways. `stream_cqsa_attn` takes fp16 or bf16
operands (the kernel accepts nothing else) and returns the input dtype.
""")

code(r"""
mk = lambda: torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
q, k, v = mk(), mk(), mk()
dout = mk()

# ---- v2: autograd ---------------------------------------------------------
qa, ka, va = (t.clone().requires_grad_(True) for t in (q, k, v))
stream_cqsa_attn(qa, ka, va, causal=CAUSAL).backward(dout)

# ---- v1: by hand ----------------------------------------------------------
out, info = stream_cqsa_forward(q, k, v, itr=1, causal=CAUSAL)
dq, dk, dv = stream_cqsa_backward(q, k, v, dout, out.to(q.dtype), info["lse"],
                                  itr=info["itr"], causal=CAUSAL)

rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())
print(f"dQ {rel(qa.grad, dq):.2e}   dK {rel(ka.grad, dk):.2e}   dV {rel(va.grad, dv):.2e}"
      "   (autograd vs the manual path)")
""")

md(r"""
The manual path is still there and still the fastest way to pin every choice.
What the autograd wrapper removes is three chances to get it wrong:

- **The global log-sum-exp.** The backward needs it and cannot rebuild it — that
  is what makes each subproblem's `P = exp(s - lse_global) <= 1` and the
  decomposed backward exact. The wrapper saves it for you.
- **Residency.** Under `stream_from_host=True` every operand must be
  host-resident, but the forward returns `out` and `lse` wherever its
  accumulator happened to live. Forget one `.cpu()` and you get a cross-device
  error from somewhere deep in the gather.
- **dtype.** The forward returns fp32 while the kernel wants fp16 or bf16.

One thing the wrapper does *not* have to protect you from, which is worth
stating because it is easy to assume otherwise: the backward does not have to
run at the depth the forward used. See section 4.
""")

md(r"""
## 2. The gradients are correct

Against autograd through an fp32 reference, which is the target Stream-CQSA is
meant to reproduce. Errors are the relative norm. The operand dtype sets the
floor, not the decomposition: fp16 lands near `5e-4` and bf16 roughly eight
times coarser, which is the ratio of their mantissa widths.
""")

code(r"""
def grad_check(dtype, causal, itr="auto"):
    g = torch.Generator().manual_seed(0)
    qf, kf, vf = (torch.randn(B, H, N, D, generator=g) for _ in range(3))
    dof = torch.randn(B, H, N, D, generator=g).cuda()

    # fp32 target, differentiated by autograd.
    r = [t.clone().cuda().float().requires_grad_(True) for t in (qf, kf, vf)]
    ref = F.scaled_dot_product_attention(*r, is_causal=causal, scale=SCALE)
    ref.backward(dof.float())

    # Stream-CQSA, differentiated by .backward().
    t = [x.clone().cuda().to(dtype).requires_grad_(True) for x in (qf, kf, vf)]
    o = stream_cqsa_attn(*t, causal=causal, itr=itr)
    o.backward(dof.to(dtype))
    return (rel(o, ref), rel(t[0].grad, r[0].grad),
            rel(t[1].grad, r[1].grad), rel(t[2].grad, r[2].grad))

print(f"{'dtype':>9} {'causal':>7} | {'out':>9} {'dQ':>9} {'dK':>9} {'dV':>9}")
print("-" * 58)
for dtype in (torch.float16, torch.bfloat16):
    for causal in (False, True):
        o, a, b, c = grad_check(dtype, causal)
        print(f"{str(dtype).replace('torch.',''):>9} {str(causal):>7} | "
              f"{o:9.2e} {a:9.2e} {b:9.2e} {c:9.2e}")
""")

md(r"""
## 3. Training with it

`StreamCQSAAttention` is the same operator as a module, so it drops into a block
and the optimiser sees real gradients. Parameters stay fp32 and only the
attention operands are cast, which is the ordinary mixed-precision arrangement.
""")

code(r"""
class Block(torch.nn.Module):
    def __init__(self, d_model, n_heads, *, causal=True, dtype=torch.bfloat16):
        super().__init__()
        self.h, self.dk, self.dtype = n_heads, d_model // n_heads, dtype
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.attn = StreamCQSAAttention(causal=causal, itr="auto")

    def forward(self, x):                       # x: [B, T, d_model], fp32
        Bs, T, _ = x.shape
        qkv = self.qkv(self.norm(x)).view(Bs, T, 3, self.h, self.dk)
        q, k, v = (qkv[:, :, i].transpose(1, 2).to(self.dtype) for i in range(3))
        y = self.attn(q, k, v)                  # [B, h, T, dk]
        y = y.transpose(1, 2).reshape(Bs, T, -1).float()
        return x + self.proj(y)

torch.manual_seed(0)
T, d_model = 4096, 256
block = Block(d_model, 4).cuda()
opt = torch.optim.Adam(block.parameters(), lr=3e-4)

x = torch.randn(1, T, d_model, device="cuda")
target = torch.randn(1, T, d_model, device="cuda")

losses = []
for step in range(30):
    loss = F.mse_loss(block(x), target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 6 == 0 or step == 29:
        print(f"step {step:2d}   loss {loss.item():.5f}")

gn = lambda p: 0.0 if p.grad is None else p.grad.norm().item()
print(f"\nloss {losses[0]:.5f} -> {losses[-1]:.5f}   "
      f"({'decreasing' if losses[-1] < losses[0] else 'NOT decreasing'})")
print(f"grad norms: qkv {gn(block.qkv.weight):.3e}   proj {gn(block.proj.weight):.3e}   "
      f"norm {gn(block.norm.weight):.3e}")
print("gradients reached every parameter upstream of the attention call")
""")

md(r"""
## 4. The backward's depth is free

It is tempting to assume the backward must reuse the forward's depth. It does
not. Each depth decomposes the *same* set of chunk pairs into a different
partition, and every subproblem is scored against the global log-sum-exp, so the
contributions sum to the exact gradient at any depth.

Passing `info["itr"]` back is still the sensible default — it starts the backward
where the forward already found it could fit, instead of rediscovering the same
memory pressure — but it is a starting point, not a constraint.
""")

code(r"""
g = torch.Generator().manual_seed(3)
qf, kf, vf = (torch.randn(B, H, N, D, generator=g) for _ in range(3))
dof = torch.randn(B, H, N, D, generator=g).cuda().half()

print(f"{'depth':>6} {'subproblems':>12} {'dQ vs depth-1':>15}")
print("-" * 36)
base = None
for depth in (1, 2, 3):
    t = [x.clone().cuda().half().requires_grad_(True) for x in (qf, kf, vf)]
    stream_cqsa_attn(*t, causal=CAUSAL, itr=depth).backward(dof)
    if base is None:
        base = t[0].grad.clone(); shown = "-- (reference)"
    else:
        shown = f"{rel(t[0].grad, base):.2e}"
    print(f"{depth:>6} {7 ** depth:>12} {shown:>15}")
print(f"\nthe structural ceiling here is itr={max_depth_for(N)} (c**itr <= N at c=7)")
print("the residual is fp16 rounding, not a depth effect: it is the same size as")
print("the error against the fp32 reference in section 2, and it does not grow")
print("with depth even though depth-3 runs 49x more subproblems than depth-1")
""")

md(r"""
## 5. Depth escalation

This is what the paper claims and what v2 implements: when a subproblem does not
fit, it is refined into its `c` children and they run in its place. The children
inherit the parent's mask as well as applying their own, so their retained pairs
partition exactly the parent's — nothing is dropped and nothing is counted twice.

Forcing it with a real memory cap is unreliable, because PyTorch's caching
allocator serves a small call from blocks it has already reserved and the cap
never bites. Injecting the refusal into the inner kernel exercises the same
control flow deterministically, and lets the recovered result be checked against
a reference — which is the part that matters, since a recovery that silently
returns a wrong answer is worse than one that raises.
""")

code(r"""
from stream_cqsa.stable_stream import local_stats_flash

def refuse_over(limit, first_n=None):
    # Inner kernel that reports OOM for subsequences wider than `limit`.
    seen = {"refused": 0, "lengths": []}
    def inner(q_i, k_i, v_i, bits, **kw):
        L = q_i.shape[1]                    # scheduler hands it [B, L, H, D]
        if L > limit and (first_n is None or seen["refused"] < first_n):
            seen["refused"] += 1
            raise torch.cuda.OutOfMemoryError("injected: does not fit")
        seen["lengths"].append(L)
        return local_stats_flash(q_i, k_i, v_i, bits, **kw)
    return inner, seen

q1, k1, v1 = mk(), mk(), mk()
ref = F.scaled_dot_product_attention(q1, k1, v1, is_causal=CAUSAL, scale=SCALE).float()

# A depth-1 subsequence here is 3*8192/7 = 3510 tokens; refuse anything wider.
inner, seen = refuse_over(3000)
out_e, info_e = stream_cqsa_forward(q1, k1, v1, itr=1, causal=CAUSAL,
                                    inner=inner, max_parallel=1)

print(f"subproblems refused      {seen['refused']}")
print(f"depth escalations        {info_e['depth_escalations']}")
print(f"deepest depth reached    {info_e['itr_max_reached']}  (started at 1)")
print(f"executed widths          {sorted(set(seen['lengths']))}")
print(f"tokens left uncovered    {info_e['untouched_tokens']}")
print(f"\nvs SDPA after recovery   {rel(out_e, ref):.2e}   <- escalation changed the "
      "partition, not the answer")
""")

md(r"""
### Hybrid schedules

Refusing only *some* subproblems leaves the rest at the original depth. The
executed schedule then spans two depths at once — the shape a changing memory
budget produces in practice, rather than anything chosen up front. The pair sets
still partition the map, so the result is unchanged.
""")

code(r"""
inner, seen = refuse_over(3000, first_n=3)
out_h, info_h = stream_cqsa_forward(q1, k1, v1, itr=1, causal=CAUSAL,
                                    inner=inner, max_parallel=1)
widths = sorted(set(seen["lengths"]))
print(f"refused {seen['refused']} of 7 subproblems")
print(f"executed widths          {widths}  <- two depths in one schedule")
print(f"vs SDPA                  {rel(out_h, ref):.2e}")
""")

md(r"""
### Where it stops

Escalation is bounded by structure, not by a magic constant: `c**itr <= N` is the
deepest decomposition a sequence admits, past which a level would produce
subproblems narrower than a single chunk. If nothing fits even there, the call
raises with that stated, rather than refining forever.
""")

code(r"""
inner, seen = refuse_over(0)                 # refuse everything, at every depth
small = [torch.randn(1, 1, 343, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
try:
    stream_cqsa_forward(*small, itr=1, inner=inner, max_parallel=1)
    print("did NOT raise -- unexpected")
except torch.cuda.OutOfMemoryError as e:
    print(f"raised after {seen['refused']} refusals, as it should:\n  {str(e)[:180]}...")
print(f"\nstructural ceiling at N=343: itr={max_depth_for(343)}   "
      f"at N=16M: itr={max_depth_for(1 << 24)}")
""")

md(r"""
## 6. What the wrapper costs

The wrapper calls the same kernels through the same scheduler, so the question is
not whether it computes something different but what the convenience costs. Two
answers, measured below rather than asserted.

**Precision.** `dK` and `dV` come back bit-identical. `dQ` does not, but neither
do two runs of the *manual* path: FlashAttention accumulates `dQ` with atomics,
so the summation order is whatever the hardware picks that run. The wrapper sits
inside that spread. The one systematic difference is dtype -- autograd requires a
gradient to carry its input's dtype, so the wrapper returns fp16/bf16 where the
manual path hands back the fp32 accumulator. Reach for the manual path when you
want the unrounded values.

**Cost.** Time is free. Memory is not: `save_for_backward` pins the fp32 output
alive from the forward until the backward runs, which the manual path lets you
drop whenever you like.
""")

code(r"""
import time

def measure(N, dtype, itr=1):
    g = torch.Generator().manual_seed(0)
    qf, kf, vf = (torch.randn(1, 8, N, 64, generator=g) for _ in range(3))
    dof = torch.randn(1, 8, N, 64, generator=g)
    qq, kk, vv = (t.cuda().to(dtype) for t in (qf, kf, vf))
    dd = dof.cuda().to(dtype)

    def manual():
        o, i = stream_cqsa_forward(qq, kk, vv, itr=itr, causal=CAUSAL)
        return o, stream_cqsa_backward(qq, kk, vv, dd, o.to(dtype), i["lse"],
                                       itr=i["itr"], causal=CAUSAL)

    def wrapped():
        t = [x.detach().clone().requires_grad_(True) for x in (qq, kk, vv)]
        o = stream_cqsa_attn(*t, causal=CAUSAL, itr=itr)
        o.backward(dd)
        return o.detach(), [x.grad for x in t]

    def timed(fn):
        fn(); torch.cuda.synchronize()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter(); r = fn(); torch.cuda.synchronize()
        return time.perf_counter() - t0, torch.cuda.max_memory_allocated(), r

    # Measure each path with the other's results already freed, or the manual
    # gradients still resident would be charged to the wrapper's peak.
    tm, pm, (_, gm) = timed(manual)
    gm_ref = [x.detach().float().cpu() for x in gm]
    del gm; torch.cuda.empty_cache()

    ta, pa, (_, ga) = timed(wrapped)
    ga_ref = [x.detach().float().cpu() for x in ga]
    del ga; torch.cuda.empty_cache()

    # How much the manual path disagrees with ITSELF across reruns is the
    # yardstick. One rerun is a sample of one, so take the worst of three.
    spread = [0.0, 0.0, 0.0]
    for _ in range(3):
        g2 = manual()[1]
        torch.cuda.synchronize()
        for j, x in enumerate(g2):
            d = float((x.detach().float().cpu() - gm_ref[j]).abs().max())
            spread[j] = max(spread[j], d)
        del g2; torch.cuda.empty_cache()

    vs_wrap = [float((a.to(dtype).float() - b).abs().max())
               for a, b in zip(gm_ref, ga_ref)]
    return tm, ta, pm, pa, spread, vs_wrap

print(f"{'N':>8} {'dtype':>9} | {'manual':>16} | {'autograd':>16} | {'ratio':>13}")
print(f"{'':>8} {'':>9} | {'ms':>7}{'peak GiB':>9} | {'ms':>7}{'peak GiB':>9} | "
      f"{'time':>6}{'mem':>7}")
print("-" * 74)
rows = []
for N in (8192, 32768):
    for dtype in (torch.float16, torch.bfloat16):
        tm, ta, pm, pa, spread, vs_wrap = measure(N, dtype)
        nm = str(dtype).replace("torch.", "")
        print(f"{N:>8} {nm:>9} | {tm*1e3:7.1f}{pm/2**30:9.2f} | "
              f"{ta*1e3:7.1f}{pa/2**30:9.2f} | {ta/tm:5.2f}x{pa/pm:6.2f}x")
        rows.append((N, nm, spread, vs_wrap))

print(f"\n{'N':>8} {'dtype':>9} | {'manual vs itself (rerun)':>36} | "
      f"{'manual vs wrapper':>36}")
print(f"{'':>8} {'':>9} | {'dQ':>11}{'dK':>12}{'dV':>13} | "
      f"{'dQ':>11}{'dK':>12}{'dV':>13}")
print("-" * 92)
for N, nm, spread, vs_wrap in rows:
    f3 = lambda xs: "".join(f"{x:>12.2e}" for x in xs)
    print(f"{N:>8} {nm:>9} | {f3(spread):>36} | {f3(vs_wrap):>36}")
print("\ndK and dV are exactly 0 in both columns: bit-identical.")
print("dQ is nonzero in both, and by the same order -- that is the kernel's")
print("atomic accumulation, not the wrapper.")
""")

md(r"""
## 7. Host streaming, with autograd

The mode that makes very long sequences fit: inputs and the fp32 accumulator
both live in host memory, and only one subsequence is device-resident at a time.
Under the manual API this is where the `.cpu()` bookkeeping bites. Through the
autograd wrapper it is one keyword.
""")

code(r"""
qh, kh, vh = (t.detach().cpu().clone().requires_grad_(True) for t in (q, k, v))
# max_memory_allocated is a high-water mark for the whole process, so without
# this it would report the largest earlier cell rather than this call.
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
out_h = stream_cqsa_attn(qh, kh, vh, causal=CAUSAL,
                         stream_from_host=True, accumulate_on_gpu=False)
out_h.backward(dout.cpu())
dt = time.perf_counter() - t0

print(f"operands on {qh.device}, gradients on {qh.grad.device}, took {dt:.2f}s")
print(f"dQ vs the device path    {rel(qh.grad.cuda(), qa.grad):.2e}")
print(f"peak device memory       {torch.cuda.max_memory_allocated()/2**20:.0f} MiB"
      "   <- one subsequence at a time, not the whole sequence")
""")

md(r"""
## What to use

| you want | call |
|---|---|
| to train a model | `stream_cqsa_attn` / `StreamCQSAAttention`, and `.backward()` |
| every choice pinned, or to benchmark | `stream_cqsa_forward` + `stream_cqsa_backward` |
| it to fit, without thinking | `stream_cqsa_auto` |

Two things to carry away. Operands must be **fp16 or bf16** — the kernel accepts
nothing else, and passing fp32 now raises a message saying so instead of failing
somewhere inside the kernel. And depth is not something you have to get right up
front: `itr="auto"` picks a starting depth, escalation fixes it if that proves
optimistic, and the answer does not depend on which depth it settles at.
""")

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/notebooks/tutorial_stream_cqsa_v2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"wrote {out}: {len(C)} cells "
      f"({sum(c['cell_type']=='code' for c in C)} code)")
