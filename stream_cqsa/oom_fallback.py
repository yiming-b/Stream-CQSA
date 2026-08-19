"""
Drop-in OOM recovery: run the normal attention path, fall back to Stream-CQSA.

    from stream_cqsa.oom_fallback import attention_oom_safe
    out = attention_oom_safe(q, k, v, causal=True)      # same shape/dtype as SDPA

Why this is not just `try: sdpa(...) except OutOfMemoryError: stream_cqsa(...)`
-----------------------------------------------------------------------------
Four things have to happen between the exception and the retry, and skipping any
of them makes the fallback fail too:

1. **Drain the allocator.** The failed call leaves its partial allocations in the
   caching allocator. Without `empty_cache()` the retry meets the same wall.

2. **Get the inputs off the device.** This is the one that actually matters, and
   it is not optional. Stream-CQSA *device-resident* needs the whole input set
   resident exactly like the baseline does, so it OOMs at the same N -- measured:
   at N=8M both FlashAttention and device-resident Stream-CQSA fail, and only the
   host-streamed configuration returns a result. Recovery therefore requires
   Q/K/V in host memory.

   But the caller still holds references to the device tensors, so a plain
   `q.cpu()` copies without freeing anything. `release_inputs=True` rebinds
   `t.data` to host storage, which frees the device allocation even though the
   caller's variable is still alive -- at the cost of mutating the caller's
   tensors. It is opt-in for that reason, and it is what makes the difference
   between recovering and failing again.

3. **Match the output contract.** `stream_cqsa_forward` returns `(out, info)` in
   fp32; SDPA returns a bare tensor in the input dtype. The wrapper reconciles
   both so it substitutes cleanly.

4. **Let the planner pick the depth.** `itr="auto"` reads free memory and chooses;
   a hardcoded depth either fails or over-decomposes.

Limitations, stated plainly
---------------------------
* **Forward only.** There is no `torch.autograd.Function` wrapper yet, so the
  returned tensor does not carry a backward. Use `stream_cqsa_backward`
  explicitly (it needs `info["lse"]`, so pass `return_info=True`).
* `dropout_p` must be 0 and GQA/MQA is unsupported, matching the kernel.
* With `release_inputs=False` (the default) recovery only succeeds if the
  transient workspace was what overflowed, not the inputs themselves.
"""
from __future__ import annotations

import gc
import warnings
from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["attention_oom_safe", "stream_cqsa_auto", "ESCALATION"]

_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, _OOM) or "out of memory" in str(exc).lower()


def _drain():
    gc.collect()
    torch.cuda.empty_cache()


def attention_oom_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    primary=None,
    release_inputs: bool = False,
    max_itr: int = 4,
    return_info: bool = False,
    verbose: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Attention that survives an out-of-memory failure of the normal path.

    q/k/v are ``[B, H, N, D]``. Returns the output in the inputs' dtype and
    device, so it substitutes for ``scaled_dot_product_attention``.

    `primary` is the fast path to try first (default: SDPA). `release_inputs`
    permits moving Q/K/V to host memory on the fallback -- see the module
    docstring; without it, inputs that do not fit cannot be recovered.
    """
    from .stable_stream import stream_cqsa_forward

    if q.dim() != 4:
        raise ValueError(f"expected [B, H, N, D], got {tuple(q.shape)}")
    out_dtype, out_device = q.dtype, q.device
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5
    info: dict[str, Any] = {"path": "primary", "recovered": False}

    if primary is None:
        def primary(qq, kk, vv):
            return F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal,
                                                  scale=scale)

    if q.device.type == "cuda":
        try:
            out = primary(q, k, v)
            return (out, info) if return_info else out
        except Exception as exc:                                # noqa: BLE001
            if not _is_oom(exc):
                raise
            if verbose:
                warnings.warn(
                    f"attention OOMed ({str(exc).split(chr(10))[0][:80]}); "
                    "falling back to Stream-CQSA", RuntimeWarning, stacklevel=2)
            info["primary_error"] = str(exc).split("\n")[0][:200]
            _drain()

    # ---- fallback ---------------------------------------------------------
    if release_inputs and q.device.type == "cuda":
        # Rebind the caller's tensors onto host storage. `.cpu()` alone would
        # copy and leave the device allocation alive behind the caller's own
        # reference, which is the usual reason a retry OOMs as well.
        for t in (q, k, v):
            t.data = t.data.to("cpu")
        _drain()

    host = q.device.type == "cpu"

    # Escalate the decomposition depth until it fits. Two reasons this cannot
    # just be itr="auto":
    #
    #   * The planner sizes itself from torch.cuda.mem_get_info, which reports
    #     DEVICE-WIDE free memory. It does not see a per-process cap
    #     (set_per_process_memory_fraction), another tenant's allocation, or a
    #     container limit, so it can report far more headroom than this process
    #     can actually use and pick too shallow a depth.
    #   * itr=0 means "no decomposition" -- a single monolithic call, which is
    #     exactly what just failed. Accepting it here guarantees a second OOM.
    #
    # So: start at least at 1, and on OOM go deeper. Each level shrinks the
    # subproblems by ~c/l^2, which is the guardrail behaviour the method claims.
    attempts, out, cinfo = [], None, None
    for depth in range(1, max_itr + 1):
        try:
            out, cinfo = stream_cqsa_forward(q, k, v, itr=depth, causal=causal,
                                             scale=scale, stream_from_host=host)
            break
        except Exception as exc:                                # noqa: BLE001
            if not _is_oom(exc):
                raise
            attempts.append(depth)
            if verbose:
                warnings.warn(f"itr={depth} also OOMed; trying itr={depth + 1}",
                              RuntimeWarning, stacklevel=2)
            _drain()
    if out is None:
        raise torch.cuda.OutOfMemoryError(
            f"Stream-CQSA could not fit even at itr={max_itr}. "
            + ("Pass release_inputs=True so Q/K/V can move to host memory."
               if not host else
               "Q/K/V are already host-resident; the device cannot hold even one "
               "subproblem at this depth.")) from None
    info.update({"path": "stream_cqsa", "recovered": True,
                 "itr": cinfo.get("itr"), "streamed": host,
                 "n_subproblems": cinfo.get("n_subproblems"),
                 "itr_attempts_oomed": attempts,
                 "plan_reason": cinfo.get("plan_reason")})
    if return_info:
        info["lse"] = cinfo.get("lse")       # the backward needs this
    out = out.to(device=out_device, dtype=out_dtype)
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------
# The no-brainer entry point
# ---------------------------------------------------------------------------

# Ordered cheapest-first. Each rung relaxes ONE thing, and the order matters:
# deeper `itr` is tried before relocating anything, because moving a term to the
# host costs bus traffic on every subproblem that touches it.
#
# The crucial point is that `itr` alone is NOT sufficient. Device memory is three
# terms and deeper decomposition only shrinks one of them:
#
#     inputs Q/K/V        O(N)  -- invariant in itr; needs stream_from_host
#     fp32 accumulator    O(N)  -- invariant in itr; needs accumulate_on_gpu=False
#     in-flight pieces    O(N/l^itr) -- this is the only one itr touches
#
# So a ladder that only deepens `itr` stalls at a floor of (inputs + accumulator)
# and OOMs forever after. That floor is why the "streamed" configuration still
# fails under a tight cap while "min-device" does not.
ESCALATION = [
    dict(itr=1),
    dict(itr=2),
    dict(itr=1, stream_from_host=True),
    dict(itr=2, stream_from_host=True),
    dict(itr=2, stream_from_host=True, accumulate_on_gpu=False),
    dict(itr=3, stream_from_host=True, accumulate_on_gpu=False),
    dict(itr=4, stream_from_host=True, accumulate_on_gpu=False),
]


def stream_cqsa_auto(q, k, v, *, causal=False, scale=None, return_info=False,
                     ladder=None, verbose=False):
    """
    Exact attention that keeps trying until it fits.

    Walks `ESCALATION` cheapest-first, relaxing one constraint per rung, and
    returns the first configuration that completes. Inputs may start on either
    device; rungs that need host residency relocate them (rebinding `.data`, so
    the device allocation is actually released rather than merely copied).

    This is the "just run it" entry point. It is deliberately not the fastest
    path -- when a monolithic call fits, call the monolithic kernel.

    Returns the output in the inputs' original dtype, on their original device
    (or `(out, info)` with `return_info=True`; `info["lse"]` is required by
    `stream_cqsa_backward`).
    """
    from .stable_stream import stream_cqsa_forward

    out_dtype = q.dtype
    out_device = q.device
    tried = []
    for cfg in (ladder or ESCALATION):
        need_host = cfg.get("stream_from_host", False)
        try:
            if need_host and q.device.type == "cuda":
                for t in (q, k, v):
                    t.data = t.data.to("cpu")     # release, not copy
                _drain()
            if not need_host and q.device.type == "cpu":
                continue                          # cannot run device-resident now
            out, info = stream_cqsa_forward(q, k, v, causal=causal, scale=scale,
                                            **cfg)
            info = dict(info)
            info.update(config=cfg, rungs_tried=tried)
            out = out.to(device=out_device if not need_host else out.device,
                         dtype=out_dtype)
            return (out, info) if return_info else out
        except Exception as exc:                                # noqa: BLE001
            if not _is_oom(exc):
                raise
            tried.append(cfg)
            if verbose:
                warnings.warn(f"{cfg} OOMed; escalating", RuntimeWarning, stacklevel=2)
            _drain()
    raise torch.cuda.OutOfMemoryError(
        f"exhausted the escalation ladder ({len(tried)} rungs). The device cannot "
        f"hold even one subproblem at the deepest setting; reduce N, H or D, or "
        f"use a larger GPU.")
