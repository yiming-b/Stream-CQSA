"""
Stream-CQSA in 20 lines: exact attention that returns a result where the
monolithic kernel would raise OutOfMemoryError.

    python examples/quickstart.py

`stream_cqsa_auto` is the "just run it" entry point. It starts from the safest
configuration -- inputs streamed from the host, fp32 accumulator host-resident,
depth chosen automatically -- and deepens the decomposition only if that still
runs out of memory. The result is the same attention function throughout: this
is not an approximation.

Note that it relocates q/k/v to host memory in place; that is what frees the
device allocation. Use `stream_cqsa_forward` when you want to keep control.
"""
import torch
from stream_cqsa import stream_cqsa_auto

B, H, N, D = 1, 8, 8192, 64
dev = "cuda"
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))

# Reference FIRST: stream_cqsa_auto relocates q/k/v to host memory in place,
# so anything that needs them on the device has to run before the call.
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"before: q is on {q.device}")

# --- the whole API -------------------------------------------------------
out, info = stream_cqsa_auto(q, k, v, causal=True, return_info=True)

print(f"after : q is on {q.device}   <- relocated, which is what frees the device")

err = (out.float() - ref.float().to(out.device)).norm() / ref.float().norm()
print(f"\nN={N}  out {tuple(out.shape)} {out.dtype} on {out.device}")
print(f"rung chosen        : {info.get('config')}")
print(f"depth chosen       : itr={info['itr']}  ({info['n_subproblems']} subproblems)")
print(f"rel. error vs SDPA : {err:.3e}   (fp16 rounding is ~2.7e-04)")
print(f"global lse         : {tuple(info['lse'].shape)}  <- pass to stream_cqsa_backward")
