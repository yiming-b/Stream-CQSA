"""
Stream-CQSA in 20 lines: exact attention that returns a result where the
monolithic kernel would raise OutOfMemoryError.

    python examples/quickstart.py

`stream_cqsa_auto` is the "just run it" entry point. It walks an escalation
ladder cheapest-first -- deeper decomposition, then host-resident inputs, then a
host-resident accumulator -- and returns the first rung that completes. The
result is the same attention function throughout: this is not an approximation.
"""
import torch
from stream_cqsa import stream_cqsa_auto

B, H, N, D = 1, 8, 8192, 64
dev = "cuda"
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=torch.float16) for _ in range(3))

# --- the whole API -------------------------------------------------------
out, info = stream_cqsa_auto(q, k, v, causal=True, return_info=True)

# --- check it against the stock kernel at an N where that still fits ------
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
err = (out.float() - ref.float()).norm() / ref.float().norm()

print(f"N={N}  out {tuple(out.shape)} {out.dtype}")
print(f"rung chosen        : {info.get('config')}")
print(f"rel. error vs SDPA : {err:.3e}   (fp16 rounding is ~2.7e-04)")
print(f"global lse         : {tuple(info['lse'].shape)}  <- pass to stream_cqsa_backward")
