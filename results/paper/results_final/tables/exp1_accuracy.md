*Relative error against a float64 reference, N = 8192, B=1 H=8 D=64,
causal. Median over 10 seeds. Forward compares the output; backward
compares dQ (dK and dV in the CSV).*

**Reading the forward row.** The baselines sit at the fp16 rounding
floor, near 2.7e-04, because they return the output in the input
dtype. Stream-CQSA sits below that floor because its accumulator is
fp32 and it returns fp32 -- the output is never rounded to fp16 at
all. That is a real property of the method and not a sharper
arithmetic: cast the fp32 output down and the two agree. The backward
row is the like-for-like comparison, both paths returning fp32, and
there Stream-CQSA matches the baselines to three digits, which is
what an exact decomposition should do.

### fp16

| direction | SDPA | SDPA (mem-eff) | FA-2 | CQSA itr=1 | CQSA itr=2 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **forward** | 2.69e-04 | 2.69e-04 | 2.69e-04 | 1.70e-04 | 1.68e-04 |
| **backward** | 3.07e-04 | 3.73e-04 | 3.07e-04 | 3.08e-04 | 3.08e-04 |

### bf16

| direction | SDPA | SDPA (mem-eff) | FA-2 | CQSA itr=1 | CQSA itr=2 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **forward** | 2.16e-03 | 2.15e-03 | 2.16e-03 | 1.37e-03 | 1.35e-03 |
| **backward** | 2.45e-03 | 2.99e-03 | 2.45e-03 | 2.45e-03 | 2.46e-03 |
