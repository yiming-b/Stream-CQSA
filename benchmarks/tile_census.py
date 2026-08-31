"""
Re-measure the CQS tile census, reporting both denominators explicitly.

docs/METHODOLOGY.md and docs/REPORT.md each publish a mixed-tile fraction for the
same configuration and disagree by roughly ten per cent (2.3% against 2.08% at
N=65536, itr=2). Neither states what it divided by, and the raw data is gone, so
this recomputes the census from the group bits and reports the fraction against
BOTH denominators: all tiles of the local score matrix, and only those tiles that
causal masking retains.

The census is a property of the index structure alone. No attention is executed
and no GPU is required, so the result is exact rather than measured.

A tile is 128x128. Its verdict follows from the block summaries:
  clear         OR(rows)  & OR(cols)  == 0   nothing in the tile is masked
  fully masked  AND(rows) & AND(cols) != 0   everything in the tile is masked
  mixed         otherwise
Because tokens are gathered in ascending order, tile (r,c) survives causal
masking exactly when the largest token id in row block r is at least the smallest
in column block c.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("CQSA_PKG", "packages/stream-cqsa"))
from stream_cqsa.reference import group_bits_for_path   # noqa: E402

TILE = 128
CONFIGS = [(65_536, 1), (65_536, 2), (262_144, 1), (262_144, 2),
           (1_048_576, 1), (1_048_576, 2)]
INTEREST = (0, 1, 3)
ROWS_PER_CHUNK = 512          # bounds peak memory on the largest grid


def block_reduce(bits, ids, tile):
    """Per-tile-block OR and AND of the group bits, plus the id range."""
    n = len(bits)
    nb = math.ceil(n / tile)
    OR = np.zeros(nb, dtype=np.int64)
    AND = np.full(nb, -1, dtype=np.int64)
    idmax = np.zeros(nb, dtype=np.int64)
    idmin = np.zeros(nb, dtype=np.int64)
    for b in range(nb):
        seg = bits[b * tile:(b + 1) * tile]
        seg_id = ids[b * tile:(b + 1) * tile]
        OR[b] = np.bitwise_or.reduce(seg)
        AND[b] = np.bitwise_and.reduce(seg)
        idmax[b] = seg_id.max()
        idmin[b] = seg_id.min()
    return OR, AND, idmax, idmin


def census(N, itr, path):
    ids, bits = group_bits_for_path(N, path, sorted_gather=True)
    ids = np.asarray(ids, dtype=np.int64)
    bits = np.asarray(bits, dtype=np.int64)
    L = len(bits)
    OR, AND, idmax, idmin = block_reduce(bits, ids, TILE)
    nb = len(OR)

    tot = {"clear": 0, "masked": 0, "mixed": 0}
    caus = {"clear": 0, "masked": 0, "mixed": 0}
    for lo in range(0, nb, ROWS_PER_CHUNK):
        hi = min(lo + ROWS_PER_CHUNK, nb)
        orx = OR[lo:hi, None] & OR[None, :]
        andx = AND[lo:hi, None] & AND[None, :]
        clear = orx == 0
        masked = (~clear) & (andx != 0)
        mixed = (~clear) & (~masked)
        keep = idmax[lo:hi, None] >= idmin[None, :]      # survives causal masking
        for name, m in (("clear", clear), ("masked", masked), ("mixed", mixed)):
            tot[name] += int(m.sum())
            caus[name] += int((m & keep).sum())
    return L, nb, tot, caus


def pct(part, whole):
    return 100.0 * part / whole if whole else float("nan")


if __name__ == "__main__":
    import itertools

    print(f"tile = {TILE}x{TILE}   c=7   interest set {INTEREST}")
    print("Every subproblem at a given depth is profiled, because the verdict "
          "split depends on which\none is chosen. Percentages are of "
          "causal-retained tiles unless the column says otherwise.\n")
    hdr = (f"{'N':>9} {'itr':>4} {'n_b':>6} {'paths':>6} | "
           f"{'masked mean':>12}{'range':>16} | {'clear mean':>11} | "
           f"{'mixed':>8} {'range':>13} | {'mixed, all tiles':>17}")
    print(hdr); print("-" * len(hdr))
    for N, itr in CONFIGS:
        rows = []
        for path in itertools.product(INTEREST, repeat=itr):
            L, nb, tot, caus = census(N, itr, path)
            T, C = sum(tot.values()), sum(caus.values())
            rows.append((pct(caus["masked"], C), pct(caus["clear"], C),
                         pct(caus["mixed"], C), pct(tot["mixed"], T), nb))
        m = [r[0] for r in rows]; c = [r[1] for r in rows]
        x = [r[2] for r in rows]; xa = [r[3] for r in rows]
        nb = rows[0][4]
        print(f"{N:>9} {itr:>4} {nb:>6} {len(rows):>6} | "
              f"{sum(m)/len(m):>11.2f}%{f'{min(m):.2f}-{max(m):.2f}':>16} | "
              f"{sum(c)/len(c):>10.2f}% | "
              f"{sum(x)/len(x):>7.2f}% {f'{min(x):.2f}-{max(x):.2f}':>13} | "
              f"{sum(xa)/len(xa):>16.2f}%", flush=True)

    print("\nMixed fraction against the l/n_b bound (l = 3 chunks):")
    for N, itr in CONFIGS:
        _, nb, _, caus = census(N, itr, (0,) * itr)
        C = sum(caus.values())
        print(f"  N={N:>9} itr={itr}  n_b={nb:>5}  "
              f"measured {pct(caus['mixed'], C):5.2f}%   bound {300.0/nb:5.2f}%")
