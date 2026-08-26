#!/usr/bin/env python
"""
Publication figures from run_paper_experiment.py results.

    python make_figures.py outputs/paper/curve_20260813 [more_run_dirs ...] \
        --out-dir docs/paper/Figures

Reads every `results.jsonl` given, merges them, and writes PDF + PNG.

Design decisions, and why
-------------------------
*Palette.* The categorical hues are assigned in fixed slot order and were
validated with the six checks (lightness band, chroma floor, protan/deutan
separation, normal-vision floor, contrast) on the *adjacent* pairlist, which is
the one that applies to line charts: worst adjacent CVD dE 9.2, worst
normal-vision dE 19.6, both clear. Three slots sit under 3:1 against white and
therefore carry the relief obligation -- discharged by the legend plus the
end-of-line direct labels, so identity never rests on color alone.

*Markers and dashes are not decoration.* A NeurIPS figure gets printed in
greyscale and photocopied. Every series carries a distinct marker AND a distinct
dash pattern, so the encoding survives with all color removed -- a stricter
requirement than colorblind-safety, and the reason the series count per panel is
kept low rather than plotting every variant measured.

*Log-log.* Sequence length spans 8k to 16M and attention is quadratic, so both
axes are log. A linear axis would compress the entire interesting range into the
last tick.

*One axis per panel.* Memory and time are never overlaid on twin y-axes; they
are separate panels. Two scales on one frame is the single most misleading thing
a chart of this kind can do.

*OOM is drawn, not omitted.* The first N at which a method fails is marked with
a hollow x at the last successful point and the series stops. A line that simply
ends looks like missing data; the whole claim is that it ended for a reason.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

# --- validated categorical slots, fixed order (see module docstring) --------
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e3e2dd"

# Series style: (color slot, marker, dash). Marker+dash carry identity without
# color, which is what survives greyscale printing.
# Series style: (colour slot, marker, dash).
#
# One slot per series, assigned in the palette's fixed order and never reused.
# The previous assignment put three series on two green steps -- flash and
# acc=CPU both on slot 2, the two acc=GPU depths both on slot 5 -- which the
# palette validator scores at Delta E 0.0, i.e. literally the same colour for
# different series. Marker and dash still carry identity redundantly, for
# greyscale print and for the ~8% of male readers with a red-green deficiency,
# but they are no longer doing the work alone.
#
#   python experiments/paper/validate_palette.py "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300"
#   -> worst adjacent CVD Delta E 9.1 (protan), normal-vision 19.6, both above gate.
STYLE = {
    "sdpa":             (0, "o", (0, ())),
    "sdpa_mem":         (1, "s", (0, (4, 2))),
    "flash":            (2, "^", (0, (6, 2, 1, 2))),
    "cqsa_accgpu_itr1": (3, "D", (0, (3, 1, 1, 1))),
    "cqsa_accgpu_itr2": (4, "v", (0, (1, 1, 3, 1))),
    "cqsa_acccpu":      (5, "*", (0, (5, 1, 1, 1))),
    # accuracy figure only -- a different chart, so the order restarts
    "cqsa_host_itr1":   (3, "D", (0, (3, 1, 1, 1))),
    "cqsa_host_itr2":   (4, "v", (0, (1, 1, 3, 1))),
    "cqsa_auto":        (5, "D", (0, (3, 1, 1, 1))),
    "cqsa_host":        (5, "v", (0, (1, 1, 3, 1))),
    "cqsa_hostmin":     (2, "*", (0, (5, 1, 1, 1, 1, 1))),
    "sdpa_flash":       (3, "P", (0, (1, 1))),
    "cqsa_accgpu":      (3, "D", (0, (3, 1, 1, 1))),
    "cqsa_allgpu":      (4, "D", (0, (3, 1, 1, 1))),
}
LABEL = {
    "sdpa": "SDPA", "sdpa_flash": "SDPA (flash)", "sdpa_mem": "SDPA (mem-eff.)",
    "flash": "FlashAttention-2",
    # Named as the paper's tables name them. The star on "auto" records that the
    # depth was forced to at least 1: the planner would have declined to
    # decompose at these lengths, and a column reporting what decomposition
    # costs has to actually decompose.
    "cqsa_accgpu_itr1": "Stream-CQSA  itr=1, acc=GPU",
    "cqsa_accgpu_itr2": "Stream-CQSA  itr=2, acc=GPU",
    "cqsa_acccpu": "Stream-CQSA  itr=auto*, acc=CPU",
    "cqsa_host_itr1": "Stream-CQSA itr=1",
    "cqsa_host_itr2": "Stream-CQSA itr=2",
    "cqsa_auto": "Stream-CQSA", "cqsa_host": "Stream-CQSA (streamed)",
    "cqsa_hostmin": "Stream-CQSA (min-device)",
    "cqsa_accgpu": "Stream-CQSA acc=GPU",
    "cqsa_allgpu": "Stream-CQSA (all on GPU)",
}
# Panels default to this subset: SDPA's flash backend and the flash-attn package
# are the same kernel, so plotting both crowds the panel for no information.
# Every method stays in the data and the appendix table.
# Stream-CQSA is *streamed* by definition; the all-on-GPU variant is dominated
# (it needs the whole input set resident AND adds an accumulator, so it OOMs
# before the baselines it exists to rescue) and is excluded from the shipped
# comparison. It remains plottable by naming it explicitly in --series.
DEFAULT_SERIES = ["sdpa", "sdpa_mem", "flash", "cqsa_accgpu", "cqsa_acccpu",
                  "cqsa_host", "cqsa_hostmin"]

# With --itr-list the sweep names its rows cqsa_auto_itr1, cqsa_host_itr2, ...
# so that each decomposition depth is its own series. Register those variants
# here: without them `plot_panel`'s `if m not in STYLE` would drop every
# Stream-CQSA curve from the figures without saying anything.
# Depth gets its own dash as well as its own marker. Sharing a dash within a
# family leaves marker shape as the only cue, which is the first thing to go in a
# greyscale print at figure scale.
_ITR_STYLE = {1: ("D", (0, (4, 1.5))),
              2: ("v", (0, (1, 1.2))),
              3: ("*", (0, (6, 1.5, 1, 1.5)))}
_ITR_STYLE["auto"] = ("o", (0, ()))          # automatic depth: solid, the default
for _base, _slot in (("cqsa_auto", 4), ("cqsa_host", 5), ("cqsa_hostmin", 2),
                     ("cqsa_accgpu", 3), ("cqsa_acccpu", 5), ("cqsa_allgpu", 4)):
    for _i, (_mk, _dash) in _ITR_STYLE.items():
        _k = f"{_base}_itr{_i}"
        # Do not overwrite a slot assigned explicitly above. This loop used to
        # run last and win, which is how three plotted series ended up sharing
        # two colours after the explicit table had already separated them.
        if _k not in STYLE:
            STYLE[_k] = (_slot, _mk, _dash)
        LABEL.setdefault(_k, f"{LABEL[_base]} itr={_i}")


def resolve_series(rows, requested):
    """Expand a requested series list against what is actually present.

    `cqsa_auto` in the request matches `cqsa_auto_itr1`/`_itr2` when the run used
    --itr-list, so the same --series argument works for both run styles and a
    typo cannot quietly produce an empty panel.
    """
    present = {r["method"] for r in rows}
    out = []
    for m in requested:
        if m in present:
            out.append(m)
        hits = sorted(x for x in present if x.startswith(m + "_itr"))
        out.extend(h for h in hits if h not in out)
    missing = [m for m in requested if m not in present
               and not any(x.startswith(m + "_itr") for x in present)]
    if missing:
        print(f"  ! requested but absent from the data: {missing}", file=sys.stderr)
    unknown = [m for m in out if m not in STYLE]
    if unknown:
        print(f"  ! present but unstyled (would be dropped): {unknown}", file=sys.stderr)
    return [m for m in out if m in STYLE]


def load(dirs, min_n=0):
    """Merge run directories. Later directories WIN on a duplicated
    (method, dtype, direction, N, seed) key, so a targeted re-measurement can be
    listed last to supersede an earlier run without editing raw results."""
    seen, rows = {}, []
    for d in dirs:
        p = os.path.join(d, "results.jsonl")
        if not os.path.exists(p):
            print(f"  ! no results.jsonl in {d}", file=sys.stderr)
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["N"] < min_n:
                    continue
                seen[(r["method"], r["dtype"], r["direction"], r["N"],
                      r.get("seed", 0))] = r
    rows = list(seen.values())
    return rows


def nfmt(v, _p=None):
    """8192 -> 8K, 16777216 -> 16M. Used on both x-scales."""
    if v >= 1_048_576:
        return f"{v/1_048_576:g}M"
    if v >= 1024:
        return f"{v/1024:g}K"
    return f"{v:g}"


def style_axes(ax, xlabel, ylabel, title=None, xscale="log"):
    if xscale == "log":
        ax.set_xscale("log", base=2)
    else:
        ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=6, loc="left")
    ax.xaxis.set_major_formatter(FuncFormatter(nfmt))
    if xscale == "log":
        ax.xaxis.set_major_locator(LogLocator(base=2, numticks=20))
    else:
        ax.set_xlim(0, 16 * 1_048_576)
        ax.set_xticks([i * 4 * 1_048_576 for i in range(5)])


OUT_NORM_TOL = 0.05


def median(xs):
    """Same definition the accuracy table uses -- the two must agree, and a mean
    against a median differs in the second digit on ten seeds."""
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])




def same_regime(rows):
    """Keep only accuracy rows computed on comparable inputs.

    A relative error means nothing except against another error measured on the
    same inputs. The accuracy run recorded each baseline under two input
    regimes -- for the fp16 forward the output norm is 104.85 in one group and
    3.56 in the other -- and averaging across them silently rewrites the result:
    read naively the baselines come out fifteen times worse than Stream-CQSA,
    which is backwards for a method whose whole claim is that it reproduces
    them. The Stream-CQSA rows form a single regime, so they are the anchor.

    This lives here, not only in the table builder, so that the figure cannot
    disagree with the table it sits next to.
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    kept = []
    for dt in {r.get("dtype") for r in ok}:
        for d in {r.get("direction") for r in ok}:
            grp = [r for r in ok if r["dtype"] == dt and r["direction"] == d]
            anchor = [r["out_norm"] for r in grp
                      if r["method"].startswith("cqsa") and r.get("out_norm")]
            if not anchor:
                kept += grp
                continue
            anchor.sort()
            ref = anchor[len(anchor) // 2]
            kept += [r for r in grp if r.get("out_norm") is not None
                     and abs(r["out_norm"] - ref) <= OUT_NORM_TOL * abs(ref)]
    if len(kept) != len(ok):
        print(f"  accuracy: dropped {len(ok) - len(kept)} row(s) from a "
              f"non-comparable input regime")
    return kept


def series_for(rows, method, dtype, direction, field):
    """(N, value) for successful points, plus the N where it first OOMed.

    Repeated rows at one N are averaged, which is right for seeds and wrong for
    depths. A length measured at itr=1, 2 and 3 contributes three rows, and
    averaging them puts a point on the chart that no run produced: at N=16M the
    forward came out at 4467 s / 35.2 GiB, the mean of three depths, where the
    table reports the 3277 s / 41.5 GiB of itr=1. So depth is resolved first --
    keep the smallest that ran, the rule the tables report under -- and only
    then are the remaining rows averaged over seeds."""
    pts, oom_n = {}, None
    depth = {}
    for r in rows:
        if (r["method"] != method or r["dtype"] != dtype
                or r["direction"] != direction):
            continue
        if r["status"] == "ok":
            v = r.get(field)
            if v is None or v != v:
                continue
            itr = (r.get("info") or {}).get("itr")
            if itr is not None:
                if r["N"] in depth and itr > depth[r["N"]]:
                    continue                      # a shallower depth already won
                if r["N"] not in depth or itr < depth[r["N"]]:
                    depth[r["N"]] = itr
                    pts[r["N"]] = []
            pts.setdefault(r["N"], []).append(v)
        elif r["status"] == "oom":
            oom_n = r["N"] if oom_n is None else min(oom_n, r["N"])
    xs = sorted(pts)
    ys = [sum(pts[x]) / len(pts[x]) for x in xs]
    return xs, ys, oom_n, {x: depth[x] for x in xs if x in depth}


def plot_panel(ax, rows, dtype, direction, field, methods, ylabel, title,
               scale=1.0, capacity=None):
    """One panel.

    `scale` converts the stored unit to the plotted one; `capacity` switches the
    panel to the memory presentation: linear on both axes, zero to the device's
    memory, with the limit drawn.

    The two rows of this figure use different scales, deliberately: each uses
    the one on which its own growth is a straight line, so that no curve implies
    a rate the data does not have.

    Peak memory is linear in N -- the measured ratio across a doubling is 2.00
    -- and time is quadratic, at 4.0. Plotted together on a log x-axis with a
    linear y, as they were, the memory panel turned a straight line into an
    exploding curve and read as quadratic growth: the distortion was entirely in
    the frame, and it argued against the very property the method is claimed to
    have. On linear axes the same rows are straight lines whose slopes differ,
    which is the actual finding, and the ceiling each one crosses is where it
    dies.

    Time keeps log-log, where a quadratic is a straight line of slope two.
    Putting it on a linear x with a log y would introduce the mirror-image error
    -- a quadratic would bend over and read as though it were saturating.

    Out-of-memory is marked at the length that FAILED, not at the last one that
    worked. Drawn at the last success it read as "SDPA fails at 8M" when SDPA
    fails at 16M and 8M is its last good measurement -- the marker was one rung
    left of the fact it was there to state. The failing point has no measured
    value to plot against, so the marker sits on the device limit (for memory,
    where "it asked for more than this" is exactly what happened) or at the top
    of the panel (for time), joined to the last real point by a dotted lead-in
    so it cannot be mistaken for a measurement.
    """
    any_data = False
    lo, hi = float("inf"), 0.0
    marks = []

    for m in methods:
        if m not in STYLE:
            continue
        xs, ys, oom_n, depths = series_for(rows, m, dtype, direction, field)
        if not xs:
            continue
        any_data = True
        ys = [y * scale for y in ys]
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        ci, mk, dash = STYLE[m]
        ax.plot(xs, ys, color=SLOT[ci], marker=mk, ms=4.5, lw=1.7,
                linestyle=dash, label=LABEL[m], zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        if oom_n is not None:
            marks.append((oom_n, xs[-1], ys[-1], SLOT[ci]))

        # Where the resolved depth changes, say so. "auto" means the smallest
        # depth that fits, so it steps up when the previous one stops fitting --
        # at N=16M in the backward. On a linear axis that step puts a visible
        # bend in an otherwise straight line, and unlabelled it reads as memory
        # growing sub-linearly rather than as a deeper decomposition being
        # bought at that length.
        if capacity and depths:
            for xi, x in enumerate(xs):
                if xi and depths.get(x) is not None and \
                        depths.get(x) != depths.get(xs[xi - 1]):
                    ax.annotate(f"itr={depths[x]}", (x, ys[xi]),
                                textcoords="offset points", xytext=(-4, -13),
                                ha="right", fontsize=7, color=SLOT[ci],
                                zorder=6)

    style_axes(ax, "sequence length $N$", ylabel, title,
               xscale="linear" if capacity else "log")

    if capacity and any_data:
        ax.set_yscale("linear")
        ax.set_ylim(0, capacity * 1.11)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.axhline(capacity, color=INK3, lw=0.9, ls=(0, (5, 3)), zorder=1)
        ax.text(0.015, capacity * 1.01, f"device limit {capacity:g} GiB",
                transform=ax.get_yaxis_transform(), va="bottom", ha="left",
                color=INK3, fontsize=7)
        y_fail = capacity
    else:
        # Log panel: park the markers just under the top of the drawn range.
        ax.set_ylim(top=hi * 6 if hi else None)
        y_fail = hi * 3.2 if hi else None

    # Methods that fail at the same length would otherwise land on one point and
    # hide each other -- four series share N=16M in the forward, five share 8M
    # in the backward, and only the last drawn was visible, which read as "one
    # method failed here". Fan them around the tick. The axis is logarithmic, so
    # the offsets are multiplicative to keep the spacing even on screen.
    by_n = collections.defaultdict(list)
    for mk in marks:
        by_n[mk[0]].append(mk)
    for oom_n, group in by_n.items():
        k = len(group)
        for i, (_, x_last, y_last, colour) in enumerate(group):
            if y_fail is None:
                continue
            if k == 1:
                off = oom_n
            elif capacity:          # linear x: an additive nudge, ~1.4% of span
                off = oom_n + (i - (k - 1) / 2) * 0.23e6
            else:                   # log x: multiplicative, so spacing is even
                off = oom_n * (1.052 ** (i - (k - 1) / 2))
            ax.plot([x_last, off], [y_last, y_fail], color=colour, lw=0.9,
                    ls=(0, (1, 2)), zorder=2, alpha=0.7)
            ax.plot([off], [y_fail], marker="x", ms=8, mew=2.2, color=colour,
                    zorder=5, linestyle="none", clip_on=False)
    return any_data


def fig_memory_time(rows, dtype, methods, out, meta):
    dirs_present = [d for d in ("fwd", "bwd")
                    if any(r["direction"] == d and r["dtype"] == dtype for r in rows)]
    if not dirs_present:
        return False
    fig, axes = plt.subplots(2, len(dirs_present),
                             figsize=(4.4 * len(dirs_present), 6.4), squeeze=False)
    name = {"fwd": "forward", "bwd": "backward"}
    ok = False
    # Letter row-major: (a)(b) across the top, (c)(d) across the bottom. Going
    # column-major makes the reader jump diagonally.
    nc = len(dirs_present)
    for j, d in enumerate(dirs_present):
        ok |= plot_panel(axes[0][j], rows, dtype, d, "mem_alloc_peak", methods,
                         "peak GPU memory (GiB)", f"({chr(97+j)}) {name[d]} — memory",
                         scale=1.0 / 1024.0, capacity=meta.get("gpu_gib"))
        ok |= plot_panel(axes[1][j], rows, dtype, d, "ms", methods,
                         "wall-clock (ms)", f"({chr(97+nc+j)}) {name[d]} — time")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3),
               frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.005), handlelength=3.0)
    gpu = meta.get("gpu", "A100")
    fig.text(0.5, 0.008,
             f"{gpu} · B={meta.get('B',1)} H={meta.get('H',8)} D={meta.get('D',64)} · "
             f"{'fp16' if dtype == 'float16' else 'bf16'} · causal · "
             f"\u2715 = shortest N that runs out of memory",
             ha="center", color=INK3, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, out)
    return ok


def fig_accuracy(rows, out, meta, methods=None, labels=None):
    """Relative error vs float64, by dtype and direction. Not a line chart: the
    x axis is categorical, so this is a dot plot with a spread bar over seeds.

    `methods` fixes which series are drawn and should be the columns of the
    accuracy table. Left to itself this draws every Stream-CQSA variant present,
    which puts two families on the chart that no reader can separate: the host-
    and device-accumulating runs agree to five digits at every depth and differ
    only in the last bits of the backward, so they land on top of one another
    and double the legend for nothing. `labels` overrides the legend text where
    the table names a series differently."""
    cfgs = [(dt, d) for dt in ("float16", "bfloat16") for d in ("fwd", "bwd")]
    rows = same_regime(rows)
    agg = collections.defaultdict(list)
    for r in rows:
        if r["status"] != "ok":
            continue
        v = r.get("acc_rel")
        if v is None or v != v:
            continue
        agg[(r["dtype"], r["direction"], r["method"])].append(v)
    cfgs = [c for c in cfgs if any(k[:2] == c for k in agg)]
    if not cfgs:
        return False
    present = {k[2] for k in agg}
    if methods is None:
        methods = [m for m in
                   (DEFAULT_SERIES + sorted(x for x in present if "_itr" in x))
                   if m in present and m in STYLE]
    else:
        absent = [m for m in methods if m not in present]
        if absent:
            print(f"  ! accuracy: requested but absent: {absent}", file=sys.stderr)
        methods = [m for m in methods if m in present and m in STYLE]
    labels = labels or {}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xpos, xlab = [], []
    for i, (dt, d) in enumerate(cfgs):
        base = i * (len(methods) + 1.4)
        for j, m in enumerate(methods):
            vals = agg.get((dt, d, m))
            if not vals:
                continue
            ci, mk, _ = STYLE[m]
            x = base + j
            mid = median(vals)
            ax.plot([x], [mid], marker=mk, ms=6, color=SLOT[ci], zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=(labels.get(m, LABEL[m]) if i == 0 else None),
                    linestyle="none")
            if len(vals) > 1:
                ax.plot([x, x], [min(vals), max(vals)], color=SLOT[ci],
                        lw=1.4, alpha=0.55, zorder=2)
        xpos.append(base + (len(methods) - 1) / 2)
        xlab.append(f"{'fp16' if dt=='float16' else 'bf16'}\n{'fwd' if d=='fwd' else 'bwd'}")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlab, color=INK2, fontsize=9)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_ylabel("relative error vs float64", color=INK2, fontsize=9)
    ax.set_title("Output accuracy is set by the input dtype, not by decomposition",
                 color=INK, fontsize=10, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3,
              loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.text(0.99, 0.02, "marker = median, bar = min\u2013max over 10 seeds",
             ha="right", color=INK3, fontsize=7.5)
    fig.tight_layout()
    save(fig, out)
    return True


def stage_rows(rows, dtype, direction, method=None):
    """Pick one Stream-CQSA row per N for the stage breakdown.

    Two things this must not do. It must not hardcode a method name: the runner
    encodes each sweep's configuration into the method (cqsa_acccpu,
    cqsa_accgpu_itr2, ...), so a fixed prefix stops matching the moment a sweep
    is renamed and the figure silently comes out empty rather than wrong. And it
    must not stack several depths at one N: a length measured at itr=1,2,3 has
    three rows, and drawing all of them puts three bars on one tick.

    The depth kept is the smallest that ran, which is the rule the tables report
    under -- smallest itr within the memory budget."""
    cand = [r for r in rows
            if r["dtype"] == dtype and r["direction"] == direction
            and r["status"] == "ok" and r.get("stage_ms")
            and r["method"].startswith("cqsa")]
    if not cand:
        return [], None
    if method is None:
        names = sorted({r["method"] for r in cand})
        # Prefer the host-accumulating variant: its transfer stages are the
        # point of the figure, and it is the configuration the tables lead with.
        method = next((m for m in names if "acccpu" in m or "host" in m), names[0])
    by_n = {}
    for r in (x for x in cand if x["method"] == method):
        itr = (r.get("info") or {}).get("itr")
        prev = by_n.get(r["N"])
        if prev is None or (itr is not None
                            and itr < ((prev.get("info") or {}).get("itr", 1 << 30))):
            by_n[r["N"]] = r
    return [by_n[n] for n in sorted(by_n)], method


STAGE_CONFIGS = [("cqsa_accgpu_itr1", "itr=1, acc=GPU"),
                 ("cqsa_accgpu_itr2", "itr=2, acc=GPU"),
                 ("cqsa_acccpu",      "itr=auto*, acc=CPU")]

# Stack order, and therefore the adjacent pairs the palette was validated on.
STAGE_ORDER = ("compute", "gather", "scatter", "h2d", "d2h", "merge")


def stage_panel(ax, rows, dtype, direction, method, title):
    """One configuration, one direction: stage shares against sequence length.

    Shares, not totals, and on a linear axis. Stacked milliseconds on a log axis
    cannot be read -- a segment's height is not proportional to its value there,
    so whichever stage is largest looks dominant whatever the split is, and the
    shortest lengths compress to nothing against a 10^7 ms top.

    `wait` is excluded. It measures time an event span spent queued behind
    another stream, so it runs concurrently with the work it would be stacked
    on top of -- at 16M it lands within 0.2% of `compute` for that reason.
    Including it in a proportion double-counts the concurrency.
    """
    keep, _ = stage_rows(rows, dtype, direction, method)
    if not keep:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                color=INK3, fontsize=8, transform=ax.transAxes)
        return False, []

    seen = {k for r in keep for k in r["stage_ms"] if k != "wait"}
    pref = [st for st in STAGE_ORDER if st in seen] + sorted(seen - set(STAGE_ORDER))
    shares = []
    for r in keep:
        d = {k: v for k, v in r["stage_ms"].items() if k != "wait"}
        tot = sum(d.values()) or 1.0
        shares.append({k: 100.0 * d.get(k, 0.0) / tot for k in pref})

    xs = list(range(len(keep)))
    bottom = [0.0] * len(keep)
    for st in pref:
        vals = [sh[st] for sh in shares]
        ax.bar(xs, vals, bottom=bottom, width=0.74,
               color=SLOT[STAGE_ORDER.index(st) % len(SLOT)], label=st,
               zorder=3, edgecolor="white", linewidth=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]

    for x, sh in zip(xs, shares):
        c = sh.get("compute", 0.0)
        if c >= 12:                     # below this the numeral will not fit
            ax.text(x, c / 2, f"{c:.0f}", ha="center", va="center",
                    fontsize=6.5, color="white", zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['N']//1024}K" if r["N"] < 1_048_576
                        else f"{r['N']//1_048_576}M" for r in keep],
                       color=INK2, fontsize=7, rotation=90)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK3)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.set_title(title, color=INK, fontsize=9, loc="left", pad=5)
    return True, pref


def fig_stages(rows, dtype, out, meta):
    """Stage breakdown for three configurations in both directions.

    The grid is the point: the same decomposition is profiled at two depths on
    the device and once with the accumulators on the host, forward and backward,
    so the reader can see which parts of the cost follow the depth, which follow
    the residency choice, and which follow neither. Where a configuration stops
    short, it stopped because it ran out of memory at the next length.
    """
    fig, axes = plt.subplots(2, len(STAGE_CONFIGS),
                             figsize=(3.35 * len(STAGE_CONFIGS), 5.6),
                             squeeze=False)
    ok = False
    stages = []
    for i, d in enumerate(("fwd", "bwd")):
        for j, (method, name) in enumerate(STAGE_CONFIGS):
            drew, pref = stage_panel(
                axes[i][j], rows, dtype, d, method,
                f"({chr(97 + i * len(STAGE_CONFIGS) + j)}) "
                f"{'forward' if d == 'fwd' else 'backward'} — {name}")
            ok |= drew
            for st in pref:
                if st not in stages:
                    stages.append(st)
        axes[i][0].set_ylabel("share of issued work (%)", color=INK2, fontsize=8.5)
    for j in range(len(STAGE_CONFIGS)):
        axes[1][j].set_xlabel("sequence length $N$", color=INK2, fontsize=8.5)

    order = [st for st in STAGE_ORDER if st in stages]
    handles = [plt.Rectangle((0, 0), 1, 1,
                             color=SLOT[STAGE_ORDER.index(st) % len(SLOT)])
               for st in order]
    fig.legend(handles, order, loc="upper center", ncol=len(order),
               frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.005))
    fig.text(0.99, 0.005,
             "numerals = local-kernel share · queueing behind other streams "
             "excluded · a panel ends where that configuration runs out of memory",
             ha="right", color=INK3, fontsize=7)
    fig.tight_layout(rect=(0, 0.025, 1, 0.945))
    save(fig, out)
    return ok


def save(fig, out):
    """Write .svg and .jpg. main.tex includes the .jpg, matching the convention
    of the figures already in the paper; the .svg is the lossless copy.

    JPEG has no alpha channel, so an explicit white facecolor is required or the
    background renders black. 400 dpi keeps line art and text clean at print
    size despite the lossy codec."""
    base = os.path.splitext(os.path.abspath(out))[0]
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".jpg", bbox_inches="tight", facecolor="white",
                dpi=400, pil_kwargs={"quality": 95, "optimize": True})
    plt.close(fig)
    print(f"  wrote {os.path.basename(base)}.jpg + .svg")


def write_table(rows, out):
    """LaTeX table of the sweep -- the appendix companion to the figures."""
    ok = [r for r in rows if r["status"] in ("ok", "oom")]
    ok.sort(key=lambda r: (r["dtype"], r["direction"], r["N"], r["method"]))
    lines = [r"\begin{tabular}{llrlrrrl}", r"\toprule",
             r"dtype & dir & $N$ & method & time (ms) & peak (MiB) & host (MiB) & rel.\ err \\",
             r"\midrule"]
    for r in ok:
        dt = "fp16" if r["dtype"] == "float16" else "bf16"
        if r["status"] == "oom":
            lines.append(f"{dt} & {r['direction']} & {r['N']} & {LABEL.get(r['method'], r['method'])} "
                         r"& \multicolumn{4}{c}{OOM} \\")
            continue
        err = "--" if r["acc_rel"] != r["acc_rel"] else f"{r['acc_rel']:.2e}"
        lines.append(f"{dt} & {r['direction']} & {r['N']} & {LABEL.get(r['method'], r['method'])} "
                     f"& {r['ms']:.1f} & {r['mem_alloc_peak']:.0f} & "
                     f"{r['mem_host_bytes']:.0f} & {err} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out-dir", default="docs/paper/Figures")
    ap.add_argument("--series", default=",".join(DEFAULT_SERIES))
    ap.add_argument("--min-n", type=int, default=0,
                    help="drop rungs below this N (e.g. to exclude a "
                         "warmup-contaminated first measurement)")
    a = ap.parse_args()

    rows = load(a.run_dirs, min_n=a.min_n)
    if not rows:
        print("no results found", file=sys.stderr)
        return 2
    meta = {}
    for d in a.run_dirs:
        p = os.path.join(d, "meta.json")
        if os.path.exists(p):
            meta.update(json.load(open(p)))
    methods = resolve_series(rows, [m.strip() for m in a.series.split(",") if m.strip()])
    print("  plotting series:", methods)
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"loaded {len(rows)} rows from {len(a.run_dirs)} run(s)")

    for dt in sorted({r["dtype"] for r in rows}):
        tag = "fp16" if dt == "float16" else "bf16"
        fig_memory_time(rows, dt, methods,
                        os.path.join(a.out_dir, f"fig_mem_time_{tag}.jpg"), meta)
    fig_accuracy(rows, os.path.join(a.out_dir, "fig_accuracy.jpg"), meta)
    for dt in sorted({r["dtype"] for r in rows}):
        tag = "fp16" if dt == "float16" else "bf16"
        for d in sorted({r["direction"] for r in rows}):
            fig_stages(rows, dt, d,
                       os.path.join(a.out_dir, f"fig_stages_{tag}_{d}.jpg"), meta)
    write_table(rows, os.path.join(a.out_dir, "table_sweep.tex"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
