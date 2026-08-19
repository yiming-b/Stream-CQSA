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
STYLE = {
    "sdpa":       (0, "o", (0, ())),
    "sdpa_flash": (3, "P", (0, (1, 1))),
    "sdpa_mem":   (1, "s", (0, (4, 2))),
    "flash":      (2, "^", (0, (6, 2, 1, 2))),
    "cqsa_auto":  (4, "D", (0, (3, 1, 1, 1))),
    "cqsa_host":  (5, "v", (0, (1, 1, 3, 1))),
    "cqsa_hostmin": (2, "*", (0, (5, 1, 1, 1, 1, 1))),
    "cqsa_accgpu":  (5, "v", (0, (1, 1, 3, 1))),
    "cqsa_acccpu":  (2, "*", (0, (5, 1, 1, 1))),
    "cqsa_allgpu":  (4, "D", (0, (3, 1, 1, 1))),
}
LABEL = {
    "sdpa": "SDPA", "sdpa_flash": "SDPA (flash)", "sdpa_mem": "SDPA (mem-eff.)",
    "flash": "FlashAttention-2", "cqsa_auto": "Stream-CQSA",
    "cqsa_host": "Stream-CQSA (streamed)",
    "cqsa_hostmin": "Stream-CQSA (min-device)",
    "cqsa_accgpu": "Stream-CQSA acc=GPU",
    "cqsa_acccpu": "Stream-CQSA acc=CPU",
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
                     ("cqsa_accgpu", 5), ("cqsa_acccpu", 2), ("cqsa_allgpu", 4)):
    for _i, (_mk, _dash) in _ITR_STYLE.items():
        _k = f"{_base}_itr{_i}"
        STYLE[_k] = (_slot, _mk, _dash)
        LABEL[_k] = f"{LABEL[_base]} itr={_i}"


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


def style_axes(ax, xlabel, ylabel, title=None):
    ax.set_xscale("log", base=2)
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
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, p: (f"{v/1_048_576:.0f}M" if v >= 1_048_576 else
                      f"{v/1024:.0f}K" if v >= 1024 else f"{v:.0f}")))
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=20))


def series_for(rows, method, dtype, direction, field):
    """(N, value) for successful points, plus the N where it first OOMed."""
    pts, oom_n = {}, None
    for r in rows:
        if (r["method"] != method or r["dtype"] != dtype
                or r["direction"] != direction):
            continue
        if r["status"] == "ok":
            v = r.get(field)
            if v is not None and v == v:
                pts.setdefault(r["N"], []).append(v)
        elif r["status"] == "oom":
            oom_n = r["N"] if oom_n is None else min(oom_n, r["N"])
    xs = sorted(pts)
    ys = [sum(pts[x]) / len(pts[x]) for x in xs]
    return xs, ys, oom_n


def plot_panel(ax, rows, dtype, direction, field, methods, ylabel, title):
    any_data = False
    for m in methods:
        if m not in STYLE:
            continue
        xs, ys, oom_n = series_for(rows, m, dtype, direction, field)
        if not xs:
            continue
        any_data = True
        ci, mk, dash = STYLE[m]
        ax.plot(xs, ys, color=SLOT[ci], marker=mk, ms=4.5, lw=1.7,
                linestyle=dash, label=LABEL[m], zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        if oom_n is not None:
            # Mark where it stopped and why. A line that just ends reads as
            # missing data; this is the claim.
            ax.plot([xs[-1]], [ys[-1]], marker="x", ms=9, mew=2.0,
                    color=SLOT[ci], zorder=4, linestyle="none")
    style_axes(ax, "sequence length $N$", ylabel, title)
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
                         "peak GPU memory (MiB)", f"({chr(97+j)}) {name[d]} — memory")
        ok |= plot_panel(axes[1][j], rows, dtype, d, "ms", methods,
                         "wall-clock (ms)", f"({chr(97+nc+j)}) {name[d]} — time")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3),
               frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.005), handlelength=3.0)
    gpu = meta.get("gpu", "A100")
    fig.text(0.5, 0.008,
             f"{gpu} · B={meta.get('B',1)} H={meta.get('H',8)} D={meta.get('D',64)} · "
             f"{'fp16' if dtype == 'float16' else 'bf16'} · causal · x = OOM",
             ha="center", color=INK3, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, out)
    return ok


def fig_accuracy(rows, out, meta):
    """Relative error vs float64, by dtype and direction. Not a line chart: the
    x axis is categorical, so this is a dot plot with a spread bar over seeds."""
    cfgs = [(dt, d) for dt in ("float16", "bfloat16") for d in ("fwd", "bwd")]
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
    methods = [m for m in
               (DEFAULT_SERIES + sorted(x for x in present if "_itr" in x))
               if m in present and m in STYLE]
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
            mean = sum(vals) / len(vals)
            ax.plot([x], [mean], marker=mk, ms=6, color=SLOT[ci], zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=LABEL[m] if i == 0 else None, linestyle="none")
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
    fig.text(0.99, 0.02, "bar = min–max over seeds", ha="right",
             color=INK3, fontsize=7.5)
    fig.tight_layout()
    save(fig, out)
    return True


def fig_stages(rows, dtype, direction, out, meta):
    """Where Stream-CQSA's time goes. Stacked bars: the parts sum to the whole
    only for a single-stream run, so this is labelled as a breakdown of stage
    totals, not of wall-clock."""
    keep = [r for r in rows if r["dtype"] == dtype and r["direction"] == direction
            and r["status"] == "ok" and r["stage_ms"] and r["method"].startswith("cqsa_host")]
    if not keep:
        return False
    keep.sort(key=lambda r: r["N"])
    order, seen = [], set()
    for r in keep:
        for k in r["stage_ms"]:
            if k not in seen:
                seen.add(k)
                order.append(k)
    pref = [s for s in ("compute", "gather", "scatter", "h2d", "d2h", "merge", "wait")
            if s in seen] + [s for s in order if s not in
                             ("compute", "gather", "scatter", "h2d", "d2h", "merge", "wait")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = list(range(len(keep)))
    bottom = [0.0] * len(keep)
    for i, st in enumerate(pref):
        vals = [r["stage_ms"].get(st, 0.0) for r in keep]
        ax.bar(xs, vals, bottom=bottom, width=0.62,
               color=SLOT[i % len(SLOT)], label=st, zorder=3,
               edgecolor="white", linewidth=1.2)   # 2px surface gap between segments
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['N']//1024}K" if r["N"] < 1_048_576
                        else f"{r['N']//1_048_576}M" for r in keep],
                       color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel("sequence length $N$", color=INK2, fontsize=9)
    ax.set_ylabel("stage total (ms)", color=INK2, fontsize=9)
    ax.set_title(f"Stream-CQSA (streamed) stage breakdown — {dtype}, "
                 f"{'forward' if direction=='fwd' else 'backward'}",
                 color=INK, fontsize=10, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=4,
              loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.text(0.99, 0.02,
             "stage totals over-count wall time when several streams are in flight",
             ha="right", color=INK3, fontsize=7)
    fig.tight_layout()
    save(fig, out)
    return True


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
