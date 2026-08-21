#!/usr/bin/env python
"""
Animated forward pass of Stream-CQSA, for the README and for social posts.

    python make_forward_gif.py        # writes stream_cqsa_forward.gif (+ .png stills)

The point the animation has to make, in one loop: seven subproblems, each seeing
only three of the seven chunks, together cover every one of the 49 chunk pairs
exactly once. Nothing is dropped and nothing is double counted, which is why the
recomposed output is exact rather than approximate.

Layout, left to right:
  * the sequence as c = 7 chunks, with the current quorum {i, i+1, i+3} raised
  * the local 3x3 chunk map for that subproblem, showing which of its nine chunk
    pairs it keeps and which the CQS mask removes because another subproblem owns
    them
  * the global 7x7 map, filling in as the traversal proceeds

Colour encodes subproblem index as a single-hue lightness ramp rather than seven
separate hues: the index is ordered, one hue stays legible under colour-vision
deficiency and in greyscale, and the current subproblem is picked out by a warm
outline instead of by a hue of its own.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTDIR = None
for cand in (
    "/home/yb2807/.conda/envs/CQS_ENV_311/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf",
):
    if os.path.isdir(cand):
        FONTDIR = cand
        break
if FONTDIR is None:                                   # fall back to matplotlib's copy
    import matplotlib
    FONTDIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

F = lambda name, size: ImageFont.truetype(os.path.join(FONTDIR, name), size)

C, ISET = 7, (0, 1, 3)
W, H = 1120, 620
SS = 2                                                 # supersample, then downscale

BG      = (255, 255, 255)
INK     = (24, 24, 22)
INK2    = (92, 92, 88)
INK3    = (150, 150, 144)
GRID    = (222, 224, 219)
EMPTY   = (247, 248, 245)
ACCENT  = (235, 104, 52)                               # the current subproblem
ACC_SOF = (253, 231, 220)
MASKOUT = (226, 228, 223)

# single-hue ramp, light -> dark, indexed by owning subproblem
RAMP = [(206, 228, 244), (170, 206, 234), (132, 183, 223), (95, 158, 210),
        (62, 132, 194), (38, 105, 166), (24, 78, 130)]


def quorum(i):
    return [(i + o) % C for o in ISET]


def owned(i):
    q = quorum(i)
    return [(i, i)] + [(a, b) for a in q for b in q if a != b]


OWNER = {}
for _i in range(C):
    for _p in owned(_i):
        OWNER[_p] = _i


def rr(d, box, fill, outline=None, width=2, r=6):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def centre(d, xy, text, font, fill):
    x, y = xy
    l, t, rgt, b = d.textbbox((0, 0), text, font=font)
    d.text((x - (rgt - l) / 2 - l, y - (b - t) / 2 - t), text, font=font, fill=fill)


def frame(step):
    """step = -1 (empty) .. C-1 (subproblem), or C (complete)."""
    im = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(im)
    s = SS
    f_title = F("DejaVuSans-Bold.ttf", 27 * s)
    f_sub   = F("DejaVuSans.ttf", 16 * s)
    f_head  = F("DejaVuSans-Bold.ttf", 15 * s)
    f_lab   = F("DejaVuSans.ttf", 14 * s)
    f_small = F("DejaVuSans.ttf", 12.5 * s)
    f_cell  = F("DejaVuSans-Bold.ttf", 14 * s)
    f_big   = F("DejaVuSans-Bold.ttf", 19 * s)

    active = 0 <= step < C
    done = step if step >= 0 else 0                    # subproblems completed so far

    # ------------------------------------------------------------- header --
    d.text((44 * s, 30 * s), "Stream-CQSA forward pass", font=f_title, fill=INK)
    d.text((44 * s, 66 * s),
           "c = 7 chunks,  interest set (0, 1, 3)  —  each subproblem sees 3 of 7 chunks",
           font=f_sub, fill=INK2)

    # -------------------------------------------------- 1. the chunk strip --
    x0, y0, cw, ch, gap = 44 * s, 152 * s, 46 * s, 46 * s, 7 * s
    d.text((x0, 112 * s), "1.  gather", font=f_head, fill=INK)
    q = quorum(step) if active else []
    for k in range(C):
        bx = x0 + k * (cw + gap)
        inq, own = k in q, (active and k == step)
        lift = 6 * s if inq else 0
        fill = ACC_SOF if inq else EMPTY
        outl = ACCENT if inq else GRID
        rr(d, [bx, y0 - lift, bx + cw, y0 + ch - lift], fill, outl, (3 if own else 2) * s, 5 * s)
        centre(d, (bx + cw / 2, y0 + ch / 2 - lift), f"C{k}", f_cell,
               ACCENT if inq else INK3)
        if own:
            centre(d, (bx + cw / 2, y0 + ch + 14 * s - lift), "owner", f_small, ACCENT)
    d.text((x0, y0 + ch + 30 * s),
           (f"subsequence {step}  =  concat(C{q[0]}, C{q[1]}, C{q[2]})" if active
            else ("all 7 subproblems complete" if step >= C else "7 chunks, 49 chunk pairs")),
           font=f_lab, fill=INK2 if active else INK3)

    # ------------------------------------------ 2. the local 3x3 chunk map --
    lx, ly, lc = 66 * s, 300 * s, 44 * s
    d.text((44 * s, 262 * s), "2.  local attention  +  CQS mask", font=f_head, fill=INK)
    if active:
        for r_, a in enumerate(q):
            for c_, b in enumerate(q):
                bx, by = lx + c_ * lc, ly + r_ * lc
                keep = (a != b) or (a == step)
                fill = RAMP[step] if keep else MASKOUT
                rr(d, [bx, by, bx + lc - 3 * s, by + lc - 3 * s], fill,
                   ACCENT if keep else GRID, 2 * s, 4 * s)
                if not keep:
                    m = 12 * s
                    d.line([bx + m, by + m, bx + lc - 3 * s - m, by + lc - 3 * s - m],
                           fill=INK3, width=2 * s)
                    d.line([bx + lc - 3 * s - m, by + m, bx + m, by + lc - 3 * s - m],
                           fill=INK3, width=2 * s)
            centre(d, (lx - 14 * s, ly + r_ * lc + lc / 2 - 2 * s), f"C{a}", f_small, INK3)
        for c_, b in enumerate(q):
            centre(d, (lx + c_ * lc + lc / 2 - 2 * s, ly - 11 * s), f"C{b}", f_small, INK3)
        d.text((44 * s, ly + 3 * lc + 18 * s),
               "keeps 7 of 9:  the 6 off-diagonal pairs, plus its own diagonal.",
               font=f_small, fill=INK2)
        d.text((44 * s, ly + 3 * lc + 38 * s),
               f"C{q[1]}-C{q[1]} and C{q[2]}-C{q[2]} are masked — subproblems "
               f"{q[1]} and {q[2]} own them.",
               font=f_small, fill=INK2)
    else:
        d.text((44 * s, ly + 8 * s),
               "Each subproblem keeps the 6 off-diagonal" if step < 0 else
               "Every chunk pair was computed exactly once.",
               font=f_small, fill=INK3)
        d.text((44 * s, ly + 26 * s),
               "chunk pairs of its quorum, plus its own" if step < 0 else
               "No pair was dropped; none was computed twice.",
               font=f_small, fill=INK3)
        if step < 0:
            d.text((44 * s, ly + 44 * s), "diagonal.  7 pairs each.", font=f_small, fill=INK3)

    # ------------------------------------------------- 3. the global 7x7 ----
    gx, gy, gc = 622 * s, 152 * s, 56 * s
    d.text((600 * s, 112 * s), "3.  scatter into the attention map", font=f_head, fill=INK)
    for a in range(C):
        centre(d, (gx - 16 * s, gy + a * gc + gc / 2 - 2 * s), f"C{a}", f_small, INK3)
        centre(d, (gx + a * gc + gc / 2 - 2 * s, gy - 11 * s), f"C{a}", f_small, INK3)
    for a in range(C):
        for b in range(C):
            o = OWNER[(a, b)]
            bx, by = gx + b * gc, gy + a * gc
            box = [bx, by, bx + gc - 3 * s, by + gc - 3 * s]
            if step >= C or o < done:                    # already covered
                rr(d, box, RAMP[o], None, 0, 4 * s)
            elif active and o == step:                   # being covered now
                rr(d, box, RAMP[o], ACCENT, 3 * s, 4 * s)
            else:                                        # not yet
                rr(d, box, EMPTY, GRID, 2 * s, 4 * s)
            if step >= C or o <= (step if active else -1):
                centre(d, (bx + (gc - 3 * s) / 2, by + (gc - 3 * s) / 2), str(o),
                       f_small, (255, 255, 255) if o >= 3 else (40, 60, 85))

    # ------------------------------------------------------------- counter --
    covered = 7 * (step + 1) if active else (49 if step >= C else 0)
    cy = gy + C * gc + 22 * s
    d.text((gx, cy), f"{covered} / 49 chunk pairs covered", font=f_big,
           fill=ACCENT if active else (INK if step >= C else INK3))
    bar_w = C * gc - 3 * s
    d.rounded_rectangle([gx, cy + 30 * s, gx + bar_w, cy + 40 * s], radius=5 * s,
                        fill=EMPTY, outline=GRID, width=2 * s)
    if covered:
        d.rounded_rectangle([gx, cy + 30 * s, gx + bar_w * covered / 49, cy + 40 * s],
                            radius=5 * s, fill=ACCENT if active else RAMP[6])

    if step >= C:
        d.text((44 * s, 476 * s), "7 subproblems  ×  7 chunk pairs  =  49",
               font=f_big, fill=INK)
        d.text((44 * s, 506 * s),
               "every pair exactly once — the recomposed output is exact",
               font=f_lab, fill=INK2)

    d.text((44 * s, H * s - 30 * s), "github.com/yiming-b/Stream-CQSA",
           font=f_small, fill=INK3)
    return im.resize((W, H), Image.LANCZOS)


def main():
    frames, durations = [], []
    frames.append(frame(-1)); durations.append(1100)          # the empty map
    for i in range(C):
        frames.append(frame(i)); durations.append(1250)
    frames.append(frame(C)); durations.append(2600)           # complete, held

    out = os.path.join(HERE, "stream_cqsa_forward.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True,
                   disposal=2)
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KiB, "
          f"{len(frames)} frames, {sum(durations)/1000:.1f} s loop)")

    for tag, im in (("still_first", frames[1]), ("still_last", frames[-1])):
        p = os.path.join(HERE, f"stream_cqsa_forward_{tag}.png")
        im.save(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
