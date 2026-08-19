# Interactive walkthrough

`index.html` is a single self-contained page that steps through the whole
method — decomposition, quorum selection, gather, masking, the `(acc, l, m)`
merge, finalisation, and then the backward pass — recomputing every number live
from the controls.

**Live version:** https://claude.ai/code/artifact/c4e4c79d-9184-4ed9-aa4e-8543181f7b75

## Running it

Open `index.html` in a browser. There is no build step and no bundler; the only
external request is to Google Fonts, and it degrades to system fonts offline.
To serve it over HTTP (for GitHub Pages, say):

```bash
python -m http.server -d docs/demo 8000     # then visit localhost:8000
```

## What it is not

A visualisation, not an implementation. It runs a JavaScript port of
`stream_cqsa.cqs_mask` in fp64 on toy tensors (`D = 4`) so the matrices fit on
screen. The production path is `stream_cqsa.stable_stream`.

The port is **verified against the Python** rather than eyeballed: `token_ids`
and the keep mask match `CQS_mask.gen_mask` and `cqs_keep_mask` exactly across
c ∈ {7, 13, 21, 73, 133}, single- and two-level quorum paths.

## The (c, interest set) presets

Each is a planar (Singer) difference set: `c = q² + q + 1`, `|I| = q + 1`,
λ = 1. All eight were verified — the `k(k−1)` nonzero differences hit every
residue mod `c` exactly once, which is precisely why the pair cover is a
partition.

| c | order q | \|I\| | interest set |
|:--:|:--:|:--:|:--|
| 7 | 2 | 3 | 0, 1, 3 |
| 13 | 3 | 4 | 0, 1, 3, 9 |
| 21 | 4 | 5 | 0, 1, 4, 14, 16 |
| 31 | 5 | 6 | 0, 1, 3, 8, 12, 18 |
| 57 | 7 | 8 | 0, 1, 3, 13, 32, 36, 43, 52 |
| 73 | 8 | 9 | 0, 1, 3, 7, 15, 31, 36, 54, 63 |
| 91 | 9 | 10 | 0, 1, 3, 9, 27, 49, 56, 61, 77, 81 |
| 133 | 11 | 12 | 0, 1, 3, 12, 20, 34, 38, 81, 88, 94, 104, 109 |

Any `(c, I)` can be typed in by hand. If it is not a λ=1 difference set the page
says so and names the defect — how many residues never appear as a difference,
how many appear twice — and then shows the consequence: uncovered pairs outlined
in red on the coverage map, and a forward error that is no longer machine
precision.
