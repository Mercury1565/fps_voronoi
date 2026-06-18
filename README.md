# FPS-Voronoi — adaptive point-cloud sampling

## The big picture

When you process a 3-D point cloud (say, a LiDAR scan with hundreds of thousands
of points), you usually can't afford to work with every point. You pick a much
smaller set of representative **samples** and work with those instead.

A common way to pick them is **Farthest Point Sampling (FPS)**: start somewhere,
then keep adding the point that is farthest from everything chosen so far. This
spreads samples out nicely, but it's purely geometric — it doesn't know whether
the result actually does a *good job* of representing the cloud. Some regions can
end up under-covered, some samples can land almost on top of each other, and some
samples can end up representing almost nothing.

This project is building a system that **checks the quality of a sampling and
tells you how to improve it**. The idea:

1. Pick samples `S` from the cloud `P` (FPS).
2. Each cloud point "belongs to" its nearest sample. This silently carves the
   cloud into regions — one per sample — called **Voronoi cells**. (We never
   build the cells explicitly; we just ask which sample each point is closest
   to.)
3. Measure the health of those cells and flag three kinds of problems:
   - **Coverage gaps** — a cell stretches over too large an area → we're
     under-sampling there, add a sample.
   - **Separation violations** — two samples sit redundantly close → wasteful,
     drop one.
   - **Vanishing cells** — a sample represents (almost) no points → wasted
     sample, remove or move it.
4. Act on those flags — insert/remove samples and repeat — so the sampling
   *adapts* to the actual shape of the data.

So the end goal is an **adaptive resampler**: instead of trusting raw FPS, we
keep refining the sample set until the cloud is covered evenly and efficiently.

Throughout the project two letters are used consistently: **`P`** is the point
cloud (the raw LiDAR points of one frame), and **`S`** is the set of samples
drawn from it.

---

## What's been built so far

The work is split into phases. Each phase lives in its own folder with its own
detailed README.

### Phase 1 — the measuring tools (`phase_1/`)

Five small, independent building blocks that each answer one geometric question
about the cloud-and-samples, all running on CPU or GPU:

| Tool | Description |
|------|-----------------------------------|
| **cell membership** | Which sample is each cloud point closest to, and how far? |
| **covering radius** | How far does each cell reach? (its most distant point) |
| **min pairwise distance** | Which two samples are closest together? |
| **cell occupancy** | How many points does each sample actually represent? |
| **Delaunay neighbors** | Which samples are neighbours of which? |

These are the raw instruments. On their own they just report numbers — they
don't make any judgements yet. Each one is carefully tested (including
checks that the CPU and GPU give identical answers).

### Phase 2 — fusing the tools and making judgements (`phase_2/`)

Phase 2 does two things.

**First, it fuses the measurements into a single efficient pass.** In Phase 1,
finding each point's nearest sample, measuring how far cells reach, and counting
points per cell were three separate sweeps over the data. The expensive part —
comparing every point against every sample — was repeated. Phase 2 does all of
that in **one sweep**: it computes:

- the nearest sample,
- the distance,
- the per-cell reach,
- the per-cell point count, and
- even *which* point sits farthest out in each cell

ALL AT ONCE. This is the performance win that makes the system practical on
large clouds, and it's structured so a hand-written GPU kernel can slot in later.

**Second, it turns measurements into decisions.** Three detectors apply
thresholds to the fused statistics and flag problem cells:

- **Coverage gap detector** → lists the over-stretched cells *and* hands back a
  concrete point where a new sample should be inserted.
- **Separation violator detector** → lists the sample pairs that are too close.
- **Vanishing cell detector** → lists the samples that represent too few points.

The thresholds can be set to exact values, or derived automatically from the
data (e.g. "flag anything more than twice the typical cell size"), so the system
adapts to whatever cloud it's given.

Phase 2 also ships a visualization that draws all three problems on a 2-D scene,
so you can *see* where the sampling is weak:

![Phase 2 detectors](phase_2/phase2_demo.png)

### Phase 3 — acting on the flags (`phase_3/`)

Phase 2 only *reports* problems. Phase 3 **fixes** them: it edits the sample set
and keeps all the Voronoi bookkeeping up to date — crucially, **without redoing
the expensive full nearest-neighbour search** each time. This is the piece that
closes the loop and turns the diagnostics into an actual adaptive resampler.

It has four parts:

- **Insertion** → adds a sample at a coverage gap, and figures out which existing
  samples border the new cell (its Delaunay neighbours).
- **Eviction** → removes wasteful samples in a sensible priority order: vanished
  cells first, then underpopulated ones, then the smallest (most redundant)
  cells.
- **One-hop update** → after an edit, only the handful of *neighbouring* cells
  are recomputed, not the whole cloud. This is what keeps corrections cheap, and
  it's provably identical to a full recompute.
- **Budget tracker** → counts how many edits a frame needs. If a sampling is so
  bad that it needs more fixes than the budget allows, Phase 3 stops patching and
  just rebuilds the whole thing from scratch with FPS.

Run repeatedly, Phase 3 drives a bad sampling toward a healthy one and settles at
a stable point where nothing more needs fixing — the worst cell shrinks and the
closest, most redundant samples get cleaned out.

---

## How the phases fit together

```
   point cloud P  ──►  FPS  ──►  samples S  ◄─────────────────┐
                                    │                          │
        Phase 1: measuring tools    │   (nearest sample, cell  │
        (the raw instruments)       │    reach, occupancy, …)   │
                                    ▼                          │
        Phase 2: one fused pass  ──►  health statistics         │
                                    │                          │
        Phase 2: three detectors    ▼                          │
                              ┌─────────────────────────────┐  │
                              │ coverage gaps  → add here    │  │
                              │ too-close pairs → drop one   │  │
                              │ vanishing cells → remove     │  │
                              └─────────────────────────────┘  │
                                    │                          │
        Phase 3: correction unit    ▼                          │
            insert / evict, one-hop updates, under a budget ───┘
            (or full-FPS fallback if too many fixes are needed)
```

Phase 1 gives us **trustworthy measurements**. Phase 2 makes them **fast** (one
pass) and **actionable** (flags + suggested fixes). Phase 3 **closes the loop** —
it feeds the flags back into the sampler, inserting and evicting samples to fix
the weak spots, and recomputes only what changed. Iterated, the sampling
improves until it stabilises.

---

## Working with real LiDAR data

The phases above were first demonstrated on a synthetic 2-D scene, for clarity
and easy visualization. The same engine runs **unchanged on real 3-D LiDAR**:
every primitive is dimension-agnostic (they use `torch.cdist` / `torch.norm` /
`scipy.Delaunay`, all of which work in any dimension), so feeding real data is
purely a *loading* concern, not an algorithm change.

### The loader — `data_io.py`

A single shared module at the repo root reads LiDAR frames into the point-cloud
tensor `P`:

| Function | Purpose |
|---|---|
| `load_lidar_bin(path, dims=3, max_range=None, min_z=None)` | Load one frame into an `(N, dims)` tensor. |
| `list_frames(dataset)` | Sorted list of frame paths for `"nuscenes"` or `"kitti"`. |
| `load_frame(dataset, index, **kw)` | Convenience: load the `index`-th frame by dataset name. |
| `chamfer_distance(A, B)` | Symmetric Chamfer distance, returned split into its two directed halves (see the temporal section). |

Two on-disk formats are **auto-detected by file extension**:

| Dataset | File | Binary layout | Typical size |
|---|---|---|---|
| KITTI | `*.bin` | `float32`, reshape `(-1, 4)` = x, y, z, intensity | ~121k pts/frame |
| nuScenes | `*.pcd.bin` | `float32`, reshape `(-1, 5)` = x, y, z, intensity, ring | ~35k pts/frame |

(Same conventions as the `extract/` scripts.) Only the first three columns
(x, y, z) are kept; intensity/ring are dropped.

Loader options:

- **`dims`** — `3` keeps full (x, y, z); `2` projects to a top-down (x, y) view so
  the original 2-D visualizations keep working. The numeric engine is identical
  either way; only the matplotlib panels are 2-D-specific (and the intrinsically
  2-D scipy Voronoi ridge overlay is drawn only when `dims == 2`).
- **`max_range`** — drop points beyond this horizontal radius (metres), trimming
  the sparse long-range fringe that bloats covering radii.
- **`min_z`** — drop points below this height (rough ground removal).

> **Why crop / remove ground?** On raw, uncropped LiDAR, FPS tends to grab the
> sparse far/high/low outliers as samples and leaves one giant central cell (on a
> raw nuScenes frame, ≈ 22k of 35k points landed in a single cell). Cropping with
> `--max-range` and removing the ground with `--min-z` keep the sampling
> meaningful.

### Running the demos on real frames

Both the Phase 1 and Phase 2 demos accept the same set of flags:

```bash
# Phase 1 — primitives + visualization on a real nuScenes frame (full 3-D, cropped)
python phase_1/demo.py --dataset nuscenes --frame 0 --dims 3 --max-range 40 --min-z -1.5

# Phase 2 — the three detectors on a real KITTI frame, top-down 2-D
python phase_2/demo.py --dataset kitti --frame 0 --dims 2
```

| Flag | Meaning |
|---|---|
| `--dataset {nuscenes,kitti}` | Which dataset to load from `data/`. |
| `--frame N` | Frame index (nuScenes mini: 0–404; KITTI drive 0002: 0–76). |
| `--dims {2,3}` | Run in full 3-D or top-down 2-D (default `3`). |
| `--max-range M` | Horizontal crop radius in metres. |
| `--min-z Z` | Drop points below height `Z` (ground removal). |

You can also pass a raw file path directly
(`python phase_1/demo.py path/to/frame.pcd.bin`), and with **no** dataset/file
argument both demos fall back to the synthetic 2-D scene.

> **Two distinct data products live in this repo — don't confuse them:**
> - `data/nuscenes/.../LIDAR_TOP/*.pcd.bin` and
>   `data/kitti/.../velodyne_points/data/*.bin` are the **raw point clouds** —
>   the pipeline's input `P`.
> - The `extract/` scripts together with `data/csv` and `data/json` are a
>   **separate product**: per-frame scene metadata (object boxes, ego velocity,
>   a cloud-to-cloud chamfer distance, a confidence label) intended for a
>   downstream confidence model. They are *not* point-cloud input to FPS-Voronoi.

---

## Measuring frame-to-frame change (the temporal loop)

The single-frame engine tells you whether *one* cloud is sampled well. For a
LiDAR stream the natural question is: **how much do consecutive frames differ?**
`phase_3/temporal_demo.py` answers it by measuring how much the sampling has to
*adapt* from one frame to the next.

### How it works

1. **Frame 0** — run FPS on the first cloud to get a sample set `S`.
2. **Each later frame** — carry `S` forward and run Phase 3's `correct()` on the
   new cloud. It diagnoses the carried-forward samples against the new frame and
   patches them (insert at coverage gaps, evict vanished / redundant cells) using
   cheap one-hop updates.
3. The corrected `S` becomes the input for the next frame, and so on.

The number of edits a frame *requests* is the headline difference signal: a
near-static scene needs almost no edits; a fast-changing one needs many.

```bash
python phase_3/temporal_demo.py --dataset kitti --num-frames 30 \
    --dims 3 --max-range 40 --min-z -1.5
```

Flags, in addition to `--dataset` / `--dims` / `--max-range` / `--min-z` above:

| Flag | Meaning |
|---|---|
| `--num-frames N` | How many consecutive frames to process. |
| `--start I` | First frame index. |
| `--samples M` | Target FPS sample count (default 64). |
| `--fixed-samples` / `--no-fixed-samples` | Hold the sample count constant at `--samples` every frame (**default on**), or let it drift as insertions/evictions fall out. See below. |
| `--budget B` | Max edits/frame before a full FPS rebuild (default 40). |
| `--coverage-factor`, `--separation-factor`, `--min-occupancy` | Detector thresholds. |
| `--baseline` | Also rebuild a fresh FPS each frame and report `fps_cham`, the `ratio`, and the `direct` distance. |
| `--out PATH` | Output plot path (default `temporal_demo.png`). |

**Holding the sample count constant.** By default the demo pins the sample count
at `--samples` (e.g. 64) on *every* frame. `correct()` itself inserts and evicts
independently, so left alone the count drifts (evictions tend to outpace
insertions). After each correction the demo re-pins it: if `S` came back short it
**tops up the worst-covered spots** (an FPS continuation), and if it came back
over it **drops the most-redundant samples** (closest-pair first). The `edits` /
`ins` / `evt` columns still report the *raw* correction (the scene-change signal);
`M` is then restored to the target — so `M` reads a constant 64 throughout, and
`fps_cham` is always a fair fight at that same count. Pass `--no-fixed-samples` to
watch the count drift instead.

### Patch vs. rebuild: the budget threshold

There are **two threshold layers**, and only one of them decides whether to
reuse the previous samples or rebuild from scratch:

- **Detector thresholds** decide *how many edits a frame wants*. They are
  **data-adaptive**: a coverage gap is `coverage_factor (2.0) × the frame's own
  median covering radius`; a separation violation is `separation_factor (0.5) ×
  the median sample-to-sample nearest distance`; a vanishing cell is
  `occupancy < min_occupancy (1)`. Each can instead be pinned to an absolute
  value.
- **The budget** decides *patch vs. full rebuild*. After diagnosing, the frame
  tallies `n_requested = insertions + evictions`. If `n_requested > budget`,
  Phase 3 throws the carried sampling away and re-runs FPS from scratch on that
  frame (the `note` column reads `FULL FPS REBUILD`); otherwise it patches
  incrementally. The budget is a **fixed integer you choose** (default 40) — a
  compute ceiling, *not* derived from the data.

### How to read the output

Each row is one frame. The columns answer three separate questions, and it is
easiest to read them in those groups rather than left to right.

**A worked example.** Take this row from a real KITTI run (with `--baseline`):

```
 frame     pts    M  edits  ins  evt  chamfer  fps_cham  ratio   direct  note
    10   37721   54      1    0    1    5.511     2.521   2.19    6.912
```

Read it as: *frame 10 has 37,721 points; we are now carrying 54 samples; this
frame applied 1 patch (added 0, dropped 1); the carried sampling sits 5.51 m from
the cloud, while a fresh FPS rebuild at the same 54 samples sits only 2.52 m —
so reuse is **2.19× worse** here.*

**Question 1 — how much did the scene change?** (`edits`, `ins`, `evt`)

`edits = ins + evt` is the headline signal: how many samples had to be patched to
keep up with the new frame. A near-static scene needs ~0; a fast-changing one
needs many. `ins` are samples **added** where new structure appeared; `evt` are
samples **removed** where a region emptied out or went redundant. `edits` is also
what gets compared against `--budget`.

**Question 2 — how good is the sampling, and is reuse worth it?**
(`chamfer`, `fps_cham`, `ratio` — distances in **metres, lower = closer**)

All three are built on the **Chamfer distance**: the mean L2 distance from each
point in one set to its nearest point in the other, averaged in both directions.
Crucially, every quality number is measured **against the cloud `P`**, because the
cloud is the unique, dense thing the sampling is trying to represent — judging two
sparse sample sets against each other instead would mostly measure *which points
they happened to pick*, not how well they cover the scene.

| Column | Between | Meaning |
|---|---|---|
| `chamfer` | `P` ↔ carried `S` | How far the **reused / patched** sampling is from the current cloud. |
| `fps_cham` | `P` ↔ fresh `Sf` | How far a **from-scratch FPS** at the same `M` is from the same cloud — the best achievable at this budget (a fair fight). |
| `ratio` | `chamfer / fps_cham` | The punchline: **1.0 = reuse is as good as rebuilding; 2.0 = reuse sits twice as far from the cloud.** |

`fps_cham`, `ratio` and `direct` are shown only with **`--baseline`** (it builds
the extra fresh FPS each frame).

**Question 3 — why *not* compare the two sample sets directly?** (`direct`)

`direct` is the Chamfer distance **between the two sample sets**, carried `S` ↔
fresh `Sf` — the seemingly-obvious "how close is the correction to a rebuild?"
number. It is included **only as a cautionary comparison**: in practice it runs
*higher than either cloud-grounded distance* (≈ 5–7 m vs. `chamfer` ≈ 3–5 and
`fps_cham` ≈ 2.5). That inflation is **alignment noise**, not real error — FPS has
many equally-good solutions, so `S` and `Sf` pick different (but equally valid)
points, and comparing them directly penalises that arbitrary disagreement. This
is exactly why the quality verdict uses `ratio` (both sides grounded on `P`) and
not `direct`.

**The remaining bookkeeping columns:**

| Column | What it is |
|---|---|
| `frame` | Frame index in the dataset (`--start` offsets it). Row 0 is the FPS baseline. |
| `pts` | Points in this frame's cloud `P` after `--max-range` / `--min-z`. Varies as the scene changes. |
| `M` | Sample count *after* this frame's correction. Starts at `--samples`; drifts as insertions add and evictions remove samples. |
| `note` | `FPS baseline` on frame 0; `FULL FPS REBUILD` if the budget was exceeded that frame. |

The generated plot has two stacked panels: **(top)** edits requested per frame
(insertions vs. evictions, with the budget line and any rebuilds marked), and
**(bottom)** the three chamfer distances over time — `P`↔`S`, `P`↔`Sf`, and the
`direct` `S`↔`Sf` — with the `ratio` drawn on a secondary axis (only with
`--baseline`).

### What the experiments show

- **Consecutive frames differ little once the sampling settles.** nuScenes and
  KITTI are 10 Hz, so successive sweeps are very similar; after the carried
  sampling adapts in the first ~10–15 frames, edit counts fall to 0–2 and the
  budget fallback never fires on a continuous drive.
- **Reuse is mechanically sound.** The one-hop updates are provably identical to
  a full recompute, and the sampling converges to a stable equilibrium rather
  than diverging.
- **But reuse is not free.** With `--baseline`, the chamfer `ratio` settles around
  **~2.0** — the reused sampling sits about twice as far from the cloud as a fresh
  rebuild of the same size. The cause is **staleness**: carried samples drift to
  spots the scene no longer occupies, so `chamfer` (P↔S) climbs while `fps_cham`
  (P↔Sf) stays flat at ~2.5 m.
- **Don't be fooled by the `direct` column.** The direct carried-vs-fresh distance
  (`S`↔`Sf`) runs *higher* than either cloud-grounded distance, because FPS has
  many equally-good solutions and the two sets simply pick different points. That
  gap is alignment noise, not correction error — which is why the verdict is the
  `ratio` of cloud-grounded distances, not `direct`.
- **Holding `M` constant removes the coarsening.** With `--no-fixed-samples` the
  count drifts down (e.g. 64 → ~40 over a run, since evictions outpace
  insertions); the cells grow and `chamfer` climbs steadily (to ~5.7 m by frame
  25). With the default `--fixed-samples` the count stays at 64, and `chamfer`
  plateaus instead of climbing (~3.8 m), because samples are topped back up at the
  worst-covered spots. The `ratio` still sits ~1.5–2.0 — that residual is
  **staleness** (carried samples drifting onto empty space), not sample-count
  loss, so a fresh rebuild at equal `M` remains meaningfully closer to the cloud.

---

## Trying it out

All phases share a single virtual environment at the repo root. Set it up once:

```bash
# From the repo root
python -m venv venv                      # if not already present
venv/bin/pip install -r requirements.txt
```

Then activate it and run any phase:

```bash
source venv/bin/activate

# ── Phase 1 — primitives, tests, demo ───────────────────────────────────────
python phase_1/demo.py                                   # synthetic 2-D scene
python phase_1/demo.py --dataset nuscenes --dims 3 --max-range 40 --min-z -1.5
python -m pytest phase_1/tests/ -q

# ── Phase 2 — fused pipeline, detectors, visualization ───────────────────────
python phase_2/demo.py                                   # renders phase2_demo.png
python phase_2/demo.py --dataset kitti --frame 0 --dims 3 --max-range 40 --min-z -1.5
python -m pytest phase_2/tests/ -q

# ── Phase 3 — correction unit + temporal loop ───────────────────────────────
python -m pytest phase_3/tests/ -q
python phase_3/temporal_demo.py --dataset kitti --num-frames 30 \
    --dims 3 --max-range 40 --min-z -1.5 --baseline
```

> **Tip:** run the test suites one phase at a time. Each phase has its own
> `tests/conftest.py`, and collecting all three directories in a single `pytest`
> invocation triggers a conftest name collision.

> **Shell tip:** keep a command on one line, or use the `=` form for negative
> values (`--min-z=-1.5`), so the shell doesn't split `-1.5` off as its own token.

Per-phase documentation:

- **`phase_1/README.md`** — the five primitives, their exact inputs/outputs, and
  conventions.
- **`phase_2/README.md`** — the fused pass, the three detectors, threshold
  options, and the API.
- **`phase_3/README.md`** — insertion, eviction priority, one-hop updates, the
  budget tracker, and the `correct()` frame.
