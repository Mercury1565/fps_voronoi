# Warm-started FPS resampling

An **alternative to Phase 3**. Phase 3 keeps a sample set healthy by *editing* it —
explicit insertions, evictions, and one-hop Voronoi updates under a budget. This
module takes a different route suggested: treat every frame as a
**partial rebuild**.

> Instead of manually inserting and evicting, keep the carried samples that are
> still valid, and run Farthest-Point Sampling *starting from them* to re-sample
> the rest from the cloud.

If 80% of the previous samples are still good, you reuse those 80% as FPS seeds
and let FPS choose the remaining 20% — which, by FPS's nature, land in the
worst-covered gaps (including the holes left by the samples you dropped).

## The concept in three steps

```
   carried samples S_prev ─┐
                           ▼
   1. classify   valid_mask(P, S_prev)        which carried samples are still valid?
                           │                  drop: vanished · stale · redundant
                           ▼
   2. keep       S_valid = S_prev[keep]       the survivors become FPS seeds
                           │
                           ▼
   3. refill     fps_continue(P, S_valid, M)  greedy FPS from the seeds up to M
                           │                  new points fill the worst gaps
                           ▼
                    new samples S  (exactly M)
```

A sample is dropped when it is:

| Reason | Test | Meaning |
|---|---|---|
| **vanished** | cell occupancy `< min_occupancy` | nothing maps to it any more |
| **stale** | distance to nearest cloud point `> stale_dist` | it floats in empty space the scene left behind |
| **redundant** | within `separation_min` of another kept sample | two samples doing one sample's job |

The number that survives is the **validity fraction**. It is a single smooth
knob: a static scene keeps almost everything (tiny refill, nearly free); a
fast-changing scene keeps little (large refill, approaching a full rebuild);
zero valid ⇒ it *is* a cold rebuild. There is no patch-or-rebuild branch and no
edit budget.

## What this module deliberately does *not* do

- **No incremental Voronoi bookkeeping.** Every frame recomputes the assignment
  from scratch (FPS produces it as it runs). The only state carried between
  frames is the sample positions `S`. This is simpler than Phase 3 but costs
  ~full-rebuild compute each frame (an `O(M·N)` reassignment), so it trades
  Phase 3's "cheap when stable" property for uniformity and simplicity.
- **No moving of samples.** A carried sample is kept or dropped, never nudged —
  FPS only ever *adds*. The validity test is what prunes badly-placed survivors.

## Files

| File | What it is |
|---|---|
| `resample.py` | the whole concept: `valid_mask`, `fps_continue`, `warm_resample`. |
| `temporal_demo.py` | carries a sampling across real LiDAR frames and reports keep/drop/refill + chamfer vs. a cold rebuild. |
| `tests/test_resample.py` | invariants: seeds preserved, count constant, stale/redundant samples dropped. |

### `resample.py` API

```python
warm_resample(P, S_prev, num_samples,
              min_occupancy=1, stale_dist=None, separation_min=None, seed=None)
    -> WarmResampleResult(S, n_kept, n_dropped, n_refilled, valid_fraction)
```

`fps_continue(P, seeds, M)` is ordinary FPS when `seeds is None`, and a warm
continuation otherwise — the seeds occupy the first rows of the result, so they
are preserved exactly. The continuation is deterministic; only a cold start uses
the random `seed`.

## Running it

```bash
# constant M, partial rebuild each frame, compared against a fresh cold FPS
python warm_fps/temporal_demo.py --dataset kitti --num-frames 30 \
    --max-range 40 --min-z -1.5 --baseline

python -m pytest warm_fps/tests/ -q
```

Defaults come from `config.py` / `.env` (`SAMPLES`, `DIMS`, `MAX_RANGE`, `MIN_Z`).
Validity thresholds are data-adaptive — a factor times the median sample spacing:

| Flag | Default | Meaning |
|---|---|---|
| `--min-occupancy` | `1` | drop cells holding fewer points than this |
| `--stale-factor` | `2.0` | drop samples farther than `factor × spacing` from any point (`0` disables) |
| `--separation-factor` | `0.5` | drop the lesser of two samples closer than `factor × spacing` (`0` disables) |
| `--baseline` | off | also cold-rebuild fresh FPS each frame and report `fps_cham`, `ratio`, `direct` |

### Output columns

| Column | Meaning |
|---|---|
| `kept` | carried samples judged valid and reused as FPS seeds |
| `drop` | carried samples discarded (vanished / stale / redundant) |
| `refill` | points re-sampled by the FPS continuation (`= M − kept`) |
| `chamfer` | `P ↔ warm S` — how close the warm sampling is to the cloud (m) |
| `fps_cham` | `P ↔ cold FPS` at equal M — the from-scratch reference (m) |
| `ratio` | `chamfer / fps_cham`; `1.0` = as good as a cold rebuild |
| `direct` | `warm S ↔ cold FPS` — sample-to-sample distance (cautionary; see top-level README) |

## What it shows (and the knob to turn)

On KITTI at M=512 with the default thresholds, validity stays high (~85% kept)
and `ratio ≈ 2.0` — about the same as Phase 3's insert/evict. That is expected:
keeping that many carried positions also keeps their accumulated drift, so the
warm sampling sits ~2× farther from the cloud than a fresh rebuild of equal size.

The lever is the **validity fraction**. Tightening the validity tests
(e.g. a smaller `--stale-factor`) drops more samples and refills more from the
current cloud, pulling `ratio` toward 1.0 — at the cost of reusing less and doing
more FPS work. That continuum, from cheap-but-drifted to rebuild-quality, is the
whole point of the approach and is what this demo is for measuring.
