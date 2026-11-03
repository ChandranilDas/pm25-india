# Step 3: Training cells and test cells

**Status:** done (2026-11-03).

## Purpose
Choose which grid cells the models learn from (training) and which are kept back
to test them (test). The project brief asks for testing at held-out grid
locations (RQ5).

## Idea in plain words
Neighbouring grid cells have almost the same PM2.5. If we picked single test cells
at random, each would have a training cell right next to it and every method would
look too good. So we cut the map into blocks of 5 × 5 cells (about 380 km) and
keep whole blocks for testing.

## How it works
Script: `step3_split.py`.
1. Block row = (cell row ÷ 5), block column = (cell column ÷ 5).
2. A block is a test block when (block row + 2 × block column) is a multiple of 5.
   This picks every 5th block in a diagonal pattern, so test blocks are spread
   over the whole country.
3. All days of a test cell are test data; all days of a training cell are
   training data.

## Outputs
| File | Content |
|---|---|
| `data/cells.csv` | one row per grid cell: lat, lon, region, block row/column, set (train/test) |
| `results/step3/split_summary.csv` | number of training and test cells per region |
| `results/step3/split_map.png` | map of training and test cells |

## How to run
    conda run -n pm25 python step3_split.py

## Decisions
- **Blocks of 5 × 5 cells, fixed diagonal pattern.** A fixed pattern is easy to
  explain and to repeat, and needs no random seed. We checked a few patterns and
  kept the one where every region has between 10% and 30% of its cells in the
  test set, so every region can be scored.

## Problems met
- Some other patterns put almost half of the Himalaya, or none of it, in the test
  set. The chosen pattern does not.

## Results
| Region | Test cells | Training cells | Test share |
|---|---|---|---|
| Central | 42 | 108 | 28% |
| Himalaya | 9 | 28 | 24% |
| IGP | 22 | 62 | 26% |
| Northeast | 7 | 36 | 16% |
| Northwest | 10 | 73 | 12% |
| South | 22 | 75 | 23% |
| **All** | **112** | **382** | **23%** |

## Open issues
None.
