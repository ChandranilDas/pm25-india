# Step 6: Finer PM2.5 maps (0.25°)

**Status:** done (2026-11-03).

## Purpose
Use the trained model together with spatial interpolation to make PM2.5 maps that
are 3 times finer than CAMS (0.25° instead of 0.75°), and compare them with the
standard method (RQ6).

## Idea in plain words
- **Standard way (bilinear interpolation):** take CAMS PM2.5 and smoothly
  interpolate it between the 4 surrounding grid cells.
- **ML way:** interpolate the predictors (AOD, wind, temperature, pressure, …) to
  the fine points in the same way, then let the random forest compute PM2.5 there.
  The map can then follow details in AOD and terrain.

## How it works
Script: `step6_maps.py`.
1. Make a 0.25° grid. A fine point counts as inside India when its nearest CAMS
   cell is inside India.
2. For the chosen days, interpolate all variables to the fine grid with
   `interp` (bilinear).
3. Random forest prediction at every fine point and day; average over the days.
4. Draw CAMS 0.75°, bilinear 0.25° and random forest 0.25° side by side, for one
   example day (2024-11-15) and for the 2024 average.

## Outputs
| File | Content |
|---|---|
| `results/step6/map_example_day.png` | maps for 2024-11-15 |
| `results/step6/map_2024_mean.png` | maps of the 2024 average |
| `results/step6/map_check_2024.csv` | India-wide 2024 averages of each map |

## How to run
    conda run -n pm25 python step6_maps.py    # about 1 minute

## Decisions
- **Example day 2024-11-15**: a smoggy November day, so the differences are easy
  to see.
- **India mask from the nearest CAMS cell** (simple; the edge of the map is
  therefore a bit blocky).

## Problems met
- `interp` on the whole dataset (which also holds the text variable `region`)
  messed up the time column. Interpolating only the number variables fixed it.

## Results
2024 average over India:

| Map | PM2.5 (µg/m³) |
|---|---|
| CAMS 0.75° | 72.8 |
| Bilinear 0.25° | 72.4 |
| Random forest 0.25° | 71.5 |

- Average difference between the random forest and bilinear maps: 5.4 µg/m³.
- Both fine maps keep the CAMS pattern (high over the IGP, low in the south and the
  mountains). The bilinear map is smooth; the random forest map shows sharper
  edges, for example along the foot of the Himalaya, where pressure and AOD change
  quickly. It is also a little smoother at the biggest hotspots (Delhi area),
  because a random forest averages many training rows.
- There is no finer "true" PM2.5 to check the 0.25° maps against; step 5 (test
  cells) is the accuracy check.

## Open issues
None.
