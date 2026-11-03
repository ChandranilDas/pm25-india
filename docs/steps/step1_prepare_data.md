# Step 1: Prepare the data

**Status:** done (2026-11-03).

## Purpose
Turn the 60 monthly CAMS files into one clean dataset that the next steps can use.

## Idea in plain words
CAMS gives a value every 3 hours on a grid of 0.75° (about 80 km) squares. We
average the 8 values of each day into one daily value, put the units into the
usual ones (µg/m³, °C, hPa) and keep only the squares inside India.

## How it works
Script: `step1_prepare_data.py`.
1. Read all 60 files and join them along time.
2. Rename to short names (time, lat, lon) and sort latitude from south to north.
3. Daily averages with `resample(time="1D").mean()`.
4. Units: PM2.5 × 10⁹ (kg/m³ → µg/m³), temperatures − 273.15 (K → °C),
   pressure ÷ 100 (Pa → hPa).
5. Add the region of each grid cell from `data/india_regions.nc` (made in the
   first project from Indian state borders: igp, northwest, central, south,
   northeast, himalaya).
6. Make a table (one row per day and grid cell), keep India cells, add month and
   season.

## Outputs
| File | Content |
|---|---|
| `data/pm25_daily.nc` | daily data on the full grid (used for the maps in step 6) |
| `data/pm25_daily.csv` | table: date, lat, lon, region, month, season, pm25, aod, u10, v10, t2m, d2m, sp |

## How to run
    conda run -n pm25 python step1_prepare_data.py

## Decisions
- **Daily averages instead of 3-hourly values.** 8 times fewer rows (0.9 million
  instead of 7.2 million), so everything runs fast on a laptop, and daily PM2.5 is
  what air-quality limits use.
- **Only grid cells inside India** (at least half of the square in India).

## Problems met
None.

## Results
- 60 files, 1,827 days (2020-01-01 to 2024-12-31), 494 grid cells in India,
  902,538 rows, no missing values.

## Open issues
None.
