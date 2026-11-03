# Step 2: Explore the data

**Status:** done (2026-11-03).

## Purpose
Look at the data before building models: where and when PM2.5 is high, and which
predictors move together with PM2.5.

## Idea in plain words
Simple averages and one correlation number per predictor. A correlation near +1
or −1 means a strong straight-line link with PM2.5; near 0 means no straight-line
link (there can still be a curved one).

## How it works
Script: `step2_explore.py`.
1. Average PM2.5 for each region and season (pivot table).
2. Correlation of each predictor with PM2.5.
3. Map of the 5-year average PM2.5 and a plot of monthly averages per region.

## Outputs
| File | Content |
|---|---|
| `results/step2/pm25_by_region_season.csv` | average PM2.5 per region and season |
| `results/step2/correlation_with_pm25.csv` | correlation of each predictor with PM2.5 |
| `results/step2/map_mean_pm25.png` | map of the 2020–2024 average |
| `results/step2/monthly_pm25_by_region.png` | monthly averages per region |

## How to run
    conda run -n pm25 python step2_explore.py

## Decisions
None.

## Problems met
None.

## Results
Average PM2.5 (µg/m³):

| Region | Winter | Pre-monsoon | Monsoon | Post-monsoon |
|---|---|---|---|---|
| IGP (Indo-Gangetic Plain) | 217.9 | 116.5 | 77.3 | 207.1 |
| Northwest | 85.1 | 69.2 | 67.4 | 84.9 |
| Central | 84.7 | 55.4 | 37.5 | 80.7 |
| Northeast | 68.4 | 82.3 | 22.5 | 43.9 |
| South | 63.4 | 49.9 | 30.2 | 62.2 |
| Himalaya | 51.8 | 37.4 | 29.8 | 46.8 |

- The IGP has by far the highest PM2.5, in winter and after the monsoon.
- The monsoon (rain) is the cleanest season everywhere.
- Correlation with PM2.5: AOD 0.56, surface pressure 0.32, latitude 0.19, all
  others between −0.14 and 0.05. AOD is the strongest single link (RQ3).

## Open issues
None.
