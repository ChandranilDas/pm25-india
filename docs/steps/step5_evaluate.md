# Step 5: Evaluate the predictions

**Status:** done (2026-11-03).

## Purpose
Measure how close each method is to CAMS PM2.5 at the test cells: overall, by
region and by season (RQ1, RQ2, RQ4, RQ5, RQ6), and find which predictors matter
most (RQ3).

## Idea in plain words
Compare each prediction with the true CAMS value and summarise the errors with
four numbers: RMSE, MAE, R² and bias (see [glossary](glossary.md)).

## How it works
Script: `step5_evaluate.py`.
1. A small function `scores` computes RMSE, MAE, R² and bias.
2. Scores for all test rows, then for each region and each season.
3. Bar charts of R² by region and season, and a scatter plot of the random forest
   against CAMS.
4. Feature importance: the random forest's built-in `feature_importances_` (how
   much each predictor helps the trees split the data).

## Outputs
| File | Content |
|---|---|
| `results/step5/scores_overall.csv` | scores of the three methods |
| `results/step5/scores_by_region.csv`, `scores_by_season.csv` | scores per region / season |
| `results/step5/r2_by_region.png`, `r2_by_season.png` | bar charts |
| `results/step5/scatter_random_forest.png` | random forest vs CAMS |
| `results/step5/feature_importance.csv`, `.png` | importance of each predictor |

## How to run
    conda run -n pm25 python step5_evaluate.py

## Decisions
- **Built-in random forest importance.** It is one line of code and easy to
  explain. It can favour predictors with many different values, so we also look
  at the correlations from step 2; both put AOD first.

## Problems met
None.

## Results

All test cells:

| Method | RMSE | MAE | R² | Bias |
|---|---|---|---|---|
| IDW | 35.3 | 20.4 | 0.68 | −6.8 |
| Linear regression | 46.7 | 32.1 | 0.45 | −12.9 |
| Random forest | **31.9** | **17.8** | **0.74** | −5.6 |

R² by region:

| Region | IDW | Linear | Random forest |
|---|---|---|---|
| Central | 0.70 | 0.44 | **0.75** |
| Himalaya | 0.46 | 0.49 | **0.78** |
| IGP | **0.56** | 0.29 | 0.54 |
| Northeast | −0.74 | 0.15 | **0.74** |
| Northwest | **0.80** | −0.05 | 0.75 |
| South | 0.45 | −1.80 | **0.47** |

R² by season:

| Season | IDW | Linear | Random forest |
|---|---|---|---|
| Winter | 0.61 | 0.37 | **0.69** |
| Pre-monsoon | 0.54 | 0.20 | **0.60** |
| Monsoon | 0.78 | 0.36 | **0.81** |
| Post-monsoon | 0.64 | 0.44 | **0.71** |

Feature importance (random forest): AOD 0.31, surface pressure 0.19,
temperature 0.14, dewpoint 0.10, latitude 0.09, longitude 0.08, u10 0.04,
v10 0.03, month 0.03.

What this means:
- The random forest is the best method overall and in most regions and all seasons.
- Linear regression is the weakest: a straight line cannot describe the link.
  In the south it is worse than just using the average (R² −1.8).
- IDW fails in the northeast (R² −0.74): some of the nearest training cells lie
  in other regions with different pollution (for example the IGP to the west).
  The random forest does not need neighbours, so it does well there.
- The IGP is hard for every method: PM2.5 is very high and changes fast.
- All methods predict slightly too low on average (negative bias).

## Open issues
None.
