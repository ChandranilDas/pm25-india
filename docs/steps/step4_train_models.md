# Step 4: Train the models

**Status:** done (2026-11-03).

## Purpose
Predict daily PM2.5 at the test cells with three methods: one standard
interpolation method and two ML models (RQ1, RQ2, RQ6).

## Idea in plain words
- **IDW (inverse distance weighting)** does not use any predictors. It guesses
  PM2.5 at a test cell from the PM2.5 of the 8 nearest training cells; a cell
  twice as close counts four times as much (weight = 1 / distance²).
- **Linear regression** fits a straight-line formula:
  PM2.5 = a × AOD + b × wind + … + constant.
- **Random forest** builds 100 decision trees ("if AOD > 0.6 and pressure > 950
  then …") and averages their answers. It can learn curved links and combinations.

Both ML models use the CAMS predictors at the test cell itself: AOD, u10, v10,
t2m, d2m, sp, latitude, longitude and month. They never see PM2.5 at the test
cells.

## How it works
Script: `step4_train_models.py`.
1. Join the table with `data/cells.csv` to know which rows are train and test.
2. IDW: for each test cell, find the 8 nearest training cells, compute the
   weights, and take the weighted average of their PM2.5 on every day.
3. Linear regression and random forest: `fit` on the training rows, `predict` on
   the test rows.
4. Save all predictions in one table, and save the random forest for step 6.

## Outputs
| File | Content |
|---|---|
| `results/step4/test_predictions.csv` | test rows: date, lat, lon, region, season, pm25 (CAMS), idw, linear, random_forest |
| `results/step4/random_forest.joblib` | the trained random forest (not in git, too big) |

## How to run
    conda run -n pm25 python step4_train_models.py    # 1–2 minutes

## Decisions
- **Two ML models: linear regression and random forest.** Linear regression is
  the simplest ML model; random forest is a standard non-linear model. Both come
  from scikit-learn, so no extra library is needed.
- **Random forest settings:** 100 trees, at least 10 rows per leaf (stops the
  trees from learning noise), `random_state=42` so the result is repeatable.
- **IDW with the 8 nearest training cells, power 2** (the usual textbook choice).
  Distance is measured in degrees, which is good enough to find neighbours.
- **Month as a predictor** so the models can learn the seasonal cycle.

## Problems met
- The saved random forest was 500 MB. Saving it compressed (`compress=3`) makes it
  about 170 MB.

## Results
- 697,914 training rows, 204,624 test rows.
- Linear regression coefficients: AOD +95.8 per unit of AOD, the others small
  (u10 +3.0, v10 −1.6, t2m −1.2, d2m −2.8, sp +0.45, lat +2.4, lon +1.2,
  month +1.4).
- The scores are in step 5.

## Open issues
None.
