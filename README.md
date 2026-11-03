# PM2.5 over India with machine learning (beginner version)

Course Project 3 (full brief: [docs/PROJECT.md](docs/PROJECT.md)).

**Goal:** estimate daily PM2.5 over India (2020–2024) from other CAMS variables
(aerosol optical depth, wind, temperature, dewpoint, surface pressure, location),
test it at grid cells the models never saw, and use it to make finer (0.25°) maps.

**Methods:** two ML models (linear regression and random forest), compared with a
standard interpolation method (IDW).

## How to run

Everything runs in the conda environment `pm25` (xarray, pandas, numpy,
scikit-learn, matplotlib, joblib, netCDF4). Run the steps in order:

    conda run -n pm25 python step1_prepare_data.py   # ~10 s
    conda run -n pm25 python step2_explore.py        # ~10 s
    conda run -n pm25 python step3_split.py          # ~5 s
    conda run -n pm25 python step4_train_models.py   # ~1-2 min
    conda run -n pm25 python step5_evaluate.py       # ~20 s
    conda run -n pm25 python step6_maps.py           # ~1 min

The CAMS files are not downloaded again: `data/raw` points to the files of the
first project (60 monthly files). Each step saves its tables and figures in
`results/stepN/`.

## Files

| File | What it does |
|---|---|
| `step1_prepare_data.py` | joins the monthly files, makes daily averages, converts units, makes a table |
| `step2_explore.py` | averages by region and season, correlations, first map and plot |
| `step3_split.py` | chooses the training and test grid cells |
| `step4_train_models.py` | IDW, linear regression and random forest predictions at the test cells |
| `step5_evaluate.py` | scores (overall, by region, by season) and feature importance |
| `step6_maps.py` | 0.25° maps: bilinear interpolation vs random forest |
| `docs/steps/` | one document per step, the journal and a glossary |
| `results/presentation/` | slides (14, with speaker notes) as PowerPoint and Keynote files |

## Main results (test grid cells, daily PM2.5)

| Method | RMSE (µg/m³) | MAE (µg/m³) | R² | Bias (µg/m³) |
|---|---|---|---|---|
| IDW interpolation | 35.3 | 20.4 | 0.68 | −6.8 |
| Linear regression | 46.7 | 32.1 | 0.45 | −12.9 |
| **Random forest** | **31.9** | **17.8** | **0.74** | −5.6 |

## Answers to the research questions

1. **How accurately can PM2.5 be estimated?** The random forest explains about
   74% of the daily variation (R² 0.74) at grid cells it never saw; typical error
   (MAE) is 18 µg/m³.
2. **How do the ML models compare?** The random forest is much better than linear
   regression (R² 0.74 vs 0.45), because the link between the predictors and
   PM2.5 is not a straight line.
3. **Most important features?** AOD (aerosol optical depth) by far, then surface
   pressure (which is low over the mountains and high over the plains) and
   temperature. Wind matters least.
4. **Regions and seasons?** Best in the Himalaya, central India, the northeast and
   the northwest (R² 0.74–0.78); weakest in the south (0.47) and the Indo-Gangetic
   Plain (0.54), where PM2.5 is highest and changes fastest. Best in the monsoon
   (0.81), weakest before the monsoon (0.60).
5. **Generalisation to new grid cells?** Good: all scores are from test cells
   (in blocks of about 380 km) that were never used for training.
6. **ML vs standard interpolation?** The random forest beats IDW overall (R² 0.74
   vs 0.68) and is far better where test cells are far from training cells or in
   different terrain (northeast: 0.74 vs −0.74; Himalaya: 0.78 vs 0.46). IDW is a
   little better in the northwest. On the 0.25° maps both methods keep the same
   large-scale pattern (2024 means 72.4 and 71.5 µg/m³ against 72.8 for CAMS);
   the random forest adds detail from AOD and terrain.

Details, decisions and problems are in [docs/steps/](docs/steps/00_overview.md).
