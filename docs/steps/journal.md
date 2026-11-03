# Journal

Dated timeline of the work, newest at the bottom. Decisions are in **bold**.

## 2026-11-03

| What | Where |
|---|---|
| Started a beginner version of the project: plain scripts, one per step, no helper modules. **Data reused from the first project** (link to `data/raw`, region file copied), nothing downloaded. | project folder |
| Step 1: 60 files joined, **daily averages** (1,827 days), units converted, 494 India cells, 902,538 rows, no missing values. | step1_prepare_data.py |
| Step 2: IGP highest (218 µg/m³ in winter), monsoon cleanest; AOD has the strongest correlation with PM2.5 (0.56). | step2_explore.py |
| Step 3: **blocks of 5 × 5 cells, fixed diagonal pattern**; 112 test cells (23%), 12–28% per region. | step3_split.py |
| Step 4: **two ML models, linear regression and random forest**, plus IDW as the interpolation baseline. Random forest saved compressed (500 → 170 MB). | step4_train_models.py |
| Step 5: random forest R² 0.74, IDW 0.68, linear 0.45 at test cells. IDW fails in the northeast (−0.74). AOD most important. | step5_evaluate.py |
| Step 6: 0.25° maps for 2024-11-15 and the 2024 average. Fixed a problem with `interp` and the text variable `region`. 2024 India means: CAMS 72.8, bilinear 72.4, random forest 71.5 µg/m³. | step6_maps.py |
| README with results and short answers to the 6 research questions. | README.md |

**Still open:** nothing. Ready for review.
