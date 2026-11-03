# Glossary

| Word | Meaning |
|---|---|
| **CAMS EAC4** | Copernicus Atmosphere Monitoring Service global reanalysis: a best estimate of air pollution and weather, every 3 hours, on a 0.75° grid. |
| **PM2.5** | Mass of particles smaller than 2.5 µm per cubic metre of air (µg/m³). |
| **AOD** | Aerosol optical depth at 550 nm: how much the particles in the whole air column dim sunlight. No unit. |
| **u10, v10** | Wind at 10 m height towards the east (u) and towards the north (v), in m/s. |
| **t2m, d2m** | Temperature and dewpoint at 2 m height, in °C. Dewpoint tells how humid the air is. |
| **sp** | Surface pressure in hPa. Low over mountains, high over plains. |
| **Grid cell** | One 0.75° × 0.75° square (about 80 km) of the CAMS map. |
| **IGP** | Indo-Gangetic Plain, the flat, crowded, very polluted north Indian plain. |
| **Training / test set** | Data the model learns from / data kept back to check it. |
| **Spatial block** | A group of 5 × 5 neighbouring grid cells kept together in train or test. |
| **IDW** | Inverse distance weighting: weighted average of nearby values, closer ones count more. |
| **Bilinear interpolation** | Smooth estimate between the 4 surrounding grid cells. |
| **Linear regression** | Model that predicts with a straight-line formula: a × x1 + b × x2 + … |
| **Random forest** | Many decision trees, each trained on a random part of the data; their answers are averaged. |
| **Feature / predictor** | An input of the model (for example AOD). |
| **Feature importance** | How much the random forest relies on each predictor. |
| **RMSE** | Root mean squared error: typical size of the error, big errors count more. Lower is better. |
| **MAE** | Mean absolute error: average size of the error. Lower is better. |
| **R²** | Share of the variation the method explains: 1 = perfect, 0 = as good as always guessing the average, below 0 = worse than that. |
| **Bias** | Average of (prediction − CAMS). Negative = too low on average. |
| **Correlation** | Number from −1 to 1 showing how strongly two things rise and fall together in a straight line. |
