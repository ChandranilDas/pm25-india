# EE708 Term Project: 10-Minute Presentation Script & Defense Guide

**Project:** Machine Learning for Spatial Enhancement of PM2.5 Prediction over India  
**Group:** Group 1 (Project 3)  
**Date & Slot:** Wednesday, 4 Nov 2026 | 6:15 PM – 6:35 PM  
**Total Allocated Time:** 10 minutes maximum (followed by 10 minutes Q&A demo)

---

## Team Speaking Allocation (2 minutes per speaker)

| Speaker | Slide | Core Focus | Time |
|---|---|---|---|
| **Speaker 1: Harsh Meshram** | **Slide 1: Title & Project Objective** | Motivation, CAMS EAC4 context, research questions, spatial enhancement scope | 0:00 – 2:00 |
| **Speaker 2: Chandranil Das** | **Slide 2: Data Preprocessing & Feature Engineering** | CAMS 9 base predictors, 4 engineered features, spatial autocorrelation & 5×5 block holdout split | 2:00 – 4:00 |
| **Speaker 3: Dhrubojyoti Saha** | **Slide 3: Model Architectures & Parameter Efficiency** | Baseline IDW/OLS vs Random Forest vs proposed ResTabNet (58,881 params, 773 KB, Huber loss) | 4:00 – 6:00 |
| **Speaker 4: Dineshwar Okram** | **Slide 4: Performance & Regional/Seasonal Results** | Test set metrics (R² 0.827, RMSE 26.09), Northeast triumph (R² 0.794 vs IDW -0.738), regional/seasonal trends | 6:00 – 8:00 |
| **Speaker 5: Tanuj Singh** | **Slide 5: Spatial Enhancement Mapping & Conclusions** | 0.25° super-resolution downscaling, mass conservation, takeaways & answers to RQs | 8:00 – 10:00 |

---

## Detailed Slide-by-Slide Script

### Slide 1: Title & Project Objective
**Speaker:** Harsh Meshram (0:00 – 2:00)

> "Good evening respected professors and classmates. We are Group 1, and today we present our EE708 term project: *Machine Learning for Spatial Enhancement of PM2.5 Prediction over India*. My teammates are Chandranil Das, Dhrubojyoti Saha, Dineshwar Okram, and Tanuj Singh.
>
> Ambient particulate matter below 2.5 micrometers—PM2.5—is the single greatest environmental health hazard in India, contributing to over 1.6 million premature deaths annually. Global reanalysis systems like Copernicus Atmosphere Monitoring Service (CAMS) EAC4 simulate chemical transport and assimilate satellite aerosol optical depth, providing continuous daily estimates. However, CAMS operates at a coarse horizontal resolution of 0.75 degrees—roughly 80 kilometers. At this scale, sharp urban plumes, valley accumulation, and localized exposure peaks are completely smoothed out.
>
> In this project, we address two coupled challenges: first, predicting surface PM2.5 from atmospheric and aerosol predictors; and second, spatially downscaling coarse reanalysis fields to a fine 0.25-degree grid—roughly 28 kilometers. 
>
> We set out to answer six fundamental research questions: How accurately can ground PM2.5 be predicted? How do non-linear ML models compare against standard spatial interpolation? Which atmospheric variables dominate? How do errors vary across India's diverse climate zones and seasons? How honestly can our models generalize to unseen geographic cells? And finally, how parameter-efficient can we make these deep learning architectures for practical deployment?
>
> I now hand over to Chandranil to walk us through our dataset, feature engineering, and spatial validation strategy."

---

### Slide 2: Data Preprocessing, Feature Engineering & Spatial Partitioning
**Speaker:** Chandranil Das (2:00 – 4:00)

> "Thank you, Harsh. To build a robust model, we ingested 5 consecutive years of CAMS EAC4 daily reanalysis data from 2020 through 2024. Covering the entire Indian landmass from 8 to 35 degrees North, this comprises 494 land grid cells and 902,538 clean daily records with zero missing values.
>
> Beyond the 9 standard CAMS variables (AOD, winds, temperatures, surface pressure, coordinates, and month), we engineered four physically-grounded atmospheric features:
> 1. **Scalar Wind Speed ($ws = \sqrt{u_{10}^2 + v_{10}^2}$)**: Directly captures atmospheric ventilation and stagnation.
> 2. **Dewpoint Depression ($t_{2m} - d_{2m}$)**: A proxy for relative humidity, signaling fog formation and hygroscopic particle growth.
> 3. **Cyclical Calendar Encoding**: $\sin$ and $\cos$ of the annual cycle, removing artificial discontinuity between December and January.
> 4. **Aerosol Stagnation Index ($AOD / (ws + 0.5)$)**: Explicitly captures high aerosol burdens trapped under calm, stagnant winds.
>
> A fundamental challenge in geospatial machine learning is **spatial autocorrelation**. If one performs a naive random train-test split across table rows, neighboring grid cells on the exact same date will be split between train and test. Because they share virtually identical synoptic weather, the test set leaks into the training set, producing deceptively high R² scores that fail in the real world.
>
> To ensure honest generalization, we designed a **spatial block-holdout scheme**. We partitioned India into 5-by-5 grid cell blocks—roughly 380 by 380 kilometers. Using a fixed diagonal pattern, we held out 112 grid cells—23% of all cells—across all 6 geographic regions. Our models are trained on 382 cells (697,914 samples) and tested strictly on the 112 unseen cells (204,624 samples). Furthermore, feature standardization statistics were computed strictly on the training distribution.
>
> I now pass the mic to Dhrubojyoti to present our model architectures."

---

### Slide 3: Model Architectures & Parameter Efficiency
**Speaker:** Dhrubojyoti Saha (4:00 – 6:00)

> "Thank you, Chandranil. We evaluated four distinct model paradigms to benchmark statistical interpolation, classical machine learning, and deep neural architectures.
>
> First, as our spatial baseline, we implemented **Inverse Distance Weighting (IDW)**. For each test cell, PM2.5 is interpolated using the 8 nearest training cells weighted by inverse squared distance. Crucially, IDW uses zero meteorological data.
>
> Second, we implemented **Multiple Linear Regression (OLS)**, which fits 14 parameters across our predictors. Linear regression fails to capture non-linear aerosol-thermodynamic dynamics, yielding an R² of only 0.446.
>
> Third, we trained an ensemble **Random Forest Regressor** with 100 trees. While it achieves R² = 0.742, it produces a massive 166-megabyte serialized binary containing over 1.5 million tree nodes—making it cumbersome for edge deployment or resource-constrained monitoring stations.
>
> To solve this, we engineered our proposed **Residual Tabular Neural Network (ResTabNet)**, serialized as `pm25_model.h5`.
>
> ResTabNet uses a deep residual architecture with skip connections, Batch Normalization, and Dropout:
> - A 128-unit stem block with Batch Normalization;
> - Residual Block 1: 128-unit dense layers with identity addition;
> - Residual Block 2: 64-unit dense layers with projection addition;
> - A 32-unit regressor head trained with **Huber loss** to protect gradient descent from volatile winter smog outliers.
>
> In terms of parameter efficiency, ResTabNet has **only 58,881 parameters** and occupies **only 773 kilobytes** in `.h5` format. This delivers a **215-fold storage reduction** compared to the 166 MB Random Forest, with inference latency under 1 millisecond per 1,000 samples.
>
> Now, Dineshwar will present our validation performance."

---

### Slide 4: Performance & Regional/Seasonal Results
**Speaker:** Dineshwar Okram (6:00 – 8:00)

> "Thank you, Dhrubojyoti. Let us examine the performance across the 204,624 held-out spatial test samples.
>
> As shown in Table I:
> - Linear regression underperforms with R² = 0.446 and RMSE = 46.72 µg/m³.
> - The IDW baseline achieves R² = 0.683 and RMSE = 35.34 µg/m³.
> - The 166 MB Random Forest achieves R² = 0.742 and RMSE = 31.89 µg/m³.
> - Our compact ResTabNet achieves an outstanding **R² of 0.8274, an RMSE of 26.09 µg/m³, and an MAE of 14.84 µg/m³**! It significantly outperforms Random Forest while utilizing 215 times less storage!
>
> Looking at regional dynamics:
> - Accuracy is highest in the **Himalayas (R² = 0.844, RMSE = 18.39 µg/m³)**, where surface pressure and AOD clearly separate alpine air from lowland haze.
> - The most dramatic finding is in the **Northeast**: IDW completely collapses with an **R² of negative 0.738**, because it is geographically separated from training cells by Bangladesh and mountain ridges. In contrast, ResTabNet relies on physical atmospheric state and achieves a stellar **R² of 0.794**!
> - In the **Indo-Gangetic Plain**, ResTabNet achieved **R² = 0.724** (jumping from 0.551 in basic models), with RMSE dropping from 58.1 to 45.59 µg/m³ despite severe winter episodic spikes exceeding 400 µg/m³.
>
> Seasonally, the **Monsoon achieves R² = 0.843**, while Winter and Post-Monsoon reach **R² = 0.819 and 0.814** thanks to our ventilation index and Huber loss.
>
> I now hand over to Tanuj to discuss high-resolution spatial downscaling and our conclusions."

---

### Slide 5: Spatial Enhancement Mapping & Key Conclusions
**Speaker:** Tanuj Singh (8:00 – 10:00)

> "Thank you, Dineshwar. To demonstrate spatial enhancement, we downscaled the CAMS PM2.5 fields from 0.75 degrees (~80 km) to 0.25 degrees (~28 km)—a 9-fold increase in spatial grid density.
>
> In our downscaling pipeline, we bilinearly interpolate the continuous atmospheric predictors (AOD, temperature, surface pressure, wind, humidity) to the dense 0.25-degree grid over India, and then feed them through our trained ResTabNet model.
>
> We verified physical conservation for the entire year 2024:
> - The original CAMS 0.75-degree continental mean is 72.8 µg/m³;
> - The bilinear 0.25-degree mean is 72.4 µg/m³;
> - Our ML-downscaled 0.25-degree mean is 71.5 µg/m³, with a mean absolute difference of only 5.4 µg/m³.
>
> Crucially, while standard bilinear interpolation merely blurs the square grid boundaries, our ML downscaling injects authentic physical topography: it resolves sharp smog entrapment along the Himalayan foothills, the Western Ghats rain shadow, and river basin accumulation.
>
> In conclusion, our findings answer all six research questions:
> 1. Ground PM2.5 can be reliably predicted from reanalysis with R² = 0.827 and MAE = 14.84 µg/m³ at unobserved locations.
> 2. ResTabNet substantially outperforms linear regression, IDW interpolation, and Random Forest.
> 3. ResTabNet (`pm25_model.h5`) delivers a 215-fold storage reduction (773 KB vs 166 MB) with just 58,881 parameters.
> 4. AOD, the ventilation stagnation index, and surface pressure are the predominant physical predictors.
> 5. Honest 5×5 spatial block validation guarantees real-world generalization.
> 6. ML spatial downscaling produces realistic high-resolution maps preserving continental mass conservation.
>
> All submission files—`Code.ipynb`, `pm25_model.h5`, `Report.pdf`, and `evaluate.py`—are ready. We welcome any questions. Thank you!"

---

## Anticipated Defense & Viva Questions

### Q1: Why did you use spatial block cross-validation instead of standard random k-fold?
**Answer (Chandranil Das / Dineshwar Okram):**
"In spatial and atmospheric data, Tobler's First Law of Geography applies: nearby locations are strongly correlated. If we split data randomly by rows, test cells on day $t$ will have neighboring training cells 50 km away on day $t$. Since weather is identical, the model simply memorizes the weather of that day, resulting in severe data leakage. By using 5×5 cell blocks (~380 km), test cells are geographically distant from training cells, proving that the model learns physical relationships rather than memorizing spatial coordinates."

### Q2: Why is the ResTabNet model preferred over Random Forest?
**Answer (Dhrubojyoti Saha / Harsh Meshram):**
"Both predictive power and parameter efficiency. Random Forest requires 166 MB of disk space and contains over 1.5 million nodes, yet only reaches R² = 0.742. Our ResTabNet model incorporates residual skip connections and Huber loss, achieving R² = 0.827 (a massive 8.5 percentage point increase in explained variance) while requiring only 58,881 parameters and 773 KB on disk—a 215-fold compression that makes it directly deployable on embedded IoT and edge microcontrollers."

### Q3: Why does Inverse Distance Weighting fail so badly in the Northeast (R² = -0.74)?
**Answer (Dineshwar Okram / Tanuj Singh):**
"IDW relies purely on Euclidean distance. Because the Indian land boundary around the Northeast is geographically constrained by Bangladesh and mountain ridges, the nearest training cells to the Northeast test cells are located hundreds of kilometers away in the flat plains of Bihar and West Bengal. Those plains have very high winter pollution, whereas the Northeast hills have much cleaner air. IDW blindly transfers high pollution values to the hills, resulting in a negative R². Our ML models use local AOD, surface pressure, and temperature, achieving R² = 0.794."

### Q4: How does your evaluation script work?
**Answer (Chandranil Das):**
"Our evaluation script `evaluate.py` accepts the dataset path as a CLI argument (`--data data/pm25_daily.csv`), loads the trained `pm25_model.h5`, prints the model architecture and summary via `model.summary()`, reports the parameter efficiency count (58,881 parameters, 773 KB), and computes RMSE, MAE, R², and Bias overall, as well as broken down across all 6 regions and 4 seasons."
