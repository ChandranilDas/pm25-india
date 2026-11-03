# Presentation guide: slide by slide

This guide goes with `results/presentation/pm25_india_presentation.key` (and the
`.pptx` copy). For each of the 14 slides it gives:

- **Key points**: what the slide shows.
- **The idea**: the concept behind it, in plain words.
- **Why we did it this way**: the reason for the step or choice.
- **What to say**: a short script (about 1.5–2 minutes per slide).
- **Likely questions**: questions a teacher may ask, with answers.

Numbers come from the step documents in `docs/steps/` and the result tables in
`results/`. Total talk time: about 20–25 minutes.

---

## The workflow at a glance

```
CAMS files ──► 1 Prepare ──► 2 Explore
                  │
                  └────────► 3 Split ──► 4 Train models ──► 5 Evaluate
                                               │
                                               └──────────► 6 Maps
```

| Step | Script | Question it serves | One-line reason |
|---|---|---|---|
| 1 Prepare data | `step1_prepare_data.py` | all | one clean daily table in normal units |
| 2 Explore | `step2_explore.py` | RQ3, RQ4 | know the data before modelling |
| 3 Split | `step3_split.py` | RQ5 | test honestly at places the model never saw |
| 4 Train models | `step4_train_models.py` | RQ1, RQ2, RQ6 | compare ML with standard interpolation |
| 5 Evaluate | `step5_evaluate.py` | RQ1–RQ5 | measure errors overall, by region and season, and which features matter |
| 6 Maps | `step6_maps.py` | RQ6 | turn the model into finer maps |

---

## Slide 1: Title

**Key points**
- Project: Machine Learning for Spatial Enhancement of PM2.5 Prediction over India.
- Data: CAMS EAC4 reanalysis, India, 2020–2024.

**The idea**
PM2.5 means tiny particles (smaller than 2.5 micrometres) in the air. They go deep
into the lungs, so they are the main air-quality concern in India. "Spatial
enhancement" means making the PM2.5 picture sharper or more detailed in space:
estimating PM2.5 at places where we do not have it, and at a finer grid.

**Why this project**
Ground monitoring stations in India are few and uneven, and model data like CAMS
is coarse (squares of about 80 km). If other variables that are easier to get
(aerosol optical depth from satellites, weather) can tell us PM2.5, we can fill
the gaps and draw finer maps.

**What to say**
"Our question is simple: can we estimate PM2.5 over India from weather and aerosol
data, at places the model has never seen, and use that to make finer maps?"

---

## Slide 2: Six research questions

**Key points**
1. Accuracy: how well can PM2.5 be estimated?
2. Models: how do the ML models compare?
3. Features: which inputs matter most?
4. Regions and seasons: where and when does it work well or badly?
5. New locations: how well does it work at grid cells not used in training?
6. ML vs interpolation: is ML better than the standard method?

**The idea**
The six questions come from the project brief (`docs/PROJECT.md`). The whole
project is built so that each step answers one or more of them, and the last
slide answers each in one line.

**Why we show them first**
The audience then knows what to listen for. Every later slide links back to one
of these questions.

**What to say**
"These six questions come from the brief. I will answer them one by one, and the
last slide puts all six answers together."

---

## Slide 3: Data

**Key points**
- 494 grid cells inside India (0.75° squares, about 80 km).
- 1,827 days (2020–2024), daily averages of 3-hourly data.
- About 0.9 million rows (one row = one cell on one day), no missing values.
- Target: PM2.5 in µg/m³.
- Features: AOD, wind u and v, temperature, dewpoint, surface pressure, latitude,
  longitude, month.
- Map: average PM2.5 over the 5 years.

**The idea**
- **CAMS EAC4** is a "reanalysis": a computer model of the atmosphere that is
  corrected with satellite observations. It gives a complete, gap-free record on a
  regular grid.
- **AOD (aerosol optical depth)** measures how much the particles in the whole air
  column dim sunlight. More particles means higher AOD.
- **Wind (u = east-west, v = north-south)** can blow pollution away or bring it in.
- **Temperature and dewpoint**: dewpoint shows humidity; cold, still, humid air
  in winter traps pollution near the ground.
- **Surface pressure** mainly tells the altitude: low in the mountains, high in
  the plains.
- **Latitude, longitude, month** tell the model where and when.

**Why we did it this way (step 1)**
- **We used the predictors listed in the brief**, plus month, so the model can
  learn the yearly cycle.
- **We made daily averages** of the 8 values per day. This cuts the data from 7.2
  million to 0.9 million rows, so everything runs in minutes on a laptop. Daily
  PM2.5 is also what air-quality limits use (for example the Indian daily limit of
  60 µg/m³).
- **We converted units** to the usual ones (kg/m³ → µg/m³, K → °C, Pa → hPa)
  so numbers are easy to read.
- **We kept only cells inside India** (at least half of the square in India),
  because the project is about India.
- **We reused the downloaded files** from the first version of the project; no new
  download was needed.

**What to say**
"We used CAMS EAC4, a reanalysis that combines a model with satellite data. We
averaged it to daily values and kept the 494 grid cells in India. The target is
PM2.5; the inputs are the variables named in the brief plus the month."

**Likely questions**
- *Why not use ground stations?* The brief asks to use CAMS. Stations are also few
  and have gaps; CAMS is complete, which makes a clean comparison possible.
- *Why daily and not 3-hourly?* Faster, simpler, and daily is the standard
  air-quality time scale. The cost: we do not study the day–night cycle (see
  limitations).

---

## Slide 4: PM2.5 patterns

**Key points**
- The IGP (Indo-Gangetic Plain) is by far the most polluted: 218 µg/m³ average in
  winter, 2–4 times other regions.
- The monsoon is the cleanest season everywhere (rain washes particles out).
- The northeast peaks before the monsoon (fire season).
- AOD has the strongest link with PM2.5 (correlation 0.56); surface pressure is
  next (0.32); wind and temperature have almost no straight-line link.

**The idea (step 2)**
Exploring means looking at simple averages and plots before building any model.
The correlation is a number from −1 to 1 that shows how strongly two things rise
and fall together in a straight line.

**Why we did this step**
- To check the data looks sensible (known pollution hotspots and seasons appear).
- To get a first answer to RQ3 (which inputs are linked to PM2.5) and RQ4 (regional
  and seasonal differences).
- To know what to expect: very different regions and seasons mean the model has to
  handle very different conditions, and the evaluation must be split by region and
  season.

**What to say**
"Before modelling we looked at the data. The Indo-Gangetic Plain in winter stands
out, with more than 200 µg/m³. The monsoon is clean everywhere. AOD is the
variable most linked to PM2.5."

**Likely questions**
- *Why is the IGP so polluted?* Dense population, traffic, industry, crop-residue
  burning after harvest, and in winter cold, calm air traps the pollution in a
  shallow layer near the ground. The Himalaya also blocks air from leaving.
- *Wind has almost zero correlation, so is it useless?* Not necessarily: the link
  may be curved or depend on other variables. A random forest can use such links;
  correlation only sees straight lines.

---

## Slide 5: Workflow

**Key points**
Six steps, one short Python script each:
1. Prepare data: 60 monthly files → daily table.
2. Explore: averages, correlations, maps.
3. Split: hold out blocks of 5 × 5 cells.
4. Train models: IDW, linear regression, random forest.
5. Evaluate: RMSE, MAE, R², bias; feature importance.
6. Maps: 0.25° maps, interpolation vs random forest.

**The idea**
A pipeline: each step reads what the step before saved and writes its own results
to `results/stepN/`. Anyone can rerun the whole project in order in about five
minutes.

**Why we organised it this way**
- **One script per step** keeps each part short and easy to check.
- **Fixed settings** (fixed split pattern, `random_state=42`) make the results
  repeatable: running again gives the same numbers.
- **Each step has its own document** in `docs/steps/` with its purpose, decisions,
  problems and results, and a journal records what was done when.

**What to say**
"The project is six short scripts that run in order. Each step saves its results
and has a document explaining what it does and why."

---

## Slide 6: Split (testing at new places)

**Key points**
- Neighbouring cells have almost the same PM2.5, so randomly chosen test cells
  would make the test too easy.
- We cut India into blocks of 5 × 5 cells (about 380 km). Every 5th block, in a
  diagonal pattern, goes to the test set.
- 112 test cells, 382 training cells (23% test). Every region has 12–28% of its
  cells in the test set.

**The idea (step 3)**
To know how well a model works at new places, we must test it on places it never
saw during training. The test cells play the role of "places without data": we
hide their PM2.5, predict it, and then compare with the hidden true value.

**Why we did it this way**
- **The brief requires it**: evaluation must use held-out grid locations (RQ5).
- **Blocks instead of single cells**: if one cell is in the test set but its
  neighbours are in training, the model (and especially interpolation) can almost
  copy the neighbour's value. Scores would look better than they really are. Whole
  blocks force a real gap: cells in the middle of a test block are more than
  200 km from the nearest training cell.
- **A fixed diagonal pattern instead of random blocks**: it spreads the test blocks
  over the whole country, needs no random seed and is easy to explain. We checked a
  few patterns and kept the one where every region has 10–30% of its cells in test.
  Some other patterns put almost half of the Himalaya, or none of it, in the test
  set, which would make the region results unfair or impossible.
- **All days of a cell go to the same set**: the model never sees any day of a
  test cell.

**What to say**
"This is the most important design choice. The models never see the red cells in
training. So when we score them on the red cells, we are measuring how well they
work at new places, which is exactly question 5."

**Likely questions**
- *Why not a random 80/20 split of rows?* The same cell would be in both training
  and test on different days, and neighbours would be in training. The score would
  measure memory, not prediction at new places.
- *Why 5 × 5?* Big enough to stop copying from neighbours, small enough that we
  still have test blocks in every region.

---

## Slide 7: Three methods

**Key points**
- **IDW interpolation** (standard method): weighted mean of the PM2.5 of the 8
  nearest training cells; weight = 1 / distance².
- **Linear regression** (ML model 1): a straight-line formula,
  PM2.5 = a × AOD + b × wind + … + constant.
- **Random forest** (ML model 2): the average of 100 decision trees; it can learn
  curved links and combinations.
- Both ML models: scikit-learn, trained on the 698,000 rows of the training cells,
  random seed 42.

**The idea (step 4)**
- **IDW** uses only PM2.5 at nearby known places, nothing else. It assumes nearby
  places are similar. Closer cells count more: a cell twice as close gets four
  times the weight.
- **Linear regression** looks only at the inputs at the cell itself (AOD, weather,
  location, month) and fits the best straight-line formula.
- **Random forest** also looks only at the inputs at the cell itself. Each tree asks
  a series of yes/no questions ("Is AOD above 0.6? Is pressure above 950 hPa?")
  and ends in a predicted value. Each tree is trained on a random part of the data,
  and averaging 100 trees gives a stable answer.

**Why these three methods**
- **IDW is the standard spatial interpolation method**, so it is the baseline RQ6
  asks us to compare with. If ML cannot beat it, ML adds nothing.
- **Linear regression is the simplest ML model.** It shows how far a straight-line
  formula gets.
- **Random forest is a standard non-linear model** that works well on tables of
  numbers without much tuning. Comparing it with linear regression answers RQ2.
- **Only two ML models**, both from scikit-learn, keep the project simple and easy
  to explain.
- **Settings**: 100 trees; at least 10 rows per leaf (stops the trees from learning
  noise); fixed seed for repeatable results. IDW with 8 neighbours and power 2 is
  the usual textbook choice.

**What to say**
"IDW only looks at PM2.5 in nearby cells. The ML models only look at the weather
and aerosol at the cell itself. This lets us compare the two ideas directly."

**Likely questions**
- *Why not deep learning?* The data is a table with 9 inputs; tree models are the
  usual best choice for such data, and they are faster and easier to explain.
- *Does the random forest see PM2.5 of the test cells?* No. It only sees the inputs
  there. PM2.5 of test cells is used only to score the predictions.

---

## Slide 8: Main result

**Key points**

| Method | RMSE | MAE | R² | Bias |
|---|---|---|---|---|
| IDW | 35.3 | 20.4 | 0.68 | −6.8 |
| Linear regression | 46.7 | 32.1 | 0.45 | −12.9 |
| Random forest | **31.9** | **17.8** | **0.74** | −5.6 |

(RMSE, MAE and bias in µg/m³, at the 112 test cells.)

- Random forest explains 74% of the daily variation at new places.
- It is better than IDW interpolation (R² 0.68).
- Linear regression is far behind: the links are not straight lines.

**The idea (step 5): the four scores**
- **RMSE**: typical size of the error; big errors count extra. Lower is better.
- **MAE**: average size of the error. Lower is better.
- **R²**: share of the day-to-day and place-to-place variation the method explains.
  1 = perfect, 0 = no better than always guessing the average, below 0 = worse
  than that.
- **Bias**: average of (prediction − true value). Negative means the method
  predicts too low on average.

**Why we use these scores**
The brief asks for an accuracy assessment, and these four are the standard ones.
They answer different questions: how big are the errors (RMSE, MAE), how much of
the pattern is captured (R²), and is there a constant over- or under-estimate (bias).

**What to say**
"This answers questions 1 and 2. At grid cells it never saw, the random forest has
a typical error of 18 µg/m³ and explains 74% of the variation. It beats IDW, and
linear regression is clearly the weakest."

**Likely questions**
- *Why are all biases negative?* All methods under-predict the most extreme
  pollution days, which pulls the average down.
- *Is 0.74 good?* For predicting at places 100+ km from any training cell, using
  only weather and aerosol inputs, yes. It also beats the standard method.

---

## Slide 9: By region

**Key points**

| Region | IDW | Linear | Random forest |
|---|---|---|---|
| IGP | 0.56 | 0.29 | 0.54 |
| Northwest | 0.80 | −0.05 | 0.75 |
| Central | 0.70 | 0.44 | 0.75 |
| South | 0.45 | −1.80 | 0.47 |
| Northeast | −0.74 | 0.15 | 0.74 |
| Himalaya | 0.46 | 0.49 | 0.78 |

(R² at test cells.)

- In the northeast and the Himalaya, IDW fails or is weak, while the random forest
  does well (0.74–0.78).
- The IGP and the south are hard for every method (R² about 0.5).
- IDW is slightly better in the northwest, where the land is flat and similar.

**The idea**
IDW assumes "nearby means similar". That fails where the nearest known cells are
in a different setting: across a mountain edge or in another region. The random
forest does not need neighbours; it uses the conditions at the cell itself (AOD,
pressure which tells altitude, and so on).

**Why we split results by region**
RQ4 asks where the method works. A single overall number can hide regions where a
method fails badly, such as IDW in the northeast.

**What to say**
"By region the picture is more interesting. Where neighbouring cells are different,
in the northeast and the Himalaya, interpolation fails and the random forest is
much better. The IGP is hard for everyone because PM2.5 is very high and changes
fast."

**Likely questions**
- *Why is linear regression so bad in the south (−1.8)?* One straight-line formula
  for all of India is pulled by the very polluted north. Applied to the cleaner
  south, it predicts far too high. A negative R² means worse than guessing the
  average.

---

## Slide 10: By season

**Key points**

| Season | IDW | Linear | Random forest |
|---|---|---|---|
| Winter | 0.61 | 0.37 | 0.69 |
| Pre-monsoon | 0.54 | 0.20 | 0.60 |
| Monsoon | 0.78 | 0.36 | 0.81 |
| Post-monsoon | 0.64 | 0.44 | 0.71 |

- The random forest is the best in every season.
- Best in the monsoon, weakest before the monsoon.
- Scatter plot: predictions follow the 1:1 line but stay too low at the highest
  values.

**The idea**
Seasons follow the India Meteorological Department definition: winter (Jan–Feb),
pre-monsoon (Mar–May), monsoon (Jun–Sep), post-monsoon (Oct–Dec). In the monsoon
PM2.5 is low and smooth, so it is easier to predict. Before the monsoon, dust storms
and fires cause sudden local peaks that are hard to predict.

**Why we split results by season**
RQ4 asks about seasons too, and step 2 showed PM2.5 changes a lot through the year.

**What to say**
"By season, the random forest wins every time. It is best in the monsoon and
weakest before the monsoon, when dust and fires create local peaks. The scatter
plot shows it is close to CAMS, but under-predicts the very worst days."

---

## Slide 11: Feature importance

**Key points**
- AOD 0.31, surface pressure 0.19, temperature 0.14, dewpoint 0.10, latitude 0.09,
  longitude 0.08, wind u 0.04, wind v 0.03, month 0.03 (they add up to 1).
- AOD tells how many particles are in the air.
- Surface pressure tells the altitude (mountains vs plains).
- Wind matters least.

**The idea**
A random forest keeps track of how much each input helps its trees split the data
into better groups. This "built-in importance" shows which inputs the model relies
on most.

**Why we did it this way**
- RQ3 asks which features are most important.
- The built-in importance is one line of code and easy to explain.
- It can favour inputs with many different values, so we cross-checked it with the
  correlations from step 2. Both put AOD first and surface pressure second, so the
  answer is consistent.

**What to say**
"For question 3: AOD is clearly the most important input, which makes sense because
it directly measures particles in the air. Surface pressure is second because it
tells the model whether a cell is in the mountains or the plains. Wind matters
least."

**Likely questions**
- *Is AOD's importance partly artificial?* Partly. In CAMS, AOD and PM2.5 come from
  the same aerosol model, so they are linked by construction (see limitations).

---

## Slide 12: Finer maps (0.25°)

**Key points**
- Three maps of the 2024 average: CAMS 0.75°, bilinear interpolation 0.25°, random
  forest 0.25°.
- Bilinear: smooths CAMS PM2.5 itself between grid cells.
- Random forest: interpolate the inputs (AOD, weather, pressure) to each 0.25° point,
  then predict PM2.5 there.
- 2024 mean over India: CAMS 72.8, bilinear 72.4, random forest 71.5 µg/m³.
  The two fine maps differ by 5.4 µg/m³ on average.

**The idea (step 6)**
A 0.25° grid has 3 × 3 = 9 points inside each CAMS square. Bilinear interpolation
fills them with a smooth blend of the 4 surrounding CAMS values. The ML way fills
them with a model prediction based on local conditions, so the map can follow
changes in AOD and terrain inside the square.

**Why we did this step**
- The brief asks to combine the trained model with spatial interpolation to make
  finer maps (RQ6).
- We compare with bilinear interpolation because it is the standard way to refine
  a grid.
- One example day (2024-11-15) shows a smoggy day; the 2024 average shows the
  overall pattern.

**What to say**
"Both fine maps keep the CAMS pattern, with the IGP high and the south and the
mountains low. The random forest map shows sharper edges where conditions change
quickly, for example at the foot of the Himalaya. The India-wide means stay within
about 1 µg/m³ of CAMS, so the model does not shift the overall level."

**Likely questions**
- *How do you know the 0.25° map is right?* We cannot check it directly: there is no
  finer true PM2.5 in CAMS. The accuracy check is the test-cell result (slide 8),
  which shows the model works at places it has not seen.

---

## Slide 13: Limitations

**Key points**
- **CAMS is not ground truth**: scores show agreement with the CAMS model, not with
  measured air quality.
- **PM2.5 and AOD come from the same model**: part of AOD's usefulness is built into
  CAMS.
- **No truth below 0.75°**: the fine maps can be shown but not checked.
- **Daily averages**: we do not study the day–night cycle.

**Why we show limitations**
Every result is only as good as its data and design. Stating the limits shows what
the results can and cannot claim, and points to what to do next (for example, test
against ground stations from CPCB, India's Central Pollution Control Board).

**What to say**
"These limits apply to every result. The main one: we compare with CAMS, not with
measurements. A next step would be to check the maps against ground stations."

---

## Slide 14: Conclusions

| Question | Answer |
|---|---|
| RQ1 Accuracy | Daily PM2.5 at new places: R² 0.74, typical error 18 µg/m³ |
| RQ2 Models | Random forest much better than linear regression (0.74 vs 0.45) |
| RQ3 Features | AOD most important, then surface pressure; wind least |
| RQ4 Regions/seasons | Best: Himalaya, central India, monsoon. Worst: south, IGP, pre-monsoon |
| RQ5 New locations | Works at grid cells never used in training (blocks of about 380 km) |
| RQ6 ML vs interpolation | Random forest beats IDW; far better in the northeast and the Himalaya |

**What to say**
"To sum up: a random forest using aerosol and weather data estimates PM2.5 at new
places better than standard interpolation, especially where neighbouring cells
are not similar. Thank you; I'm happy to take questions."

---

## Quick answers for the question round

| Question | Short answer |
|---|---|
| What is held-out evaluation? | Hide some grid cells during training, predict them, and compare with their true values. |
| Why blocks? | Single random cells have training neighbours next to them; that makes the test too easy. |
| Why random forest? | Handles curved links and combinations, works well on tables, little tuning. |
| Why is IDW the baseline? | It is the standard spatial interpolation method, and RQ6 asks for this comparison. |
| What does R² = 0.74 mean? | The model explains 74% of the variation in daily PM2.5 at the test cells. |
| Why negative R²? | The method is worse than always predicting the average. |
| What would you do next? | Validate against ground stations, add more years or features, and try other models such as gradient boosting. |
| Can it be repeated? | Yes: fixed split pattern, fixed seed; run the 6 scripts in order. |
