# Overview of the steps

One document per step in this folder. Every step document has the same sections:
Status, Purpose, Idea in plain words, How it works, Outputs, How to run,
Decisions, Problems met, Results, Open issues.
**[journal.md](journal.md)** is the dated timeline of what was done and decided.
**[glossary.md](glossary.md)** explains the technical words.

```
step 1 Prepare data -> step 2 Explore
                    -> step 3 Split -> step 4 Train models -> step 5 Evaluate
                                                           -> step 6 Maps
```

## Status

| Step | Status | Doc | Research questions |
|---|---|---|---|
| 1 Prepare data | **Done**: 1,827 days, 494 India cells, no missing values | [step1_prepare_data.md](step1_prepare_data.md) | needed for all |
| 2 Explore | **Done**: IGP worst in winter; AOD most related to PM2.5 | [step2_explore.md](step2_explore.md) | RQ3, RQ4 |
| 3 Split | **Done**: 112 test cells (23%), 12–28% in every region | [step3_split.md](step3_split.md) | RQ5 |
| 4 Train models | **Done**: IDW, linear regression, random forest | [step4_train_models.md](step4_train_models.md) | RQ1, RQ2, RQ6 |
| 5 Evaluate | **Done**: random forest R² 0.74, IDW 0.68, linear 0.45 | [step5_evaluate.md](step5_evaluate.md) | RQ1–RQ5 |
| 6 Maps | **Done**: 0.25° maps, random forest vs bilinear | [step6_maps.md](step6_maps.md) | RQ6 |

## Research questions (from docs/PROJECT.md) and where they are answered

| Question | Answered in |
|---|---|
| 1. How accurately can PM2.5 be estimated from CAMS variables? | step 5 |
| 2. How do different ML models compare? | step 5 |
| 3. Which features are most important? | step 2 (correlation), step 5 (importance) |
| 4. How does performance vary across regions and seasons? | step 5 |
| 5. How well does the model work at grid cells not used in training? | steps 3 and 5 |
| 6. How does ML-based enhancement compare with standard interpolation? | steps 5 and 6 |

The short answers are in the [README](../../README.md).
