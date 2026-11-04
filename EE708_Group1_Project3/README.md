# EE708 Group 1, Project 3: submission

## Contents
- `Code.ipynb`: executed analysis, model-loading/retraining option, baselines, training/held-out results and efficiency checks.
- `Model.h5`: genuine HDF5 numeric random forest, not a Keras model or renamed joblib.
- `model_io.py`: required HDF5 loader/exporter.
- `evaluate.py`: dataset-path evaluation entry point; prints the estimator, summary and metrics.
- `Report.pdf`: two-page IEEE conference report, references included.
- `Presentation.pptx`: five editable slides with speaker notes, paced for approximately nine minutes plus transitions.
- `Speaker_notes.txt`: rehearsal notes, timing and evidence sources.
- `requirements.txt`: pinned versions used for verification.
- `data/`: daily processed CSV, spatial split, daily NetCDF and region NetCDF.
- `src/`: original six Python scripts, preserved for traceability.

## Setup on macOS
From the extracted folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py --dataset data/pm25_daily.csv --cells data/cells.csv
```

The loader requires scikit-learn 1.8.0, the version used to export and verify HDF5. Install the pinned requirements in this separate environment. The original uploaded joblib identifies scikit-learn 1.9.1; it loaded with a version warning in the review environment. After export, every HDF5 test prediction matched the supplied results within 2.3e-13. No joblib deserialization is required by this submission.

Open `Code.ipynb` in Jupyter/VS Code using the same environment. If needed, install Jupyter separately with `python -m pip install jupyterlab`, then run `jupyter lab`.

## Evaluation contract
`--dataset` accepts a processed CSV containing `pm25` (micrograms/m3), `aod`, `u10`, `v10`, `t2m` and `d2m` (Celsius), `sp` (hPa), `lat`, `lon`, `month`. Membership must be supplied as a `set` column with train/test labels or through `--cells`. A full original table produces both training and held-out results. Membership labels are user-supplied; the script does not establish the independence of an arbitrary new dataset. It writes `evaluation_metrics.json` by default.

Expected held-out results: RMSE 31.888101, MAE 17.818622, R2 0.742123, bias -5.583709. Training: RMSE 18.845497, MAE 8.474647, R2 0.914790. Runtime varies with hardware.

## Model format
The course requests .h5, .pth or .pt. `Model.h5` is a real HDF5 file storing each trained tree's nodes and leaf values, plus estimator metadata. It is loaded by `model_io.load_model`, not `keras.models.load_model`. If the instructor's evaluator specifically assumes a neural-network/Keras loader, this custom random-forest loader must be accepted or the format expectation clarified. Merely changing an extension is not valid, and no such renaming was used.

## Reproducibility and scope
The notebook starts from the included processed table and regenerates IDW, linear regression and forest inference. `RETRAIN=True` fits a new forest using training cells only and writes `Model_retrained.h5`; the submitted model is preserved. Original raw monthly NetCDF inputs are not included. The daily and region NetCDF files are included. To rerun original `src/` step 1, restore the monthly raw data. Original steps 5 and 6 expect the original joblib path; the primary submission instead uses the HDF5 loader. Use the executed notebook and evaluate.py for the supported submission workflow.

Forest complexity is 6,886,686 nodes and 3,443,393 leaves (100 trees, maximum depth 39). It is not claimed to be parameter-efficient relative to compact alternatives. No new model selection has used the held-out results. The validation assessment is the single spatial test split, not a separate tuning set. Fine maps have no independent 0.25-degree validation, and all targets are CAMS rather than measurements.

The IEEE report source and class file are included. Compile from this folder with `pdflatex Report.tex` twice. Review author details and follow your institution's requirements for acknowledging assistance and individual contributions before submission. No plagiarism-check result is claimed.
