# Changelog

## 2026-10-19

### Fixed
- API predictions were made on unscaled input. The fitted scalers are now
  saved during training and applied at inference (`transform`, not
  `fit_transform`).
- Credit categories were encoded with a hand-written map that didn't match
  the LabelEncoder order used in training (Salaried was read as Business).
  The saved encoders are used instead.
- Imputer and scaler were fitted on the full dataset before the train/test
  split. They are now fitted on the training split only.
- Validation rejected `0` as missing (midnight transactions, applicants with
  no existing loans).
- Models were pickled with an older scikit-learn than the one installed.
  Retrained, and dependency versions are now pinned.

### Added
- Input validation with readable 400 errors for all endpoints.
- `GET /forecast-inflation`, `GET /model-info`, `POST /predict-fraud/batch`.
- ROC-AUC and PR-AUC for both fraud models.
- Fraud decision threshold tuned on out-of-fold training predictions.
- Optional SMOTE oversampling (`python src/train_all.py --smote`).
- ARIMA backtest on the last 12 months, compared against a naive
  "no change" forecast.
- Logging, training timestamp in the metrics report, Dockerfile.

### Changed
- Shared paths, feature lists and seeds moved to `src/config.py`.
- Models are loaded once and cached instead of on every request.
- Random Forest uses a single core for single-row predictions
  (85 ms → 37 ms per request).

### Known issues
- All data is synthetic. Fraud classes are almost perfectly separable, so
  fraud metrics (F1 = 1.0) are not meaningful.
- ARIMA(2,1,2) does worse than the naive forecast in the backtest
  (MAE 0.78 vs 0.39). The series is seasonal and the model is not.
