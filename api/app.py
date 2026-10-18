"""
Flask REST API  –  Smart Financial Risk Intelligence System
Endpoints:
  POST /predict-fraud
  POST /predict-fraud/batch
  POST /predict-credit
  GET  /forecast-inflation?steps=6
  GET  /model-info
  GET  /health
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import sklearn
from flask import Flask, request, jsonify
from src.fraud_detection import predict_fraud
from src.credit_risk      import predict_credit
from src.economic_trend   import forecast_inflation
from src.config           import MODEL_DIR, REPORT_DIR
from src.logger           import get_logger
from src.validation       import validate_fraud_input, validate_credit_input

log = get_logger("api")

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Smart Financial Risk Intelligence System"})


@app.route("/predict-fraud", methods=["POST"])
def fraud_endpoint():
    """
    Input JSON:
    {
      "amount": 250.0,
      "time_of_day": 2,
      "transaction_freq": 15,
      "distance_from_home": 300,
      "v1": 1.5,
      "v2": -2.0,
      "v3": 1.8
    }
    """
    data  = request.get_json(silent=True)
    error = validate_fraud_input(data)
    if error:
        log.warning("Rejected /predict-fraud request: %s", error)
        return jsonify({"success": False, "error": error}), 400

    try:
        result = predict_fraud(data)
        return jsonify({"success": True, "prediction": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


MAX_BATCH = 1000


@app.route("/predict-fraud/batch", methods=["POST"])
def fraud_batch_endpoint():
    """Input JSON: {"transactions": [ {...}, {...} ]}  (up to 1000)"""
    data = request.get_json(silent=True)
    txns = data.get("transactions") if isinstance(data, dict) else None
    if not isinstance(txns, list) or not txns:
        return jsonify({"success": False,
                        "error": "Body must be {\"transactions\": [ ... ]} with at least one item"}), 400
    if len(txns) > MAX_BATCH:
        return jsonify({"success": False, "error": f"At most {MAX_BATCH} transactions per batch"}), 400

    # validate everything first so one bad row doesn't leave a half-done batch
    errors = {i: err for i, t in enumerate(txns) if (err := validate_fraud_input(t))}
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    results = [predict_fraud(t) for t in txns]
    return jsonify({"success": True, "count": len(results),
                    "fraud_count": sum(r["rf_prediction"] for r in results),
                    "predictions": results}), 200


@app.route("/predict-credit", methods=["POST"])
def credit_endpoint():
    """
    Input JSON:
    {
      "age": 35,
      "income": 60000,
      "loan_amount": 20000,
      "loan_tenure": 36,
      "credit_score": 650,
      "existing_loans": 1,
      "employment_type": "Salaried",
      "education": "Graduate"
    }
    """
    data  = request.get_json(silent=True)
    error = validate_credit_input(data)
    if error:
        log.warning("Rejected /predict-credit request: %s", error)
        return jsonify({"success": False, "error": error}), 400

    try:
        result = predict_credit(data)
        return jsonify({"success": True, "prediction": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/forecast-inflation", methods=["GET"])
def forecast_endpoint():
    """CPI forecast for the next `steps` months (1-24, default 6)."""
    raw = request.args.get("steps", "6")
    steps = int(raw) if raw.isdigit() else None
    if steps is None or not 1 <= steps <= 24:
        return jsonify({"success": False, "error": "steps must be an integer from 1 to 24"}), 400

    try:
        result = forecast_inflation(steps=steps)
        return jsonify({"success": True, "forecast": result}), 200
    except Exception as e:
        log.exception("Forecast failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Headline metrics from the last training run and the saved model files."""
    try:
        report = json.loads((REPORT_DIR / "metrics_report.json").read_text())
    except FileNotFoundError:
        return jsonify({"success": False, "error": "No metrics report. Run src/train_all.py first."}), 404

    fraud, credit, infl = report["fraud"], report["credit"], report["inflation"]
    metrics = {
        "fraud": {name: {k: fraud[name][k] for k in ("precision", "recall", "f1_score")}
                  for name in ("random_forest", "isolation_forest")},
        "credit": {name: {k: credit[name][k] for k in ("precision", "recall", "f1_score", "auc_roc")}
                   for name in ("logistic_regression", "decision_tree")},
        "inflation": {k: infl[k] for k in ("order", "aic", "rmse", "mae")},
    }
    artifacts = {f.name: f"{f.stat().st_size / 1024:.1f} KB"
                 for f in sorted(MODEL_DIR.glob("*.pkl"))}
    return jsonify({"success": True, "trained_at": report.get("trained_at"),
                    "metrics": metrics, "artifacts": artifacts,
                    "scikit_learn_version": sklearn.__version__}), 200


if __name__ == "__main__":
    log.info("Starting Flask API on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
