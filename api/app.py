"""
Flask REST API  –  Smart Financial Risk Intelligence System
Endpoints:
  POST /predict-fraud
  POST /predict-credit
  GET  /health
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify
from src.fraud_detection import predict_fraud
from src.credit_risk      import predict_credit

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
    try:
        data   = request.get_json(force=True)
        result = predict_fraud(data)
        return jsonify({"success": True, "prediction": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


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
    try:
        data   = request.get_json(force=True)
        result = predict_credit(data)
        return jsonify({"success": True, "prediction": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("Starting Flask API on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
