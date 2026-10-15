"""
Request validation for the API.

Each validator returns None when the input is fine, or a short error
message the API can send back with a 400 status.
"""
from src.config import FRAUD_FEATURES


def _check_numeric_fields(data, fields):
    if not isinstance(data, dict):
        return "Request body must be a JSON object"

    missing = [f for f in fields if not data.get(f)]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"

    not_numbers = [f for f in fields
                   if isinstance(data[f], bool) or not isinstance(data[f], (int, float))]
    if not_numbers:
        return f"Fields must be numbers: {', '.join(not_numbers)}"
    return None


def validate_fraud_input(data):
    return _check_numeric_fields(data, FRAUD_FEATURES)
