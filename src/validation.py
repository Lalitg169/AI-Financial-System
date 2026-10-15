"""
Request validation for the API.

Each validator returns None when the input is fine, or a short error
message the API can send back with a 400 status.
"""
from src.config import FRAUD_FEATURES, CREDIT_NUM_FEATURES, CREDIT_CAT_FEATURES
from src.model_store import load_artifact


def _check_numeric_fields(data, fields):
    if not isinstance(data, dict):
        return "Request body must be a JSON object"

    # `is None` rather than `not`: 0 is a valid value (midnight, no loans)
    missing = [f for f in fields if data.get(f) is None]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"

    not_numbers = [f for f in fields
                   if isinstance(data[f], bool) or not isinstance(data[f], (int, float))]
    if not_numbers:
        return f"Fields must be numbers: {', '.join(not_numbers)}"
    return None


def validate_fraud_input(data):
    return _check_numeric_fields(data, FRAUD_FEATURES)


def validate_credit_input(data):
    error = _check_numeric_fields(data, CREDIT_NUM_FEATURES)
    if error:
        return error

    # allowed categories come from the encoders saved at training time
    encoders = load_artifact("credit_encoders.pkl")
    for field in CREDIT_CAT_FEATURES:
        allowed = list(encoders[field].classes_)
        if data.get(field) not in allowed:
            return f"{field} must be one of: {', '.join(allowed)}"
    return None
