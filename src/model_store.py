"""
Loads saved model artifacts once and keeps them in memory.

The API used to call joblib.load on every request, re-reading the same
files from disk each time. lru_cache returns the already-loaded object
after the first call.
"""
from functools import lru_cache

import joblib

from src.config import MODEL_DIR


@lru_cache(maxsize=None)
def load_artifact(filename: str):
    return joblib.load(MODEL_DIR / filename)


def clear_cache():
    """Call after retraining so the next request picks up the new files."""
    load_artifact.cache_clear()
