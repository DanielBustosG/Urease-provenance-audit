"""Applicability-domain-aware urease cascade predictor.

A view over ``app.cascade`` — the single inference library. The Streamlit UI
(``app/app.py``) and the CLI (``app/predict.py``) both call it; neither
reimplements any inference step.
"""

__version__ = "1.0.0"
