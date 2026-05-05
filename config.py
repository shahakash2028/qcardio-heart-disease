"""
config.py — Heart Disease Predictor (Final Verified Version)
Place at PROJECT ROOT alongside main.py, train_model.py, backend/
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")

MODEL_PATH  = os.path.join(MODEL_DIR, "hybrid_model.pth")   # quantum model weights
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")          # scaler for engineered features
SCALER_Q_PATH = os.path.join(MODEL_DIR, "scaler_q.pkl")      # scaler for raw 13 features
STACK_PATH  = os.path.join(MODEL_DIR, "stack_model.pkl")     # stacking ensemble
CALIB_PATH  = os.path.join(MODEL_DIR, "calibrator.pkl")      # quantum probability calibrator

# ── Data path ──────────────────────────────────────────────────────────────
# Single file mode (heart.csv) — default
DATA_PATH   = os.path.join(BASE_DIR, "heart.csv")

# Multi-file mode (full UCI) — set USE_FULL_UCI=True and place files in heart_disease/
# Gives ~740 unique rows vs 302 — recommended for 90%+ accuracy
USE_FULL_UCI = True
UCI_DIR      = os.path.join(BASE_DIR, "heart_disease")

# ── Quantum hyper-parameters ───────────────────────────────────────────────
NUM_QUBITS   = 4    # keep ≤ 6 for simulation speed
QUANTUM_REPS = 1    # reps=1 is 4× faster than reps=2; increase only with full UCI dataset

# ── Decision threshold ─────────────────────────────────────────────────────
# 0.45 balances sensitivity/specificity on this dataset.
# Lower = more sensitive (fewer missed disease cases) but more false alarms.
DECISION_THRESHOLD = 0.45

# ── Ensemble weights ───────────────────────────────────────────────────────
# Stacking is more reliable on small data (≤302 rows).
# Quantum contributes 30% — adds probabilistic diversity.
STACK_WEIGHT   = 0.70
QUANTUM_WEIGHT = 0.30

# ── Feature names — must exactly match CSV column order ───────────────────
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# ── Feature bounds for UI validation ──────────────────────────────────────
FEATURE_BOUNDS = {
    "age":      (1,   120),
    "sex":      (0,     1),
    "cp":       (0,     3),
    "trestbps": (80,  200),
    "chol":     (100, 600),
    "fbs":      (0,     1),
    "restecg":  (0,     2),
    "thalach":  (60,  220),
    "exang":    (0,     1),
    "oldpeak":  (0.0,  6.2),
    "slope":    (0,     2),
    "ca":       (0,     4),
    "thal":     (0,     3),
}