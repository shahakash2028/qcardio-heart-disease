"""
main.py — CLI sanity-check for the trained Heart Disease model.

Run from project root:
    python main.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
import joblib
import numpy as np
import pandas as pd
import torch

try:
    from backend.quantum_model import HybridModel
except ModuleNotFoundError:
    try:
        from backend.quantum_model import HybridModel
    except ModuleNotFoundError:
        HybridModel = None


# ── Feature engineering (mirrors train_model.py exactly) ──────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_group"]       = pd.cut(d["age"], bins=[0, 40, 50, 60, 120],
                                   labels=[0, 1, 2, 3]).astype(int)
    d["hr_reserve"]      = d["thalach"] - (220 - d["age"])
    d["angina_score"]    = d["cp"] * (1 + d["exang"])
    d["vascular_burden"] = d["ca"] + d["thal"] * 0.5
    d["st_composite"]    = d["oldpeak"] * (3 - d["slope"] + 1)
    d["chol_age_risk"]   = d["chol"] * d["age"] / 5000.0
    d["cp_thalach"]      = d["cp"]    * d["thalach"]  / 100.0
    d["exang_oldpeak"]   = d["exang"] * d["oldpeak"]
    d["ca_thal"]         = d["ca"]    * d["thal"]
    d["age_thalach"]     = d["age"]   * d["thalach"]  / 10000.0
    d["oldpeak_slope"]   = d["oldpeak"] * (3.0 - d["slope"])
    return d


# ══════════════════════════════════════════════════════════════════════════════
def load_models():
    for path, name in [(config.STACK_PATH, "Stacking model"),
                        (config.SCALER_PATH, "Scaler")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at {path!r}.\nRun `python train_model.py` first."
            )

    stack    = joblib.load(config.STACK_PATH)
    scaler   = joblib.load(config.SCALER_PATH)
    scaler_q = joblib.load(config.SCALER_Q_PATH) if os.path.exists(config.SCALER_Q_PATH) else scaler
    calib    = joblib.load(config.CALIB_PATH)     if os.path.exists(config.CALIB_PATH)    else None

    qmodel = None
    if HybridModel and os.path.exists(config.MODEL_PATH):
        try:
            m = HybridModel()
            m.load_state_dict(torch.load(config.MODEL_PATH,
                                          map_location="cpu", weights_only=True))
            m.eval()
            qmodel = m
        except Exception as e:
            print(f"[WARN] Quantum model load failed: {e}")

    return stack, scaler, scaler_q, qmodel, calib


# ══════════════════════════════════════════════════════════════════════════════
def predict(features: dict, stack, scaler, scaler_q, qmodel, calib) -> tuple[str, float, float | None]:
    raw_df = pd.DataFrame([features])
    eng_df = engineer(raw_df)

    raw_arr = raw_df.values.astype(np.float32)
    eng_arr = eng_df.values.astype(np.float32)

    eng_sc = scaler.transform(eng_arr)
    eng_sc = np.nan_to_num(eng_sc)

    stack_prob = float(stack.predict_proba(eng_sc)[0][1])

    raw_q = None
    if qmodel is not None:
        raw_sc = scaler_q.transform(raw_arr)
        raw_sc = np.nan_to_num(raw_sc)
        with torch.no_grad():
            raw_q = float(qmodel(torch.tensor(raw_sc, dtype=torch.float32)).item())
        cal_q = float(calib.predict_proba([[raw_q]])[0][1]) if calib else raw_q
        prob  = config.STACK_WEIGHT * stack_prob + config.QUANTUM_WEIGHT * cal_q
    else:
        prob = stack_prob

    prob  = float(np.clip(prob, 0.0, 1.0))
    label = "Heart Disease" if prob > config.DECISION_THRESHOLD else "No Heart Disease"
    return label, prob, raw_q


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SEP = "=" * 60
    print(SEP)
    print("  Hybrid Quantum-Classical Heart Disease Predictor")
    print(SEP)

    try:
        stack, scaler, scaler_q, qmodel, calib = load_models()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"  Stack      : ✅")
    print(f"  Quantum    : {'✅' if qmodel else '⚠ not available'}")
    print(f"  Calibrator : {'✅' if calib  else '⚠ not available'}")
    print(f"  Threshold  : {config.DECISION_THRESHOLD}")
    print(f"  Weights    : {config.STACK_WEIGHT:.0%} stack / {config.QUANTUM_WEIGHT:.0%} quantum")
    print()

    # ── Test samples verified against actual dataset rows ─────────────────
    samples = [
        {
            "desc":  "✅ Dataset row — confirmed NO disease (target=0)",
            "age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,
            "fbs":0,"restecg":1,"thalach":168,"exang":0,
            "oldpeak":1.0,"slope":2,"ca":2,"thal":3,
        },
        {
            "desc":  "✅ Dataset row — confirmed HAS disease (target=1)",
            "age":58,"sex":0,"cp":0,"trestbps":100,"chol":248,
            "fbs":0,"restecg":0,"thalach":122,"exang":0,
            "oldpeak":1.0,"slope":1,"ca":0,"thal":2,
        },
        {
            "desc":  "Strong disease indicators (high cp, high thalach, low ca)",
            "age":50,"sex":0,"cp":3,"trestbps":120,"chol":230,
            "fbs":0,"restecg":1,"thalach":180,"exang":0,
            "oldpeak":0.1,"slope":2,"ca":0,"thal":2,
        },
        {
            "desc":  "Strong no-disease indicators (low cp, exang=1, high ca)",
            "age":65,"sex":1,"cp":0,"trestbps":150,"chol":260,
            "fbs":1,"restecg":0,"thalach":120,"exang":1,
            "oldpeak":3.5,"slope":0,"ca":3,"thal":3,
        },
    ]

    for i, s in enumerate(samples, 1):
        desc = s.pop("desc")
        feat = {k: v for k, v in s.items() if k in config.FEATURE_NAMES}
        label, prob, raw_q = predict(feat, stack, scaler, scaler_q, qmodel, calib)
        rq_str = f"{raw_q:.4f}" if raw_q is not None else "N/A"
        print(f"Sample {i}: {desc}")
        print(f"  Prediction : {label}")
        print(f"  Probability: {prob:.4f}  (quantum raw: {rq_str})")
        print()

    print("-" * 60)
    print(f"API : http://localhost:5000/predict  (run backend/app.py)")
    print(f"UI  : open frontend/index.html")