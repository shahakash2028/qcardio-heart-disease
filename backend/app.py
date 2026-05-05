"""
app.py — Heart Disease Predictor Flask API (FINAL VERIFIED VERSION)

Loads 5 artifacts produced by train_model.py:
  models/stack_model.pkl    — stacking ensemble   (primary, 70% weight)
  models/scaler.pkl         — scaler for engineered features
  models/scaler_q.pkl       — scaler for raw 13 features   ← was missing before
  models/hybrid_model.pth   — quantum model weights
  models/calibrator.pkl     — quantum probability calibrator

Gracefully degrades: if quantum model missing, uses 100% stacking.
"""
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "config.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from backend.quantum_model import HybridModel
except ModuleNotFoundError:
    try:
        from quantum_model import HybridModel
    except ModuleNotFoundError:
        HybridModel = None
        print("⚠ quantum_model not found — quantum inference disabled")

app = Flask(__name__)
CORS(app)

# ── Global handles ─────────────────────────────────────────────────────────
_stack      = None
_scaler     = None
_scaler_q   = None   # ← raw-feature scaler (was missing before)
_qmodel     = None
_calibrator = None


# ── Feature engineering (mirrors train_model.py exactly) ──────────────────
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
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
def load_all():
    global _stack, _scaler, _scaler_q, _qmodel, _calibrator

    if _stack is not None:
        return True

    missing_critical = []

    # ── Stacking ensemble (required) ──────────────────────────────────────
    if os.path.exists(config.STACK_PATH):
        _stack = joblib.load(config.STACK_PATH)
        print(f"✅ Stack loaded       → {config.STACK_PATH}")
    else:
        missing_critical.append(config.STACK_PATH)
        print(f"❌ Stack MISSING      → {config.STACK_PATH}")

    # ── Engineered-feature scaler (required) ──────────────────────────────
    if os.path.exists(config.SCALER_PATH):
        _scaler = joblib.load(config.SCALER_PATH)
        print(f"✅ Scaler loaded      → {config.SCALER_PATH}")
    else:
        missing_critical.append(config.SCALER_PATH)
        print(f"❌ Scaler MISSING     → {config.SCALER_PATH}")

    # ── Raw-feature scaler for quantum model ──────────────────────────────
    if os.path.exists(config.SCALER_Q_PATH):
        _scaler_q = joblib.load(config.SCALER_Q_PATH)
        print(f"✅ Scaler_q loaded    → {config.SCALER_Q_PATH}")
    else:
        print(f"⚠ Scaler_q missing   → quantum will use main scaler as fallback")

    # ── Quantum model (optional) ───────────────────────────────────────────
    if HybridModel and os.path.exists(config.MODEL_PATH):
        try:
            m = HybridModel()
            sd = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            m.eval()
            _qmodel = m
            print(f"✅ Quantum loaded     → {config.MODEL_PATH}")
        except Exception as e:
            print(f"⚠ Quantum load fail: {e}")
    else:
        print("⚠ Quantum model unavailable — stacking ensemble only (100% weight)")

    # ── Calibrator (optional) ─────────────────────────────────────────────
    if os.path.exists(config.CALIB_PATH):
        _calibrator = joblib.load(config.CALIB_PATH)
        print(f"✅ Calibrator loaded  → {config.CALIB_PATH}")
    else:
        print("⚠ Calibrator missing — raw quantum prob used directly")

    if missing_critical:
        print(f"\n❌ CRITICAL files missing: {missing_critical}")
        print("   Run `python train_model.py` first.\n")
        return False

    print("✅ All components ready\n")
    return True


with app.app_context():
    load_all()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return jsonify({
        "status":            "online",
        "stack_loaded":      _stack      is not None,
        "quantum_loaded":    _qmodel     is not None,
        "scaler_loaded":     _scaler     is not None,
        "scaler_q_loaded":   _scaler_q   is not None,
        "calibrator_loaded": _calibrator is not None,
    })


@app.route("/health")
def health():
    ready = (_stack is not None and _scaler is not None)
    return jsonify({"status": "ok" if ready else "not_ready"}), (200 if ready else 503)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_all()

        if _stack is None or _scaler is None:
            return jsonify({
                "prediction":  0,
                "label":       "Model Not Ready",
                "probability": 0.0,
                "status":      "error",
                "detail":      "Run train_model.py first.",
            }), 503

        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "prediction": 0, "label": "No Input",
                "probability": 0.0, "status": "error"
            }), 400

        print(f"📥 /predict | keys: {sorted(data.keys())}")

        # ── Extract raw 13 features ────────────────────────────────────────
        try:
            raw = {k: float(data.get(k, 0.0)) for k in config.FEATURE_NAMES}
        except (ValueError, TypeError) as fe:
            print(f"❌ Feature parse error: {fe}")
            raw = {k: 0.0 for k in config.FEATURE_NAMES}

        print(f"   Features: { {k: raw[k] for k in ['age','cp','thalach','ca','oldpeak']} } …")

        raw_df  = pd.DataFrame([raw])          # shape (1, 13)
        eng_df  = _engineer(raw_df)             # shape (1, 24)

        raw_arr = raw_df.values.astype(np.float32)
        eng_arr = eng_df.values.astype(np.float32)

        # ── Scale ─────────────────────────────────────────────────────────
        eng_sc = _scaler.transform(eng_arr)
        eng_sc = np.nan_to_num(eng_sc, nan=0.0, posinf=1.0, neginf=-1.0)

        # For quantum: use dedicated scaler_q (raw 13 features)
        sc_q     = _scaler_q if _scaler_q is not None else _scaler
        raw_sc   = sc_q.transform(raw_arr)
        raw_sc   = np.nan_to_num(raw_sc,  nan=0.0, posinf=1.0, neginf=-1.0)

        # ── Stacking probability (primary) ────────────────────────────────
        stack_prob = float(_stack.predict_proba(eng_sc)[0][1])
        print(f"   Stack prob:   {stack_prob:.4f}")

        # ── Quantum probability (secondary) ───────────────────────────────
        raw_q_prob = None
        cal_q_prob = stack_prob   # safe default

        if _qmodel is not None:
            try:
                tensor = torch.tensor(raw_sc, dtype=torch.float32)
                with torch.inference_mode():
                    raw_q_prob = float(_qmodel(tensor).item())

                if _calibrator is not None:
                    cal_q_prob = float(
                        _calibrator.predict_proba([[raw_q_prob]])[0][1]
                    )
                else:
                    cal_q_prob = raw_q_prob

                print(f"   Quantum raw:  {raw_q_prob:.4f} | calibrated: {cal_q_prob:.4f}")

            except Exception as qe:
                print(f"   ⚠ Quantum failed ({qe}) — using stacking only")
                cal_q_prob = stack_prob
                raw_q_prob = None

        # ── Weighted ensemble ─────────────────────────────────────────────
        if _qmodel is not None:
            sw, qw = config.STACK_WEIGHT, config.QUANTUM_WEIGHT
        else:
            sw, qw = 1.0, 0.0

        probability = float(np.clip(sw * stack_prob + qw * cal_q_prob, 0.0, 1.0))
        prediction  = 1 if probability > config.DECISION_THRESHOLD else 0

        print(f"   Final prob:   {probability:.4f} → prediction: {prediction}")

        return jsonify({
            "prediction":  prediction,
            "label":       "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": round(probability, 4),
            "status":      "success",
            "raw_q_prob":  round(raw_q_prob, 4) if raw_q_prob is not None else None,
            "stack_prob":  round(stack_prob, 4),
        })

    except Exception as e:
        print(f"❌ /predict error: {e}")
        traceback.print_exc()
        return jsonify({
            "prediction":  0,
            "label":       "Processing Error",
            "probability": 0.0,
            "status":      "fallback",
            "detail":      str(e),
        })


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)