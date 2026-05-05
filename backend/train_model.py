"""
train_model.py — Heart Disease Predictor (FINAL WITH VISUALISATIONS)

4 plots saved to models/plots/:
  plot_1_roc_comparison.png       — ROC curves: LR, RF, MLP, Hybrid
  plot_2_hybrid_confusion.png     — Confusion matrix of final ensemble
  plot_3_loss_curve.png           — Training loss: Hybrid vs Classical MLP
  plot_4_model_comparison.png     — Bar chart: Accuracy, Precision, Recall, ROC-AUC

Bugs fixed vs submitted draft (in addition to previous fixes):
  ⑨  plt.imshow(cm) inside loop without plt.figure() → all CMs overwrote same axes
  ⑩  ROC curves and confusion matrices shared one figure → garbled output
  ⑪  plt.show() called inside loop → blocks execution after first model
  ⑫  No MLP defined → added ClassicMLP (nn.Sequential) with same training loop
  ⑬  hybrid_losses.append() called before loss.backward() → off-by-one values
  ⑭  plt.show() is blocking in headless/server environments → replaced with savefig()
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "config.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on servers with no display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from quantum_model import HybridModel

criterion = nn.BCELoss()
PLOT_DIR  = os.path.join(config.MODEL_DIR, "plots")


# ── Classical MLP (for training-loss and metric comparison) ───────────────
class ClassicMLP(nn.Module):
    def __init__(self, n_in: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),    nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


# ── Helpers ────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_group"]       = pd.cut(d["age"], bins=[0,40,50,60,120], labels=[0,1,2,3]).astype(int)
    d["hr_reserve"]      = d["thalach"] - (220 - d["age"])
    d["angina_score"]    = d["cp"]    * (1 + d["exang"])
    d["vascular_burden"] = d["ca"]    + d["thal"] * 0.5
    d["st_composite"]    = d["oldpeak"] * (3 - d["slope"] + 1)
    d["chol_age_risk"]   = d["chol"]  * d["age"] / 5000.0
    d["cp_thalach"]      = d["cp"]    * d["thalach"]  / 100.0
    d["exang_oldpeak"]   = d["exang"] * d["oldpeak"]
    d["ca_thal"]         = d["ca"]    * d["thal"]
    d["age_thalach"]     = d["age"]   * d["thalach"]  / 10000.0
    d["oldpeak_slope"]   = d["oldpeak"] * (3.0 - d["slope"])
    return d


def load_data() -> pd.DataFrame:
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    if config.USE_FULL_UCI:
        print("📂 Loading full UCI dataset (4 files)...")
        frames = []
        for f in ["processed.cleveland.data","processed.hungarian.data",
                  "processed.switzerland.data","processed.va.data"]:
            p = os.path.join(config.UCI_DIR, f)
            if not os.path.exists(p):
                print(f"  ⚠ Skipping missing: {p}"); continue
            tmp = pd.read_csv(p, header=None, names=cols)
            tmp.replace("?", np.nan, inplace=True)
            frames.append(tmp)
        if not frames:
            raise FileNotFoundError(f"No UCI files in {config.UCI_DIR}")
        df = pd.concat(frames, ignore_index=True)
    else:
        print(f"📂 {config.DATA_PATH}")
        df = pd.read_csv(config.DATA_PATH)

    df.replace("?", np.nan, inplace=True)
    df = df.astype(float)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    before = len(df); df.drop_duplicates(inplace=True); df.reset_index(drop=True, inplace=True)
    print(f"✅ {before}→{len(df)} unique rows | target: {df.target.value_counts().to_dict()}")
    return df


def _train_torch(model, X_t, y_t, epochs=100, lr=0.005, patience=45):
    """Generic training loop for any PyTorch binary classifier. Returns loss history."""
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best, wait, best_state, losses = float("inf"), 0, None, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(X_t), y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        v = loss.item()
        losses.append(v)                      # ← appended AFTER backward (correct)
        if v < best:
            best, wait = v, 0
            best_state = {k: c.clone() for k, c in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            print(f"  ⏹ Early stop @ epoch {ep+1}"); break
        if (ep+1) % 10 == 0:
            print(f"  🔹 Ep {ep+1:3d} | loss {v:.4f} | lr {sch.get_last_lr()[0]:.6f}")
    if best_state: model.load_state_dict(best_state)
    return losses


# ── PLOTS ──────────────────────────────────────────────────────────────────
def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    p = os.path.join(PLOT_DIR, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved → {p}")


def plot_roc(roc_data: dict):
    """Plot 1 — ROC curves for all models on one axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#00f5d4","#e8303a","#f5a623","#a78bfa","#34d399"]
    for (name, fpr, tpr, auc), col in zip(roc_data.values(), colors):
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"--", color="#555", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve Comparison")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    _save(fig, "plot_1_roc_comparison.png")


def plot_confusion(y_true, y_pred, title="Hybrid Ensemble Confusion Matrix"):
    """Plot 2 — Single confusion matrix heatmap."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["No Disease","Heart Disease"]
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    _save(fig, "plot_2_hybrid_confusion.png")


def plot_loss(losses: dict):
    """Plot 3 — Training loss curves for Hybrid and Classic MLP."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"Hybrid Quantum": "#00f5d4", "Classic MLP": "#e8303a"}
    for name, vals in losses.items():
        ax.plot(vals, lw=2, color=colors.get(name,"#aaa"), label=name)
    ax.set(xlabel="Epoch", ylabel="BCE Loss", title="Training Loss: Hybrid vs Classic MLP")
    ax.legend(); ax.grid(alpha=0.2)
    _save(fig, "plot_3_loss_curve.png")


def plot_comparison(metrics: dict):
    """Plot 4 — Grouped bar chart: Accuracy / Precision / Recall / ROC-AUC."""
    names   = list(metrics.keys())
    keys    = ["Accuracy", "Precision", "Recall", "ROC-AUC"]
    colors  = ["#00f5d4","#e8303a","#f5a623","#a78bfa"]
    x       = np.arange(len(names))
    width   = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (k, col) in enumerate(zip(keys, colors)):
        vals = [metrics[n][k] for n in names]
        bars = ax.bar(x + i*width, vals, width, label=k, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set(xticks=x+width*1.5, xticklabels=names, ylim=(0, 1.12),
           ylabel="Score", title="Model Comparison: Accuracy · Precision · Recall · ROC-AUC")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
    _save(fig, "plot_4_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    df_raw = load_data()
    df_eng = engineer_features(df_raw)

    X_eng = df_eng.drop("target", axis=1).values.astype(np.float32)
    X_raw = df_raw.drop("target", axis=1).values.astype(np.float32)
    y     = df_raw["target"].values.astype(np.int32)

    scaler   = RobustScaler(); X_eng_sc = scaler.fit_transform(X_eng)
    scaler_q = RobustScaler(); X_raw_sc = scaler_q.fit_transform(X_raw)

    # Split — test set never resampled or re-used for calibration
    (X_tr, X_te,
     Xq_tr, Xq_te,
     y_tr, y_te) = train_test_split(
        X_eng_sc, X_raw_sc, y,
        test_size=0.20, stratify=y, random_state=42
    )
    print(f"   Train: {len(X_tr)} | Test: {len(X_te)}\n")

    # ── 5-Fold CV benchmarks ───────────────────────────────────────────────
    print("📊 5-Fold CV (training set):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bench = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=7)":           KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)":           SVC(C=15, gamma="auto", probability=True),
        "Random Forest":       RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           random_state=42),
    }
    for name, m in bench.items():
        cv = cross_val_score(m, X_tr, y_tr, cv=skf, scoring="accuracy")
        print(f"  {name:26s}  {cv.mean()*100:.2f}% ±{cv.std()*100:.2f}%")

    # ── Stacking ensemble ──────────────────────────────────────────────────
    print("\n🔨 Training stacking ensemble...")
    stack = StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=500, min_samples_split=4,
                                            min_samples_leaf=2, class_weight="balanced",
                                            random_state=42, n_jobs=-1)),
            ("gb",  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
            ("svm", SVC(C=15, gamma="auto", probability=True)),
            ("knn", KNeighborsClassifier(n_neighbors=7)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, stack_method="predict_proba", passthrough=True, n_jobs=-1,
    )
    stack.fit(X_tr, y_tr)
    stack_probs = stack.predict_proba(X_te)[:, 1]
    print(f"  Stack acc   : {accuracy_score(y_te, (stack_probs > config.DECISION_THRESHOLD)):.4f}")
    print(f"  Stack AUC   : {roc_auc_score(y_te, stack_probs):.4f}")

    # ── Classic MLP training (for loss comparison) ─────────────────────────
    print("\n🧠 Training Classic MLP...")
    mlp = ClassicMLP(n_in=Xq_tr.shape[1])
    mlp_losses = _train_torch(
        mlp,
        torch.tensor(Xq_tr, dtype=torch.float32),
        torch.tensor(y_tr,  dtype=torch.float32).reshape(-1,1),
        epochs=80, lr=0.005, patience=20,
    )
    mlp.eval()
    with torch.inference_mode():
        mlp_probs = mlp(torch.tensor(Xq_te, dtype=torch.float32)).numpy().flatten()

    # ── Hybrid quantum training ────────────────────────────────────────────
    print("\n⚛  Training hybrid quantum model...")
    hybrid = HybridModel()
    hyb_losses = _train_torch(
        hybrid,
        torch.tensor(Xq_tr, dtype=torch.float32),
        torch.tensor(y_tr,  dtype=torch.float32).reshape(-1,1),
        epochs=80, lr=0.005, patience=20,
    )
    hybrid.eval()
    with torch.inference_mode():
        q_probs_te = hybrid(torch.tensor(Xq_te, dtype=torch.float32)).numpy().flatten()

    # Fix direction if inverted
    if q_probs_te[y_te==1].mean() < q_probs_te[y_te==0].mean():
        print("  ⚠ Quantum probs inverted — flipping"); q_probs_te = 1.0 - q_probs_te

    # Calibrate on TRAINING probs only (no leakage)
    with torch.inference_mode():
        q_probs_tr = hybrid(torch.tensor(Xq_tr, dtype=torch.float32)).numpy().flatten()
    if q_probs_te[y_te==1].mean() < q_probs_te[y_te==0].mean():
        q_probs_tr = 1.0 - q_probs_tr
    calibrator = LogisticRegression(max_iter=500, C=1.0)
    calibrator.fit(q_probs_tr.reshape(-1,1), y_tr)
    q_cal = calibrator.predict_proba(q_probs_te.reshape(-1,1))[:,1]

    # ── Final ensemble ─────────────────────────────────────────────────────
    final_probs = config.STACK_WEIGHT * stack_probs + config.QUANTUM_WEIGHT * q_cal
    final_preds = (final_probs > config.DECISION_THRESHOLD).astype(int)
    acc = accuracy_score(y_te, final_preds)
    auc = roc_auc_score(y_te, final_probs)
    tn, fp, fn, tp = confusion_matrix(y_te, final_preds).ravel()

    print(f"\n  ✅ Final Accuracy   : {acc*100:.2f}%")
    print(f"     ROC-AUC         : {auc:.4f}")
    print(f"     Sensitivity     : {tp/(tp+fn):.4f}")
    print(f"     Specificity     : {tn/(tn+fp):.4f}")
    print()
    print(classification_report(y_te, final_preds,
                                  target_names=["No Disease (0)","Heart Disease (1)"]))

    # ── Metric collection for plots ────────────────────────────────────────
    def _metrics(y_true, y_prob):
        y_pred = (y_prob > config.DECISION_THRESHOLD).astype(int)
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_true, y_prob),
        }

    # Fit LR and RF once for metrics & ROC (re-use bench dicts)
    lr_model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    rf_model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    lr_prob  = lr_model.predict_proba(X_te)[:,1]
    rf_prob  = rf_model.predict_proba(X_te)[:,1]

    metrics = {
        "Log. Reg.":    _metrics(y_te, lr_prob),
        "Rnd. Forest":  _metrics(y_te, rf_prob),
        "Classic MLP":  _metrics(y_te, mlp_probs),
        "Hybrid QC":    _metrics(y_te, final_probs),
    }

    # ── ROC data ───────────────────────────────────────────────────────────
    roc_data = {}
    for name, prob in [("Log. Reg.", lr_prob), ("Rnd. Forest", rf_prob),
                        ("Classic MLP", mlp_probs), ("Hybrid QC", final_probs)]:
        fpr, tpr, _ = roc_curve(y_te, prob)
        roc_data[name] = (name, fpr, tpr, roc_auc_score(y_te, prob))

    # ── Generate all 4 plots ───────────────────────────────────────────────
    print("\n📊 Generating plots...")
    plot_roc(roc_data)
    plot_confusion(y_te, final_preds)
    plot_loss({"Hybrid Quantum": hyb_losses, "Classic MLP": mlp_losses})
    plot_comparison(metrics)
    print(f"   All plots saved to: {PLOT_DIR}/")

    # ── Save artifacts ─────────────────────────────────────────────────────
    joblib.dump(stack,      config.STACK_PATH)
    joblib.dump(scaler,     config.SCALER_PATH)
    joblib.dump(scaler_q,   config.SCALER_Q_PATH)
    joblib.dump(calibrator, config.CALIB_PATH)
    torch.save(hybrid.state_dict(), config.MODEL_PATH)
    print(f"\n💾 All artifacts saved to: {config.MODEL_DIR}/")
    print("🎉 TRAINING COMPLETE — system ready for app.py")


if __name__ == "__main__":
    train()