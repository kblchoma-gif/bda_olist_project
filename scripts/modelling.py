"""Phase 5 — Machine learning: late-delivery prediction.

Target  : is_late (1 = delivered after estimated date, 0 = on-time)
Features: order-time information only (no post-dispatch data → no leakage)
Models  : Baseline (majority class) → Logistic Regression → Random Forest
Metrics : Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

Produces:
  outputs/models/logreg.pkl, rf.pkl, preprocessor.pkl
  outputs/tables/model_metrics.csv
  outputs/tables/classification_reports.txt
  outputs/tables/feature_importance_rf.csv
  outputs/figures/06_confusion_logreg.png
  outputs/figures/07_confusion_rf.png
  outputs/figures/08_roc_curves.png
  outputs/figures/09_feature_importance.png
  outputs/tables/class_balance.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.config import PROC_DIR, TAB_DIR, FIG_DIR, MODEL_DIR, RANDOM_STATE

sns.set_theme(style="whitegrid", context="talk")


# ------------------------------------------------------------------ 5.2.1 load + feature selection
LEAKY_COLS = [
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "delivery_days", "delay_days", "order_approved_at",
]
ID_COLS = [
    "order_id", "customer_id", "customer_unique_id",
    "product_id", "seller_id",
    "customer_zip_code_prefix", "seller_zip_code_prefix",
    "flag_timestamp_inversion", "flag_unknown_status",
]

NUMERIC_FEATURES = [
    "price_total", "freight_total", "freight_ratio",
    "payment_installments", "n_items",
    "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm",
    "estimated_days",
]
CATEGORICAL_FEATURES = [
    "customer_state", "seller_state",
    "payment_type", "product_category_name_english",
]

def load_and_prepare() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    p = PROC_DIR / "master.parquet"
    if not p.exists():
        p = PROC_DIR / "master.csv"
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    # Order-purchase temporal features (known at purchase → safe)
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    df["purchase_month"]   = df["order_purchase_timestamp"].dt.month
    df["purchase_dow"]     = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"]    = df["order_purchase_timestamp"].dt.hour

    # Restrict to delivered orders with known timeliness label
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["is_late", "order_delivered_customer_date"])
    df["is_late"] = df["is_late"].astype(int)

    y = df["is_late"]
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + \
                   ["purchase_month", "purchase_dow", "purchase_hour"]
    X = df[feature_cols].copy()

    balance = pd.DataFrame({
        "class": ["on_time (0)", "late (1)"],
        "count": [int((y == 0).sum()), int((y == 1).sum())],
    })
    balance["share_pct"] = (100 * balance["count"] / balance["count"].sum()).round(3)

    return X, y, balance


# ------------------------------------------------------------------ 5.2.2 build pipeline
def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore",
                                  min_frequency=50, sparse_output=False)),
    ])
    all_num = NUMERIC_FEATURES + ["purchase_month", "purchase_dow", "purchase_hour"]
    return ColumnTransformer([
        ("num", numeric_pipe, all_num),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


# ------------------------------------------------------------------ 5.2.3 evaluation helpers
def metrics_row(name: str, y_true, y_pred, y_proba) -> dict:
    return {
        "model":      name,
        "accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_true, y_proba), 4)
                        if y_proba is not None else np.nan,
        "avg_precision": round(average_precision_score(y_true, y_proba), 4)
                        if y_proba is not None else np.nan,
    }


def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred on-time", "Pred late"],
                yticklabels=["True on-time", "True late"], ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print("[INFO] Loading master ...")
    X, y, balance = load_and_prepare()
    balance.to_csv(TAB_DIR / "class_balance.csv", index=False)
    print("\n=== CLASS BALANCE ===")
    print(balance.to_string(index=False))

    # Stratified 80/20 split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"[INFO] Train late-rate: {y_tr.mean():.4%}  "
          f"Test late-rate: {y_te.mean():.4%}")

    pre = build_preprocessor()

    # --- 5.2.4 BASELINE (majority class)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    dummy_pred  = dummy.predict(X_te)
    dummy_proba = dummy.predict_proba(X_te)[:, 1]

    # --- 5.2.5 LOGISTIC REGRESSION (balanced)
    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])
    logreg.fit(X_tr, y_tr)
    lr_pred  = logreg.predict(X_te)
    lr_proba = logreg.predict_proba(X_te)[:, 1]

    # --- 5.2.6 RANDOM FOREST (balanced)
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    rf.fit(X_tr, y_tr)
    rf_pred  = rf.predict(X_te)
    rf_proba = rf.predict_proba(X_te)[:, 1]

    # --- metrics table
    rows = [
        metrics_row("Baseline (majority)", y_te, dummy_pred, dummy_proba),
        metrics_row("Logistic Regression",  y_te, lr_pred,    lr_proba),
        metrics_row("Random Forest",        y_te, rf_pred,    rf_proba),
    ]
    mdf = pd.DataFrame(rows)
    mdf.to_csv(TAB_DIR / "model_metrics.csv", index=False)
    print("\n=== MODEL METRICS (test set) ===")
    print(mdf.to_string(index=False))

    # --- classification reports
    rep_lines = []
    for name, pred in [("Baseline", dummy_pred),
                       ("Logistic Regression", lr_pred),
                       ("Random Forest", rf_pred)]:
        rep_lines.append(f"\n--- {name} ---\n")
        rep_lines.append(classification_report(y_te, pred, digits=4,
                                               zero_division=0))
    rep_txt = "\n".join(rep_lines)
    (TAB_DIR / "classification_reports.txt").write_text(rep_txt, encoding="utf-8")
    print("\n=== CLASSIFICATION REPORTS ===")
    print(rep_txt)

    # --- cross-validation on training set
    print("\n=== 5-FOLD STRATIFIED CV (training set) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores_lr = cross_val_score(logreg, X_tr, y_tr, cv=cv,
                                   scoring="roc_auc", n_jobs=-1)
    cv_scores_rf = cross_val_score(rf, X_tr, y_tr, cv=cv,
                                   scoring="roc_auc", n_jobs=-1)
    print(f"Logistic Regression ROC-AUC: {cv_scores_lr.round(4)} "
          f"| mean={cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
    print(f"Random Forest       ROC-AUC: {cv_scores_rf.round(4)} "
          f"| mean={cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    pd.DataFrame({
        "model": ["Logistic Regression", "Random Forest"],
        "cv_roc_auc_mean": [round(cv_scores_lr.mean(), 4),
                            round(cv_scores_rf.mean(), 4)],
        "cv_roc_auc_std":  [round(cv_scores_lr.std(), 4),
                            round(cv_scores_rf.std(), 4)],
    }).to_csv(TAB_DIR / "cv_scores.csv", index=False)

    # --- figures
    plot_confusion(y_te, lr_pred, "Logistic Regression — Confusion Matrix",
                   FIG_DIR / "06_confusion_logreg.png")
    plot_confusion(y_te, rf_pred, "Random Forest — Confusion Matrix",
                   FIG_DIR / "07_confusion_rf.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in [("Logistic Regression", lr_proba),
                        ("Random Forest", rf_proba)]:
        fpr, tpr, _ = roc_curve(y_te, proba)
        auc = roc_auc_score(y_te, proba)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_title("ROC Curves — Late-Delivery Prediction")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_roc_curves.png", dpi=140)
    plt.close(fig)

    # --- feature importance from RF
    feat_names = rf.named_steps["pre"].get_feature_names_out()
    importances = rf.named_steps["clf"].feature_importances_
    fi = (pd.DataFrame({"feature": feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(25).reset_index(drop=True))
    fi.to_csv(TAB_DIR / "feature_importance_rf.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=fi, y="feature", x="importance", color="#3b5b92", ax=ax)
    ax.set_title("Random Forest — Top 25 Feature Importances")
    ax.set_xlabel("Gini importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_feature_importance.png", dpi=140)
    plt.close(fig)
    print(f"\n[INFO] Figures written to {FIG_DIR}")

    # --- persist models
    joblib.dump(logreg, MODEL_DIR / "logreg.pkl")
    joblib.dump(rf,     MODEL_DIR / "rf.pkl")
    joblib.dump(pre,    MODEL_DIR / "preprocessor.pkl")
    print(f"[INFO] Models written to {MODEL_DIR}")