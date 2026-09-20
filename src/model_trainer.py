"""
Model Training & Evaluation Pipeline for Upstream Predictive Maintenance.
Trains an ensemble tree classifier, records validation metrics, and serializes artifacts.
"""

import os
from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report

from src.data_generator import generate_telemetry_stream
from src.feature_engineering import prepare_train_test_data, FEATURE_COLUMNS

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "esp_failure_classifier.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

def train_and_evaluate_model():
    print("=" * 70)
    print("TRAINING ESP PREDICTIVE MAINTENANCE CLASSIFIER")
    print("   Author: Jerry A. Nabasu (@JayNabasu)")
    print("=" * 70)

    # 1. Generate & prepare data
    print("\n[Step 1/4] Generating synthetic ESP sensor telemetry across OML fields...")
    df = generate_telemetry_stream(num_wells=18, days=40)
    print(f"  * Generated {len(df):,} hourly telemetry records across 18 downhole wells.")

    print("\n[Step 2/4] Engineering time-series degradation features...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    print(f"  * Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
    print(f"  * Class distribution (Failures in next 72h): {y_train.mean():.1%}")

    # 2. Train Ensemble Model
    print("\n[Step 3/4] Fitting Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. Evaluate Metrics
    print("\n[Step 4/4] Evaluating Model Performance...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = round(float(roc_auc_score(y_test, y_pred_proba)), 4)
    f1 = round(float(f1_score(y_test, y_pred)), 4)
    prec = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)

    print(f"  * ROC-AUC Score : {roc_auc:.4f}")
    print(f"  * Precision     : {prec:.4f}")
    print(f"  * Recall        : {rec:.4f}")
    print(f"  * F1 Score      : {f1:.4f}")

    # Feature Importance
    importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in model.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "metrics": {"roc_auc": roc_auc, "f1": f1, "precision": prec, "recall": rec}
    }, MODEL_PATH)

    metrics_payload = {
        "model_name": "GradientBoosting_ESP_v1.0",
        "evaluation_metrics": {
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec
        },
        "top_features": list(sorted_importances.items())[:5]
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[SUCCESS] Model serialized to {MODEL_PATH}")
    return metrics_payload

if __name__ == "__main__":
    train_and_evaluate_model()
