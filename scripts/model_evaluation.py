"""
Comprehensive Model Performance Evaluation Module
===================================================

Trains four regression models on the Wuxi urban energy dataset and computes:
  - Regression metrics: MAE (kWh), MSE (kWh²), RMSE (kWh), Std Dev of residuals (kWh)
  - Classification metrics (derived by binning predictions into Low / Medium / High
    categories using 33rd/67th percentiles of y_train): Accuracy, Recall (Sensitivity),
    Specificity, Precision, NPV, FPR, FDR, FNR, F1-Score — macro-averaged across
    the three categories using a one-vs-rest approach.

Outputs saved to Data/Processed/:
  - all_model_performance_metrics.csv
  - <model>_confusion_matrix.png  (×4)
  - <model>_confusion_matrix.csv  (×4)

Run from the project root:
    python scripts/model_evaluation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ============================================================
# CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Processed"
DATASET_PATH = DATA_DIR / "ml_energy_dataset.csv"

FEATURES = [
    "building_count",
    "total_footprint_m2",
    "mean_height_m",
    "building_density",
    "area_residential_m2",
    "area_commercial_m2",
    "area_office_m2",
    "area_education_m2",
    "area_industrial_m2",
    "area_other_m2",
    "floor_area_total_m2",
    "single_family_m2",
    "multi_family_m2",
    "facility_sales_m2",
    "facility_office_m2",
    "facility_neighborhood_m2",
    "facility_education_m2",
    "facility_industrial_m2",
    "parks_green_m2",
    "water_area_m2",
    "road_area_m2",
    "subway_influence_m2",
    "bus_routes_cnt",
]

TARGET = "total_kwh"

SEED = 42
TEST_SIZE = 0.2

CATEGORY_LABELS = ["Low", "Medium", "High"]
CATEGORY_MAP = {"Low": 0, "Medium": 1, "High": 2}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def categorise(values: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    """Map continuous values into Low (0), Medium (1), High (2)."""
    cats = np.zeros(len(values), dtype=int)
    cats[values >= thresholds[0]] = 1
    cats[values >= thresholds[1]] = 2
    return cats


def one_vs_rest_metrics(cm: np.ndarray) -> dict[str, float]:
    """
    Given a K×K confusion matrix, compute per-class TP/TN/FP/FN using a
    one-vs-rest binarisation, then return macro-averaged classification metrics.

    Metrics: Accuracy, Recall (Sensitivity), Specificity, Precision,
             NPV, FPR, FDR, FNR, F1-Score.
    """
    n_classes = cm.shape[0]
    total = cm.sum()

    acc_list, rec_list, spec_list = [], [], []
    prec_list, npv_list = [], []
    fpr_list, fdr_list, fnr_list, f1_list = [], [], [], []

    for k in range(n_classes):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp

        # Accuracy (overall, same for every class but included for completeness)
        acc = (tp + tn) / total if total > 0 else 0.0

        # Recall / Sensitivity
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # False Discovery Rate
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

        # False Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # F1-Score
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        acc_list.append(acc)
        rec_list.append(rec)
        spec_list.append(spec)
        prec_list.append(prec)
        npv_list.append(npv)
        fpr_list.append(fpr)
        fdr_list.append(fdr)
        fnr_list.append(fnr)
        f1_list.append(f1)

    return {
        "Accuracy": float(np.mean(acc_list)),
        "Recall_Sensitivity": float(np.mean(rec_list)),
        "Specificity": float(np.mean(spec_list)),
        "Precision": float(np.mean(prec_list)),
        "NPV": float(np.mean(npv_list)),
        "FPR": float(np.mean(fpr_list)),
        "FDR": float(np.mean(fdr_list)),
        "FNR": float(np.mean(fnr_list)),
        "F1_Score": float(np.mean(f1_list)),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path,
) -> None:
    """Save a styled confusion-matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=11)

    ax.set_ylabel("Actual Category", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Category", fontsize=12, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def main() -> None:
    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    if not DATASET_PATH.exists():
        sys.exit(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        sys.exit(f"Missing feature columns: {missing}")
    if TARGET not in df.columns:
        sys.exit(f"Target column '{TARGET}' not found.")

    X = df[FEATURES].values
    y = df[TARGET].values
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # ----------------------------------------------------------
    # 2. Train / test split
    # ----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ----------------------------------------------------------
    # 3. Derive classification thresholds (33rd / 67th pctile of y_train)
    # ----------------------------------------------------------
    p33 = float(np.percentile(y_train, 33))
    p67 = float(np.percentile(y_train, 67))
    thresholds = (p33, p67)
    print(f"Category thresholds (from y_train): Low < {p33:,.0f}  |  "
          f"Medium [{p33:,.0f} – {p67:,.0f})  |  High ≥ {p67:,.0f} kWh")

    y_test_cat = categorise(y_test, thresholds)

    # ----------------------------------------------------------
    # 4. Define models
    # ----------------------------------------------------------
    models: dict[str, object] = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=SEED),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, random_state=SEED, verbosity=0
        )
    else:
        print("⚠  xgboost not installed — skipping XGBoost model")

    # ----------------------------------------------------------
    # 5. Train, predict, evaluate
    # ----------------------------------------------------------
    results: list[dict] = []
    file_prefixes = {
        "Random Forest": "random_forest",
        "Extra Trees": "extra_trees",
        "Gradient Boosting": "gradient_boosting",
        "XGBoost": "xgboost",
    }

    for name, model in models.items():
        print(f"\n▶ Training {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Sanity check
        assert len(y_test) == len(y_pred), "Actual/Predicted length mismatch!"

        # --- Regression metrics (continuous values) ---
        residuals = y_test - y_pred
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        std_residuals = float(np.std(residuals, ddof=1))

        # --- Classification metrics (derived categories) ---
        y_pred_cat = categorise(y_pred, thresholds)
        cm = confusion_matrix(y_test_cat, y_pred_cat, labels=[0, 1, 2])
        cls_metrics = one_vs_rest_metrics(cm)

        # Combine
        row = {"Model": name}
        # Confusion matrix flattened for reference (not in final table, but saved separately)
        row.update(cls_metrics)
        row["Std_Dev_Residuals_kWh"] = std_residuals
        row["MAE_kWh"] = mae
        row["MSE_kWh2"] = mse
        row["RMSE_kWh"] = rmse
        results.append(row)

        # --- Save confusion matrix ---
        prefix = file_prefixes[name]
        plot_confusion_matrix(
            cm,
            labels=CATEGORY_LABELS,
            title=f"{name} — Confusion Matrix",
            save_path=DATA_DIR / f"{prefix}_confusion_matrix.png",
        )
        cm_df = pd.DataFrame(cm, index=CATEGORY_LABELS, columns=CATEGORY_LABELS)
        cm_df.index.name = "Actual"
        cm_csv_path = DATA_DIR / f"{prefix}_confusion_matrix.csv"
        cm_df.to_csv(cm_csv_path)
        print(f"  Saved: {cm_csv_path.name}")

        # Print summary
        print(f"  MAE={mae:,.2f} kWh  |  MSE={mse:,.2f} kWh²  |  "
              f"RMSE={rmse:,.2f} kWh  |  StdDev(residuals)={std_residuals:,.2f} kWh")
        print(f"  Accuracy={cls_metrics['Accuracy']:.4f}  |  "
              f"F1={cls_metrics['F1_Score']:.4f}  |  "
              f"Precision={cls_metrics['Precision']:.4f}  |  "
              f"Recall={cls_metrics['Recall_Sensitivity']:.4f}")

    # ----------------------------------------------------------
    # 6. Save combined performance table
    # ----------------------------------------------------------
    metrics_df = pd.DataFrame(results)
    # Reorder columns for readability
    col_order = [
        "Model",
        "Accuracy",
        "Recall_Sensitivity",
        "Specificity",
        "Precision",
        "NPV",
        "FPR",
        "FDR",
        "FNR",
        "F1_Score",
        "Std_Dev_Residuals_kWh",
        "MAE_kWh",
        "MSE_kWh2",
        "RMSE_kWh",
    ]
    metrics_df = metrics_df[col_order]
    out_path = DATA_DIR / "all_model_performance_metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"\n✅ Performance table saved: {out_path}")
    print(metrics_df.to_string(index=False))

    print("\nDone! All outputs saved to Data/Processed/")


if __name__ == "__main__":
    main()
