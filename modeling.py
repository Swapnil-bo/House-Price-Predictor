"""
================================================================================
 HOUSE PRICE PREDICTOR — PHASE 2: Professional Modeling (Pipelines)
 Dataset : Ames Housing (Kaggle)
 Author  : Production ML Pipeline
================================================================================
"""

# ── 0. IMPORTS ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ── Plot Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#0f0f0f",
                     "axes.facecolor": "#1a1a2e", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white",
                     "text.color": "white", "axes.titlecolor": "white"})
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b6b"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CLEANED DATA FROM PHASE 1
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 | STEP 1 — LOADING CLEANED DATA FROM PHASE 1")
print("=" * 70)

try:
    X_train = pd.read_csv("X_train.csv")
    X_test  = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()   # DataFrame → Series
    y_test  = pd.read_csv("y_test.csv").squeeze()
    print(f"\n  ✔  X_train : {X_train.shape}")
    print(f"  ✔  X_test  : {X_test.shape}")
    print(f"  ✔  y_train : {y_train.shape}  (log-transformed SalePrice)")
    print(f"  ✔  y_test  : {y_test.shape}")
except FileNotFoundError:
    raise FileNotFoundError(
        "\n[ERROR] Phase 1 output files not found.\n"
        "  → Make sure you ran phase1_eda.py successfully first.\n"
        "  → Required files: X_train.csv, X_test.csv, y_train.csv, y_test.csv"
    )

FEATURES = list(X_train.columns)
print(f"\n  Features used ({len(FEATURES)}):")
for i, f in enumerate(FEATURES, 1):
    print(f"    {i:>2}. {f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PIPELINE 1: LINEAR REGRESSION
#          StandardScaler → LinearRegression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 | STEP 2 — PIPELINE 1: LINEAR REGRESSION")
print("=" * 70)
print("""
  Architecture:
  ┌───────────────────┐     ┌───────────────────┐
  │  StandardScaler   │────▶│  LinearRegression  │
  │  (zero mean,      │     │  (OLS)             │
  │   unit variance)  │     │                    │
  └───────────────────┘     └───────────────────┘
""")

linear_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model",  LinearRegression())
])

# — Train ──────────────────────────────────────────────────────────────────────
linear_pipeline.fit(X_train, y_train)
print("  ✔  Linear Pipeline trained successfully")

# — Cross-Validation (5-Fold) ─────────────────────────────────────────────────
print("\n  Running 5-Fold Cross-Validation on training set...")
lr_cv_scores = cross_val_score(
    linear_pipeline, X_train, y_train,
    cv=5, scoring="r2", n_jobs=-1
)
lr_cv_rmse = cross_val_score(
    linear_pipeline, X_train, y_train,
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)

print(f"\n  Linear Regression — Cross-Validation Results (5-Fold):")
print(f"  {'Fold':<8} {'R² Score':>10}  {'RMSE':>10}")
print(f"  {'-'*32}")
for i, (r2, rmse) in enumerate(zip(lr_cv_scores, -lr_cv_rmse), 1):
    print(f"  Fold {i:<3}  {r2:>10.4f}  {rmse:>10.4f}")
print(f"  {'-'*32}")
print(f"  {'Mean':<8} {lr_cv_scores.mean():>10.4f}  {(-lr_cv_rmse).mean():>10.4f}")
print(f"  {'Std':<8} {lr_cv_scores.std():>10.4f}  {(-lr_cv_rmse).std():>10.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PIPELINE 2: POLYNOMIAL REGRESSION (degree=2)
#          StandardScaler → PolynomialFeatures → LinearRegression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 | STEP 3 — PIPELINE 2: POLYNOMIAL REGRESSION (degree=2)")
print("=" * 70)
print("""
  Architecture:
  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐
  │StandardScaler│──▶│PolynomialFeatures│──▶ │  LinearRegression  │
  │             │    │  degree=2        │    │  (on poly features)│
  │             │    │  interaction     │    │                    │
  └─────────────┘    │  terms included  │    └────────────────────┘
                     └──────────────────┘
""")

poly_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("model",  LinearRegression())
])

# — Train ──────────────────────────────────────────────────────────────────────
poly_pipeline.fit(X_train, y_train)

# How many features after polynomial expansion?
n_poly_features = poly_pipeline.named_steps["poly"].n_output_features_
print(f"  ✔  Polynomial Pipeline trained successfully")
print(f"  ✔  Features after degree-2 expansion: {len(FEATURES)} → {n_poly_features}")

# — Cross-Validation (5-Fold) ─────────────────────────────────────────────────
print("\n  Running 5-Fold Cross-Validation on training set...")
poly_cv_scores = cross_val_score(
    poly_pipeline, X_train, y_train,
    cv=5, scoring="r2", n_jobs=-1
)
poly_cv_rmse = cross_val_score(
    poly_pipeline, X_train, y_train,
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)

print(f"\n  Polynomial Regression — Cross-Validation Results (5-Fold):")
print(f"  {'Fold':<8} {'R² Score':>10}  {'RMSE':>10}")
print(f"  {'-'*32}")
for i, (r2, rmse) in enumerate(zip(poly_cv_scores, -poly_cv_rmse), 1):
    print(f"  Fold {i:<3}  {r2:>10.4f}  {rmse:>10.4f}")
print(f"  {'-'*32}")
print(f"  {'Mean':<8} {poly_cv_scores.mean():>10.4f}  {(-poly_cv_rmse).mean():>10.4f}")
print(f"  {'Std':<8} {poly_cv_scores.std():>10.4f}  {(-poly_cv_rmse).std():>10.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL COMPARISON: CV Score Visualization
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 | STEP 4 — CV SCORE COMPARISON PLOT")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("5-Fold Cross-Validation Comparison", fontsize=15,
             color=ACCENT, fontweight="bold", y=1.01)

folds = [f"Fold {i}" for i in range(1, 6)]

# — R² Comparison ─────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#1a1a2e")
x = np.arange(5)
width = 0.35
bars1 = ax.bar(x - width/2, lr_cv_scores,   width, label="Linear",     color=ACCENT,  alpha=0.85)
bars2 = ax.bar(x + width/2, poly_cv_scores,  width, label="Polynomial", color=ACCENT2, alpha=0.85)
ax.axhline(lr_cv_scores.mean(),   color=ACCENT,  linestyle="--", lw=1.2, alpha=0.7)
ax.axhline(poly_cv_scores.mean(), color=ACCENT2, linestyle="--", lw=1.2, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.set_ylabel("R² Score")
ax.set_title("R² Score per Fold", color="white")
ax.legend(facecolor="#1a1a2e", labelcolor="white")
ax.set_ylim(0, 1)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=ACCENT)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=ACCENT2)

# — RMSE Comparison ───────────────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor("#1a1a2e")
bars3 = ax.bar(x - width/2, -lr_cv_rmse,   width, label="Linear",     color=ACCENT,  alpha=0.85)
bars4 = ax.bar(x + width/2, -poly_cv_rmse, width, label="Polynomial", color=ACCENT2, alpha=0.85)
ax.axhline((-lr_cv_rmse).mean(),   color=ACCENT,  linestyle="--", lw=1.2, alpha=0.7)
ax.axhline((-poly_cv_rmse).mean(), color=ACCENT2, linestyle="--", lw=1.2, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.set_ylabel("RMSE (log scale)")
ax.set_title("RMSE per Fold  (lower = better)", color="white")
ax.legend(facecolor="#1a1a2e", labelcolor="white")
for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=ACCENT)
for bar in bars4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=ACCENT2)

plt.tight_layout()
plt.savefig("phase2_01_cv_comparison.png", bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("  ✔  Saved: phase2_01_cv_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE TRAINED PIPELINES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 | STEP 5 — SAVING TRAINED PIPELINES")
print("=" * 70)

joblib.dump(linear_pipeline, "linear_pipeline.pkl")
joblib.dump(poly_pipeline,   "poly_pipeline.pkl")
print("\n  ✔  Saved: linear_pipeline.pkl")
print("  ✔  Saved: poly_pipeline.pkl")
print("\n  These pipelines are fully self-contained:")
print("  → Scaler parameters are frozen from training data")
print("  → No data leakage possible at inference time")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  MODEL              │  CV R² (mean ± std)   │  CV RMSE (mean)   │
  ├──────────────────────────────────────────────────────────────────┤
  │  Linear Regression  │  {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}     │  {(-lr_cv_rmse).mean():.4f}            │
  │  Poly  Regression   │  {poly_cv_scores.mean():.4f} ± {poly_cv_scores.std():.4f}     │  {(-poly_cv_rmse).mean():.4f}            │
  └──────────────────────────────────────────────────────────────────┘

  Input features   : {len(FEATURES)}
  Poly features    : {n_poly_features}  (after degree-2 expansion)
  CV strategy      : 5-Fold on training set only (no test leakage)
  Saved models     : linear_pipeline.pkl  |  poly_pipeline.pkl

  Outputs:
    📊 phase2_01_cv_comparison.png
    💾 linear_pipeline.pkl
    💾 poly_pipeline.pkl

  ⏸  PHASE 2 COMPLETE. Review CV scores and reply 'Phase 3' to continue.
""")
print("=" * 70)