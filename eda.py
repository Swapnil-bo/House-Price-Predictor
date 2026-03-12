"""
================================================================================
 HOUSE PRICE PREDICTOR — PHASE 1: Advanced EDA & Preprocessing
 Dataset : Ames Housing (Kaggle)
 Author  : Production ML Pipeline
================================================================================
"""

# ── 0. IMPORTS ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ── Plot Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#0f0f0f",
                     "axes.facecolor": "#1a1a2e", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white",
                     "text.color": "white", "axes.titlecolor": "white"})
ACCENT = "#00d4ff"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 1 | STEP 1 — LOADING DATASET")
print("=" * 70)

# ▸ Update this path to your local Kaggle CSV
CSV_PATH = "train.csv"   # ← place train.csv in the same directory
TARGET   = "SalePrice"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"  ✔  Dataset loaded   → shape: {df.shape}")
    print(f"  ✔  Target column    → '{TARGET}'")
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n[ERROR] File not found: '{CSV_PATH}'\n"
        "  → Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data\n"
        "  → Place 'train.csv' in the same directory as this script."
    )

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — INITIAL AUDIT (Missing Values & Duplicates)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 1 | STEP 2 — DATA AUDIT")
print("=" * 70)

# — Duplicates ————————————————————————————————————————————————————————————————
n_dupes = df.duplicated().sum()
print(f"\n  Duplicate rows   : {n_dupes}  {'← DROPPED' if n_dupes > 0 else '(none)'}")
if n_dupes:
    df.drop_duplicates(inplace=True)

# — Missing Value Report ———————————————————————————————————————————————————————
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]
print(f"\n  Columns with missing values: {len(missing)}")
print(f"\n  {'Column':<25} {'Missing %':>10}")
print(f"  {'-'*35}")
for col, pct in missing.items():
    flag = " ◄ DROP" if pct > 45 else ""
    print(f"  {col:<25} {pct:>9.1f}%{flag}")

# — Strategy: Drop columns missing > 45%, impute the rest —————————————————————
HIGH_MISS_THRESH = 45
high_miss_cols = missing[missing > HIGH_MISS_THRESH].index.tolist()
df.drop(columns=high_miss_cols, inplace=True)
print(f"\n  ✔  Dropped {len(high_miss_cols)} high-missingness columns: {high_miss_cols}")

# — Imputation ————————————————————————————————————————————————————————————————
for col in df.columns:
    if df[col].isnull().sum() == 0:
        continue
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)   # categorical → mode
    else:
        df[col].fillna(df[col].median(), inplace=True)     # numeric → median

print(f"  ✔  Remaining nulls after imputation: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION  (keep numeric only for regression baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 1 | STEP 3 — FEATURE SELECTION (Numeric Baseline)")
print("=" * 70)

numeric_df = df.select_dtypes(include=[np.number]).copy()

# Drop Id column if present
if "Id" in numeric_df.columns:
    numeric_df.drop(columns=["Id"], inplace=True)

# Log-transform target to correct right-skew (standard Kaggle practice)
numeric_df[TARGET] = np.log1p(numeric_df[TARGET])
print(f"\n  ✔  Applied log1p() to '{TARGET}' (corrects right skew)")
print(f"  ✔  Numeric features available: {numeric_df.shape[1] - 1}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EDA: Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 1 | STEP 4 — EDA: CORRELATION MATRIX")
print("=" * 70)

corr_matrix = numeric_df.corr()
target_corr = corr_matrix[TARGET].abs().sort_values(ascending=False)

# Keep only top-N correlated features for readability
TOP_N_FEATURES = 15
top_features = target_corr.iloc[1:TOP_N_FEATURES + 1].index.tolist()  # exclude target itself
print(f"\n  Top {TOP_N_FEATURES} features by |correlation| with {TARGET}:")
print(f"\n  {'Feature':<25} {'|Corr|':>8}")
print(f"  {'-'*35}")
for feat in top_features:
    print(f"  {feat:<25} {target_corr[feat]:>8.3f}")

# — Plot Heatmap ——————————————————————————————————————————————————————————————
subset_cols = top_features + [TARGET]
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor("#0f0f0f")
ax.set_facecolor("#1a1a2e")

mask = np.triu(np.ones_like(corr_matrix.loc[subset_cols, subset_cols], dtype=bool))
sns.heatmap(
    corr_matrix.loc[subset_cols, subset_cols],
    mask=mask,
    annot=True, fmt=".2f", linewidths=0.5,
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    annot_kws={"size": 8}, ax=ax,
    cbar_kws={"shrink": 0.8}
)
ax.set_title(f"Correlation Matrix — Top {TOP_N_FEATURES} Features + {TARGET}",
             fontsize=14, pad=15, color=ACCENT, fontweight="bold")
plt.tight_layout()
plt.savefig("phase1_01_correlation_heatmap.png", bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("  ✔  Saved: phase1_01_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EDA: Pairplot (Top 5 features vs target)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 1 | STEP 5 — EDA: PAIRPLOT")
print("=" * 70)

TOP_PAIR = 5
pair_cols = top_features[:TOP_PAIR] + [TARGET]
pair_df   = numeric_df[pair_cols].copy()

g = sns.pairplot(
    pair_df, corner=True, diag_kind="kde",
    plot_kws={"alpha": 0.4, "color": ACCENT, "s": 15},
    diag_kws={"color": ACCENT, "fill": True, "alpha": 0.5}
)
g.figure.suptitle(f"Pairplot — Top {TOP_PAIR} Features vs {TARGET}",
                  y=1.02, fontsize=14, color=ACCENT, fontweight="bold")
g.figure.patch.set_facecolor("#0f0f0f")
for ax in g.axes.flatten():
    if ax:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

plt.savefig("phase1_02_pairplot.png", bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("  ✔  Saved: phase1_02_pairplot.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — STATISTICAL CHECK: Variance Inflation Factor (VIF)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 1 | STEP 6 — MULTICOLLINEARITY CHECK (VIF)")
print("=" * 70)
print("""
  Rule of Thumb:
    VIF = 1        → No correlation
    1 < VIF < 5    → Moderate (acceptable)
    5 ≤ VIF < 10   → High (investigate)
    VIF ≥ 10       → Severe multicollinearity → DROP
""")

VIF_THRESHOLD = 10.0

# Use top correlated features as candidate set
vif_features = top_features.copy()
X_vif = numeric_df[vif_features].copy()

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for all columns in X."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"]     = [variance_inflation_factor(X.values, i)
                           for i in range(X.shape[1])]
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

# Iterative VIF pruning
dropped_vif = []
iteration   = 0
while True:
    iteration += 1
    vif_df = compute_vif(X_vif)
    max_vif_row = vif_df.iloc[0]

    if max_vif_row["VIF"] < VIF_THRESHOLD:
        print(f"  ✔  All VIFs below threshold ({VIF_THRESHOLD}) after {iteration-1} iteration(s).")
        break

    drop_col = max_vif_row["Feature"]
    print(f"  Iter {iteration}: Dropping '{drop_col}' (VIF = {max_vif_row['VIF']:.2f})")
    X_vif.drop(columns=[drop_col], inplace=True)
    dropped_vif.append(drop_col)

# Final VIF table
vif_final = compute_vif(X_vif)
print(f"\n  Final VIF table ({len(vif_final)} features retained):")
print(f"\n  {'Feature':<25} {'VIF':>8}  {'Status'}")
print(f"  {'-'*50}")
for _, row in vif_final.iterrows():
    status = "✔ OK" if row["VIF"] < 5 else "⚠  Moderate"
    print(f"  {row['Feature']:<25} {row['VIF']:>8.2f}  {status}")

print(f"\n  ✔  Dropped for high VIF: {dropped_vif if dropped_vif else 'None'}")

# — VIF Bar Chart ——————————————————————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#0f0f0f")
ax.set_facecolor("#1a1a2e")

colors = ["#ff4444" if v >= 10 else "#ffaa00" if v >= 5 else ACCENT
          for v in vif_final["VIF"]]
bars = ax.barh(vif_final["Feature"], vif_final["VIF"], color=colors, edgecolor="none")
ax.axvline(x=VIF_THRESHOLD, color="#ff4444", linestyle="--", lw=1.5, label=f"Threshold = {VIF_THRESHOLD}")
ax.axvline(x=5,             color="#ffaa00", linestyle="--", lw=1.0, label="Moderate = 5")
ax.set_xlabel("VIF Score", color="white")
ax.set_title("Variance Inflation Factor — Final Feature Set", color=ACCENT,
             fontsize=13, fontweight="bold")
ax.legend(facecolor="#1a1a2e", labelcolor="white")
for bar, val in zip(bars, vif_final["VIF"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", ha="left", color="white", fontsize=9)
plt.tight_layout()
plt.savefig("phase1_03_vif.png", bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("  ✔  Saved: phase1_03_vif.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FINAL FEATURE SET & TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 1 | STEP 7 — TRAIN / TEST SPLIT  (80 / 20)")
print("=" * 70)

FINAL_FEATURES = list(X_vif.columns)
print(f"\n  Final feature set ({len(FINAL_FEATURES)} features):")
for i, f in enumerate(FINAL_FEATURES, 1):
    print(f"    {i:>2}. {f}")

X = numeric_df[FINAL_FEATURES].copy()
y = numeric_df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n  ✔  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
print(f"  ✔  X_test  : {X_test.shape}  |  y_test  : {y_test.shape}")

# — Save split datasets for Phase 2 ——————————————————————————————————————————
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv",  index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv",  index=False)

print("\n  ✔  Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 1 COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Original shape         : {df.shape[0]} rows × {df.shape[1]} cols                  
  │  High-miss cols dropped : {len(high_miss_cols)}                               
  │  VIF-pruned features    : {len(dropped_vif)}                               
  │  Final feature count    : {len(FINAL_FEATURES)}                               
  │  Target transform       : log1p(SalePrice)               
  │  Train set size         : {X_train.shape[0]}                              
  │  Test set size          : {X_test.shape[0]}                               
  └─────────────────────────────────────────────────────────┘

  Outputs:
    📊 phase1_01_correlation_heatmap.png
    📊 phase1_02_pairplot.png
    📊 phase1_03_vif.png
    💾 X_train.csv / X_test.csv / y_train.csv / y_test.csv

  ⏸  PHASE 1 COMPLETE. Review outputs and reply 'Phase 2' to continue.
""")
print("=" * 70)