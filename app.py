"""
================================================================================
 HOUSE PRICE PREDICTOR — Streamlit Web App
 Dataset : Ames Housing (Kaggle)
 Run     : streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f0f; color: #ffffff; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #00d4ff33;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid #00d4ff44;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: #00d4ff;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .main-header p {
        color: #aaaacc;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }

    /* Prediction card */
    .price-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px #00d4ff22;
    }
    .price-label {
        color: #aaaacc;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .price-value {
        color: #00d4ff;
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.3rem 0;
    }
    .price-range {
        color: #aaffaa;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h4 { color: #aaaacc; font-size: 0.8rem;
                      text-transform: uppercase; letter-spacing: 1px; margin: 0; }
    .metric-card p  { color: #00d4ff; font-size: 1.5rem;
                      font-weight: 700; margin: 0.3rem 0 0 0; }

    /* Section headers */
    .section-title {
        color: #00d4ff;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 1px solid #00d4ff33;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
    }

    /* Sliders & widgets */
    .stSlider > div > div > div { background: #00d4ff !important; }
    .stSelectbox label, .stSlider label { color: #ccccee !important; }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0066ff);
        color: #000000;
        font-weight: 800;
        font-size: 1.1rem;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    linear = joblib.load("linear_pipeline.pkl")
    poly   = joblib.load("poly_pipeline.pkl")
    return linear, poly

@st.cache_data
def load_training_data():
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    return X_train, y_train

try:
    linear_pipeline, poly_pipeline = load_models()
    X_train, y_train = load_training_data()
    FEATURES = list(X_train.columns)
    models_loaded = True
except FileNotFoundError:
    models_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🏠 House Price Predictor</h1>
    <p>Ames Housing Dataset · Scikit-Learn Pipelines · Linear & Polynomial Regression</p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("""
    ❌ **Model files not found.**

    Make sure these files are in the same folder as `app.py`:
    - `linear_pipeline.pkl`
    - `poly_pipeline.pkl`
    - `X_train.csv`
    - `y_train.csv`

    → Run `phase1_eda.py` then `phase2_modeling.py` first.
    """)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUT CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configure House")
    st.markdown("---")

    # Model selector
    st.markdown('<div class="section-title">🤖 Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Choose Regression Model",
        ["Linear Regression", "Polynomial Regression (degree=2)"],
        help="Polynomial captures non-linear relationships but may overfit."
    )
    st.markdown("---")

    # Build input sliders dynamically from actual feature ranges in training data
    st.markdown('<div class="section-title">🏡 House Features</div>', unsafe_allow_html=True)

    # Friendly display names for features
    FRIENDLY_NAMES = {
        "OverallQual":  "Overall Quality (1–10)",
        "GrLivArea":    "Above-Ground Living Area (sq ft)",
        "GarageCars":   "Garage Capacity (cars)",
        "GarageArea":   "Garage Area (sq ft)",
        "TotalBsmtSF":  "Basement Area (sq ft)",
        "1stFlrSF":     "1st Floor Area (sq ft)",
        "FullBath":     "Full Bathrooms",
        "TotRmsAbvGrd": "Total Rooms Above Ground",
        "YearBuilt":    "Year Built",
        "YearRemodAdd": "Year Remodelled",
        "MasVnrArea":   "Masonry Veneer Area (sq ft)",
        "Fireplaces":   "Number of Fireplaces",
        "BsmtFinSF1":   "Finished Basement Area (sq ft)",
        "LotFrontage":  "Lot Frontage (ft)",
        "WoodDeckSF":   "Wood Deck Area (sq ft)",
        "OpenPorchSF":  "Open Porch Area (sq ft)",
        "LotArea":      "Lot Area (sq ft)",
        "BedroomAbvGr": "Bedrooms Above Ground",
        "HalfBath":     "Half Bathrooms",
        "BsmtFullBath": "Basement Full Bathrooms",
    }

    user_inputs = {}
    for feature in FEATURES:
        col_min  = float(X_train[feature].min())
        col_max  = float(X_train[feature].max())
        col_mean = float(X_train[feature].mean())
        col_med  = float(X_train[feature].median())

        label = FRIENDLY_NAMES.get(feature, feature)

        # Integer features get integer sliders
        if X_train[feature].dtype in ["int64", "int32"] or \
           X_train[feature].nunique() < 20:
            user_inputs[feature] = st.slider(
                label,
                min_value=int(col_min),
                max_value=int(col_max),
                value=int(col_med),
                step=1
            )
        else:
            user_inputs[feature] = st.slider(
                label,
                min_value=round(col_min, 1),
                max_value=round(col_max, 1),
                value=round(col_med, 1),
                step=10.0
            )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

# — Default state message ──────────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    avg_price  = np.expm1(y_train.mean())
    min_price  = np.expm1(y_train.min())
    max_price  = np.expm1(y_train.max())

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Avg Sale Price</h4>
            <p>${avg_price:,.0f}</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 Min Sale Price</h4>
            <p>${min_price:,.0f}</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Max Sale Price</h4>
            <p>${max_price:,.0f}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Adjust the sliders on the left** and click **Predict Price** to get started.")

    # Dataset overview chart
    st.markdown('<div class="section-title">📊 Training Data — Sale Price Distribution</div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    prices_orig = np.expm1(y_train)
    ax.hist(prices_orig, bins=50, color="#00d4ff", alpha=0.7, edgecolor="none")
    ax.axvline(avg_price, color="#ff6b6b", lw=2, linestyle="--", label=f"Mean: ${avg_price:,.0f}")
    ax.set_xlabel("Sale Price (USD)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Ames Housing — Sale Price Distribution (Training Set)",
                 color="#00d4ff", fontweight="bold")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.legend(facecolor="#0f0f0f", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    st.pyplot(fig)

# — Prediction output ──────────────────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([user_inputs])

    # Select model
    pipeline = linear_pipeline if "Linear" in model_choice else poly_pipeline

    # Predict (log scale → original scale)
    log_pred  = pipeline.predict(input_df)[0]
    pred_price = np.expm1(log_pred)

    # Confidence range (±1 RMSE approximation using training residuals)
    train_log_preds = pipeline.predict(X_train)
    train_residuals = np.expm1(y_train) - np.expm1(train_log_preds)
    margin = train_residuals.std()

    low_price  = max(0, pred_price - margin)
    high_price = pred_price + margin

    # ── Prediction Card ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Sale Price</div>
        <div class="price-value">${pred_price:,.0f}</div>
        <div class="price-range">
            📊 Confidence Range: ${low_price:,.0f} — ${high_price:,.0f}
        </div>
        <div style="color:#888; font-size:0.8rem; margin-top:0.5rem;">
            Model: {model_choice}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: gauge + feature summary ──────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-title">📍 Price Positioning</div>',
                    unsafe_allow_html=True)

        # Where does this prediction sit in the training distribution?
        prices_orig  = np.expm1(y_train)
        percentile   = (prices_orig < pred_price).mean() * 100

        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        ax.hist(prices_orig, bins=40, color="#00d4ff", alpha=0.4, edgecolor="none")
        ax.axvline(pred_price, color="#ff6b6b", lw=2.5,
                   linestyle="-", label=f"Your House: ${pred_price:,.0f}")
        ax.axvspan(low_price, high_price, alpha=0.15, color="#ff6b6b", label="Confidence Range")

        ax.set_xlabel("Sale Price (USD)", color="white")
        ax.set_ylabel("Count",            color="white")
        ax.set_title(f"Your house is in the {percentile:.0f}th percentile",
                     color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.legend(facecolor="#0f0f0f", labelcolor="white", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        st.pyplot(fig)

    with col_right:
        st.markdown('<div class="section-title">🏡 Your Input Summary</div>',
                    unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Feature": [FRIENDLY_NAMES.get(f, f) for f in FEATURES],
            "Your Value": [user_inputs[f] for f in FEATURES],
            "Dataset Median": [round(X_train[f].median(), 1) for f in FEATURES]
        })
        summary_df["vs Median"] = summary_df.apply(
            lambda r: "▲ Above" if r["Your Value"] > r["Dataset Median"]
                      else ("▼ Below" if r["Your Value"] < r["Dataset Median"]
                            else "═ Equal"), axis=1
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature":        st.column_config.TextColumn("Feature"),
                "Your Value":     st.column_config.NumberColumn("Your Value"),
                "Dataset Median": st.column_config.NumberColumn("Median"),
                "vs Median":      st.column_config.TextColumn("vs Median"),
            }
        )

    # ── Feature impact bar chart ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Feature Impact on This Prediction</div>',
                unsafe_allow_html=True)

    if "Linear" in model_choice:
        coefs   = linear_pipeline.named_steps["model"].coef_
        scaler  = linear_pipeline.named_steps["scaler"]
        scaled  = scaler.transform(input_df)[0]
        impacts = coefs * scaled

        impact_df = pd.DataFrame({
            "Feature": [FRIENDLY_NAMES.get(f, f) for f in FEATURES],
            "Impact":  impacts
        }).sort_values("Impact", key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(FEATURES) * 0.45)))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        colors = ["#a8ff78" if v > 0 else "#ff6b6b" for v in impact_df["Impact"]]
        ax.barh(impact_df["Feature"], impact_df["Impact"],
                color=colors, edgecolor="none", alpha=0.85)
        ax.axvline(0, color="white", lw=1, linestyle="--")
        ax.set_xlabel("Impact on Predicted Log-Price", color="white")
        ax.set_title("Green = Increases Price  |  Red = Decreases Price",
                     color="#aaaacc", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Feature impact chart is available for Linear Regression model. "
                "Switch to Linear in the sidebar to see it.")

    # ── Quick stats row ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    prices_orig = np.expm1(y_train)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h4>🎯 Prediction</h4><p>${pred_price:,.0f}</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h4>📊 Dataset Avg</h4><p>${prices_orig.mean():,.0f}</p></div>""",
            unsafe_allow_html=True)
    with c3:
        diff = pred_price - prices_orig.mean()
        sign = "+" if diff > 0 else ""
        st.markdown(f"""<div class="metric-card">
            <h4>📐 vs Average</h4><p>{sign}${diff:,.0f}</p></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <h4>📈 Percentile</h4><p>{percentile:.0f}th</p></div>""",
            unsafe_allow_html=True)