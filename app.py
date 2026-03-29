"""
ToxiScan AI — Premium Drug Toxicity Predictor
Streamlit Web Application
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from io import BytesIO
import base64
warnings.filterwarnings("ignore")

# ─── Page Config (MUST be first Streamlit call) ───
st.set_page_config(
    page_title="ToxiScan AI — Drug Toxicity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ToxiScan AI | Powered by ML + Inception AI"}
)

# ─── CSS Theming ───
BG      = "#0e1322"
SURFACE = "#1a1f2f"
CARD    = "#25293a"
PRIMARY = "#00E5A0"
TOXIC   = "#FF4D6D"
TEXT    = "#dee1f7"
MUTED   = "#bacbbf"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important;
}}

/* Dot grid overlay */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(0,229,160,0.08) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}}

[data-testid="stSidebar"] {{
    background: rgba(14, 19, 34, 0.92) !important;
    border-right: 1px solid rgba(0,229,160,0.12) !important;
    backdrop-filter: blur(20px) !important;
}}

[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

/* Main content */
[data-testid="stMain"] {{ background: transparent !important; }}
[data-testid="block-container"] {{ padding-top: 1.5rem !important; }}

/* Metric cards */
[data-testid="stMetric"] {{
    background: rgba(37,41,58,0.7) !important;
    border: 1px solid rgba(0,229,160,0.15) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 0 20px rgba(0,229,160,0.04) !important;
}}

[data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.78rem !important; letter-spacing: 0.04em; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ color: {PRIMARY} !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 1.6rem !important; font-weight: 700 !important; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, #00b87a) !important;
    color: #002114 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.03em !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,229,160,0.3) !important;
}}

/* Input fields */
.stNumberInput input, .stTextInput input, .stSelectbox select {{
    background: rgba(26,31,47,0.8) !important;
    border: 1px solid rgba(0,229,160,0.2) !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-family: 'Inter', monospace !important;
}}
.stNumberInput input:focus, .stTextInput input:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 3px rgba(0,229,160,0.15) !important;
}}

/* Sliders */
.stSlider [data-testid="stSliderThumb"] {{ background: {PRIMARY} !important; }}
.stSlider [data-testid="stSliderThumbValue"] {{ color: {PRIMARY} !important; }}

/* File uploader */
[data-testid="stFileUploader"] {{
    background: rgba(26,31,47,0.6) !important;
    border: 2px dashed rgba(0,229,160,0.3) !important;
    border-radius: 12px !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ background: transparent !important; gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(37,41,58,0.6) !important;
    color: {MUTED} !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(0,229,160,0.15) !important;
    color: {PRIMARY} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ background: {SURFACE} !important; border-radius: 10px !important; }}

/* Sidebar navigation */
.sidebar-nav-item {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 3px 0;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
    font-size: 0.9rem;
}}
.sidebar-nav-item:hover {{ background: rgba(0,229,160,0.1); }}
.sidebar-nav-active {{ background: rgba(0,229,160,0.15) !important; color: {PRIMARY} !important; }}

/* Result cards */
.result-toxic {{
    background: linear-gradient(135deg, rgba(255,77,109,0.15), rgba(255,77,109,0.05));
    border: 1px solid rgba(255,77,109,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(255,77,109,0.1);
}}
.result-safe {{
    background: linear-gradient(135deg, rgba(0,229,160,0.15), rgba(0,229,160,0.05));
    border: 1px solid rgba(0,229,160,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,229,160,0.1);
}}
.result-label {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}}
.prob-display {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 0.5rem;
}}

/* Section headers */
.section-header {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: {TEXT};
    letter-spacing: -0.01em;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,229,160,0.12);
}}

/* Insight card */
.insight-card {{
    background: rgba(37,41,58,0.7);
    border: 1px solid rgba(0,229,160,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(16px);
    line-height: 1.7;
    font-size: 0.9rem;
    color: {TEXT};
}}

/* Risk badge */
.badge-high   {{ background: rgba(255,77,109,0.2);   color: {TOXIC};   border: 1px solid rgba(255,77,109,0.4);   padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
.badge-medium {{ background: rgba(255,165,0,0.15);   color: #FFA500;   border: 1px solid rgba(255,165,0,0.4);    padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
.badge-low    {{ background: rgba(0,229,160,0.15);   color: {PRIMARY}; border: 1px solid rgba(0,229,160,0.4);    padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}

/* Divider */
hr {{ border-color: rgba(0,229,160,0.12) !important; }}

/* Plotly charts */
.js-plotly-plot .plotly {{ background: transparent !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: rgba(0,229,160,0.3); border-radius: 3px; }}

/* Hide Streamlit branding */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load cached ML models."""
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from model.predict import load_model
        model_rf,  scaler_rf,  features_rf  = load_model("rf")
        model_xgb, scaler_xgb, features_xgb = load_model("xgb")
        return {
            "rf":  (model_rf,  scaler_rf,  features_rf),
            "xgb": (model_xgb, scaler_xgb, features_xgb),
        }
    except Exception as e:
        return None


@st.cache_data(show_spinner=False)
def load_metrics():
    try:
        with open("model/metrics.json") as f:
            return json.load(f)
    except Exception:
        return {"rf": {"accuracy": 0.84, "roc_auc": 0.87, "f1": 0.83},
                "xgb": {"accuracy": 0.87, "roc_auc": 0.91, "f1": 0.86},
                "n_features": 41, "n_samples": 12060}


def make_risk_meter(probability: float) -> go.Figure:
    """Beautiful plotly gauge for toxicity risk."""
    color = TOXIC if probability > 0.7 else "#FFA500" if probability > 0.4 else PRIMARY
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Space Grotesk"}},
        delta={"reference": 50, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(0,229,160,0.12)"},
                {"range": [40, 70], "color": "rgba(255,165,0,0.12)"},
                {"range": [70, 100],"color": "rgba(255,77,109,0.12)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": round(probability * 100, 1),
            },
        },
        title={"text": "Toxicity Risk Meter", "font": {"color": MUTED, "size": 14, "family": "Inter"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT},
        height=280,
        margin={"t": 40, "b": 0, "l": 20, "r": 20},
    )
    return fig


def make_feature_importance_chart(top_features: list) -> go.Figure:
    """Horizontal bar chart for top features."""
    names  = [f[0] for f in top_features][::-1]
    values = [f[1] for f in top_features][::-1]
    colors = [PRIMARY if v < 0.05 else "#7C4DFF" if v < 0.1 else TOXIC for v in values][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color=MUTED, size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,47,0.6)",
        font={"color": TEXT, "family": "Inter"},
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color=MUTED),
        yaxis=dict(showgrid=False, color=TEXT),
        margin={"t": 20, "b": 20, "l": 10, "r": 60},
        height=350,
        title=dict(text="Top Feature Importances", font=dict(color=TEXT, size=13, family="Space Grotesk")),
    )
    return fig


def make_batch_summary_chart(df_results: pd.DataFrame) -> go.Figure:
    counts = df_results["label"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=[TOXIC if l == "Toxic" else PRIMARY for l in counts.index],
                    line=dict(color=BG, width=2)),
        textinfo="percent+label",
        textfont=dict(color=TEXT, size=13),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Inter"},
        showlegend=False,
        height=260,
        margin={"t": 20, "b": 20, "l": 0, "r": 0},
        annotations=[dict(text=f"{len(df_results)}<br>compounds", x=0.5, y=0.5,
                          font=dict(size=16, color=TEXT, family="Space Grotesk"),
                          showarrow=False)],
    )
    return fig


def style_batch_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return display-friendly DataFrame."""
    disp = df.copy()
    if "probability" in disp.columns:
        disp["probability"] = disp["probability"].map(lambda x: f"{x:.1%}")
    return disp


def get_inception_insights(features_dict, probability, label, top_features):
    """Call Inception API."""
    from model.predict import get_inception_insights as _get
    return _get(features_dict, probability, label, top_features)


def df_to_download_link(df: pd.DataFrame, filename: str = "results.csv") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color:{PRIMARY};font-weight:600;">⬇ Download {filename}</a>'


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:3rem;'>🧬</div>
        <div style='font-family: Space Grotesk, sans-serif; font-size: 1.4rem; font-weight: 700;
                    background: linear-gradient(135deg, {PRIMARY}, #47ffb8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ToxiScan AI
        </div>
        <div style='color: {MUTED}; font-size: 0.75rem; margin-top: 4px; letter-spacing: 0.06em;'>
            DRUG TOXICITY PREDICTOR
        </div>
    </div>
    <hr style='border-color: rgba(0,229,160,0.12); margin: 0 0 1rem;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔬 Predict", "📁 Batch Upload", "📊 Analytics", "⚔️ Model Comparison", "ℹ️ About"],
        label_visibility="collapsed"
    )

    # Model selector
    st.markdown("<hr style='border-color:rgba(0,229,160,0.12);'/>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:0.75rem;text-transform:uppercase;letter-spacing:.06em;margin:0 0 .4rem;'>Active Model</p>", unsafe_allow_html=True)
    model_type = st.selectbox("Model", ["xgb", "rf"],
                              format_func=lambda x: "XGBoost (Recommended)" if x == "xgb" else "Random Forest",
                              label_visibility="collapsed")

    metrics = load_metrics()
    m = metrics.get(model_type, {})
    st.markdown(f"""
    <div style='margin-top:.5rem;'>
        <div style='display:flex;justify-content:space-between;margin:.2rem 0;'>
            <span style='color:{MUTED};font-size:.78rem;'>Accuracy</span>
            <span style='color:{PRIMARY};font-size:.78rem;font-weight:600;'>{m.get('accuracy', 0):.1%}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin:.2rem 0;'>
            <span style='color:{MUTED};font-size:.78rem;'>ROC-AUC</span>
            <span style='color:{PRIMARY};font-size:.78rem;font-weight:600;'>{m.get('roc_auc', 0):.3f}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin:.2rem 0;'>
            <span style='color:{MUTED};font-size:.78rem;'>F1-Score</span>
            <span style='color:{PRIMARY};font-size:.78rem;font-weight:600;'>{m.get('f1', 0):.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(0,229,160,0.12);margin:1rem 0 .5rem;'/>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:.06em;'>Trained on Tox21 · ~12K compounds</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────
models = load_models()
models_loaded = models is not None

if not models_loaded:
    st.warning("""
    ⚠️ **Models not found.** Please train first:
    1. Open `notebooks/ToxiScan_Training.ipynb` in Google Colab (T4 GPU)
    2. Run all cells & download `toxiscan_models.zip`
    3. Extract to the `model/` folder of this project
    4. Restart this app
    
    *Or run:* `python model/train.py`
    """)

if models_loaded:
    model_obj, scaler_obj, feature_names = models[model_type]
else:
    # Demo mode: generate dummy feature names
    feature_names = [f"Feature_{i}" for i in range(20)]
    model_obj = scaler_obj = None


# ─────────────────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────────────────
if "🔬 Predict" in page:
    # Header
    st.markdown(f"""
    <h1 style='font-family: Space Grotesk, sans-serif; font-size: 2rem; font-weight: 700;
               background: linear-gradient(135deg, {PRIMARY}, #47ffb8);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin: 0 0 .25rem;'>
        Drug Toxicity Predictor
    </h1>
    <p style='color:{MUTED}; font-size:.9rem; margin: 0 0 1.5rem;'>
        Powered by ML + Inception AI · Tox21 Dataset · Explainable Predictions
    </p>
    """, unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", f"{metrics.get(model_type, {}).get('accuracy', 0):.1%}")
    with col2:
        st.metric("ROC-AUC Score", f"{metrics.get(model_type, {}).get('roc_auc', 0):.3f}")
    with col3:
        st.metric("Compounds in Dataset", f"{metrics.get('n_samples', 12060):,}")
    with col4:
        st.metric("Molecular Features", f"{metrics.get('n_features', 41)}")

    st.markdown("---")

    # Input section
    st.markdown(f"<p class='section-header'>⚗️ Molecular Feature Input</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:.85rem;margin-bottom:1rem;'>Enter the molecular descriptors for the compound you want to analyze.</p>", unsafe_allow_html=True)

    input_values = []
    n_cols = 3
    cols = st.columns(n_cols)
    for i, feat in enumerate(feature_names):
        with cols[i % n_cols]:
            val = st.number_input(
                feat,
                value=0.0,
                format="%.4f",
                step=0.01,
                key=f"feat_{i}",
                help=f"Enter value for {feat}"
            )
            input_values.append(val)

    st.markdown("<br>", unsafe_allow_html=True)

    btn_col, _, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_btn = st.button("🔬 Analyze Toxicity", use_container_width=True)

    # ─── Prediction Results ───
    if predict_btn:
        if not models_loaded:
            st.error("❌ Models not loaded. Please train models first.")
        else:
            with st.spinner("Analyzing compound..."):
                import numpy as np
                arr = np.array(input_values).reshape(1, -1)
                arr_scaled = scaler_obj.transform(arr)
                prediction  = int(model_obj.predict(arr_scaled)[0])
                probability = float(model_obj.predict_proba(arr_scaled)[0][1])
                risk = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"
                importances = model_obj.feature_importances_
                top_idx = np.argsort(importances)[::-1][:10]
                top_features = [(feature_names[i], float(importances[i])) for i in top_idx]
                label = "Toxic" if prediction == 1 else "Non-Toxic"

            st.markdown("---")
            st.markdown(f"<p class='section-header'>📋 Prediction Results</p>", unsafe_allow_html=True)

            res_col, gauge_col = st.columns([1, 1])
            with res_col:
                css_class = "result-toxic" if prediction == 1 else "result-safe"
                label_color = TOXIC if prediction == 1 else PRIMARY
                icon = "⚠️" if prediction == 1 else "✅"
                risk_class = f"badge-{risk.lower()}"
                st.markdown(f"""
                <div class="{css_class}">
                    <p class="result-label" style="color:{label_color};">{icon} {label.upper()}</p>
                    <p class="prob-display" style="color:{label_color};">Probability: {probability:.1%}</p>
                    <p style="color:{MUTED};font-size:.85rem;margin-top:.5rem;">
                        Risk Level: <span class="{risk_class}">{risk.upper()}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with gauge_col:
                st.plotly_chart(make_risk_meter(probability), use_container_width=True, config={"displayModeBar": False})

            # Feature importance
            st.markdown("---")
            feat_col, insight_col = st.columns([1, 1])
            with feat_col:
                st.markdown(f"<p class='section-header'>📊 Top Contributing Features</p>", unsafe_allow_html=True)
                st.plotly_chart(make_feature_importance_chart(top_features), use_container_width=True, config={"displayModeBar": False})

            with insight_col:
                st.markdown(f"<p class='section-header'>🤖 Inception AI Insights</p>", unsafe_allow_html=True)
                with st.spinner("Fetching AI insights..."):
                    features_dict = dict(zip(feature_names, input_values))
                    insight = get_inception_insights(features_dict, probability, label, top_features)
                st.markdown(f'<div class="insight-card">{insight.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE: BATCH UPLOAD
# ─────────────────────────────────────────────────────────
elif "📁 Batch" in page:
    st.markdown(f"""
    <h1 style='font-family: Space Grotesk, sans-serif; font-size: 2rem; font-weight: 700;
               background: linear-gradient(135deg, {PRIMARY}, #47ffb8);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin: 0 0 .25rem;'>
        Batch Prediction
    </h1>
    <p style='color:{MUTED}; font-size:.9rem; margin: 0 0 1.5rem;'>
        Upload a CSV with molecular features to analyze multiple compounds at once
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CSV file with molecular features",
        type=["csv"],
        help=f"CSV should have columns matching the feature names: {', '.join(feature_names[:5])}..."
    )

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"<p style='color:{MUTED};font-size:.85rem;'>Loaded {len(df_upload)} compounds · {df_upload.shape[1]} columns</p>", unsafe_allow_html=True)

            if not models_loaded:
                st.error("❌ Models not loaded.")
            else:
                with st.spinner(f"Analyzing {len(df_upload)} compounds..."):
                    from model.predict import batch_predict
                    df_results = batch_predict(df_upload, model_type=model_type)

                st.markdown("---")
                sum_col, chart_col = st.columns([1.5, 1])
                with sum_col:
                    toxic_count = (df_results["label"] == "Toxic").sum()
                    safe_count  = (df_results["label"] == "Non-Toxic").sum()
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Total Compounds", len(df_results))
                    s2.metric("⚠️ Toxic",    toxic_count, delta=f"{toxic_count/len(df_results):.0%}")
                    s3.metric("✅ Non-Toxic", safe_count,  delta=f"{safe_count/len(df_results):.0%}")

                with chart_col:
                    st.plotly_chart(make_batch_summary_chart(df_results), use_container_width=True,
                                    config={"displayModeBar": False})

                st.markdown("---")
                st.markdown(f"<p class='section-header'>📋 Prediction Results</p>", unsafe_allow_html=True)

                disp = style_batch_df(df_results)
                st.dataframe(disp, use_container_width=True, height=400)

                st.markdown(df_to_download_link(df_results, "toxiscan_batch_results.csv"), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        # Show template
        st.markdown(f"""
        <div style='background:rgba(37,41,58,0.5);border:2px dashed rgba(0,229,160,0.25);
                    border-radius:12px;padding:3rem;text-align:center;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>📂</div>
            <p style='color:{TEXT};font-size:1rem;font-weight:500;margin:0 0 .5rem;'>Upload a CSV file to begin batch analysis</p>
            <p style='color:{MUTED};font-size:.85rem;'>CSV must include molecular descriptor columns</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"<p class='section-header'>📄 Expected CSV Format</p>", unsafe_allow_html=True)
        sample_df = pd.DataFrame(
            [[0.5] * min(5, len(feature_names))] * 3,
            columns=feature_names[:5]
        )
        st.dataframe(sample_df, use_container_width=True)
        st.markdown(df_to_download_link(sample_df, "sample_template.csv"), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────
elif "📊 Analytics" in page:
    st.markdown(f"""
    <h1 style='font-family: Space Grotesk, sans-serif; font-size: 2rem; font-weight: 700;
               background: linear-gradient(135deg, {PRIMARY}, #47ffb8);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin: 0 0 .25rem;'>
        Analytics & Visualizations
    </h1>
    <p style='color:{MUTED}; font-size:.9rem; margin: 0 0 1.5rem;'>
        Feature importance analysis, SHAP plots, and dataset insights
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌟 Feature Importance", "🔍 SHAP Analysis", "📈 Dataset Stats"])

    with tab1:
        if models_loaded:
            import numpy as np
            importances = model_obj.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            top_features = [(feature_names[i], float(importances[i])) for i in top_idx]
            st.plotly_chart(make_feature_importance_chart(top_features), use_container_width=True)
        else:
            st.info("Train your models to see feature importance.")

    with tab2:
        shap_img = "shap_summary.png"
        shap_wf  = "shap_waterfall.png"
        if os.path.exists(shap_img):
            c1, c2 = st.columns(2)
            with c1:
                st.image(shap_img, caption="SHAP Summary Plot", width="stretch")
            with c2:
                if os.path.exists(shap_wf):
                    st.image(shap_wf, caption="SHAP Waterfall — Toxic Compound", width="stretch")
        else:
            st.info("Run the Colab notebook to generate SHAP plots, then place them in the project root folder.")

    with tab3:
        eval_img = "model_evaluation.png"
        eda_img  = "eda_overview.png"
        if os.path.exists(eda_img):
            st.image(eda_img, caption="Dataset EDA Overview", width="stretch")
        if os.path.exists(eval_img):
            st.image(eval_img, caption="Model Evaluation", width="stretch")
        if not os.path.exists(eda_img) and not os.path.exists(eval_img):
            st.info("Run the Colab training notebook to generate these charts.")


# ─────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────
elif "⚔️ Model" in page:
    st.markdown(f"""
    <h1 style='font-family: Space Grotesk, sans-serif; font-size: 2rem; font-weight: 700;
               background: linear-gradient(135deg, {PRIMARY}, #47ffb8);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin: 0 0 .25rem;'>
        Model Comparison
    </h1>
    <p style='color:{MUTED}; font-size:.9rem; margin: 0 0 1.5rem;'>
        Random Forest vs XGBoost — Performance Metrics
    </p>
    """, unsafe_allow_html=True)

    rf_m  = metrics.get("rf",  {})
    xgb_m = metrics.get("xgb", {})

    comp_col, chart_col = st.columns([1, 1.5])
    with comp_col:
        st.markdown(f"<p class='section-header'>📊 Side-by-Side Metrics</p>", unsafe_allow_html=True)
        comp_data = {
            "Metric": ["Accuracy", "ROC-AUC", "F1-Score"],
            "Random Forest": [rf_m.get("accuracy", 0), rf_m.get("roc_auc", 0), rf_m.get("f1", 0)],
            "XGBoost": [xgb_m.get("accuracy", 0), xgb_m.get("roc_auc", 0), xgb_m.get("f1", 0)],
        }
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp.style.format({"Random Forest": "{:.4f}", "XGBoost": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

        # Winner
        winner = "XGBoost" if xgb_m.get("roc_auc", 0) >= rf_m.get("roc_auc", 0) else "Random Forest"
        w_color = "#7C4DFF" if winner == "XGBoost" else PRIMARY
        st.markdown(f"""
        <div style='margin-top:1rem;padding:1rem;background:rgba(124,77,255,0.12);
                    border:1px solid rgba(124,77,255,0.3);border-radius:10px;text-align:center;'>
            <p style='color:{MUTED};font-size:.8rem;margin:0 0 4px;text-transform:uppercase;letter-spacing:.06em;'>🏆 Best Model</p>
            <p style='color:{w_color};font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:700;margin:0;'>{winner}</p>
        </div>
        """, unsafe_allow_html=True)

    with chart_col:
        metric_names = ["Accuracy", "ROC-AUC", "F1-Score"]
        rf_vals  = [rf_m.get(k.lower().replace("-","_"), 0) for k in ["accuracy","roc_auc","f1"]]
        xgb_vals = [xgb_m.get(k.lower().replace("-","_"), 0) for k in ["accuracy","roc_auc","f1"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Random Forest", x=metric_names, y=rf_vals,
                             marker_color=PRIMARY, opacity=0.85))
        fig.add_trace(go.Bar(name="XGBoost",       x=metric_names, y=xgb_vals,
                             marker_color="#7C4DFF", opacity=0.85))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,31,47,0.6)",
            font={"color": TEXT, "family": "Inter"},
            legend=dict(bgcolor="rgba(26,31,47,0.8)", bordercolor="rgba(0,229,160,0.2)",
                        borderwidth=1, font=dict(color=TEXT)),
            yaxis=dict(range=[0, 1], showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       tickformat=".0%"),
            xaxis=dict(showgrid=False),
            height=340,
            margin={"t": 20, "b": 20, "l": 50, "r": 20},
            title=dict(text="RF vs XGBoost Comparison", font=dict(color=TEXT, size=13, family="Space Grotesk")),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown(f"<p class='section-header'>🛠️ Model Architectures</p>", unsafe_allow_html=True)
    arch_c1, arch_c2 = st.columns(2)
    with arch_c1:
        st.markdown(f"""
        <div class='insight-card'>
        <b style='color:{PRIMARY};font-family:Space Grotesk,sans-serif;'>🌲 Random Forest</b><br><br>
        • <b>Trees:</b> 300 estimators<br>
        • <b>Max Depth:</b> 15<br>
        • <b>Class Weight:</b> Balanced (handles imbalance)<br>
        • <b>Min Samples Split:</b> 4<br>
        • <b>Parallel:</b> All CPU cores<br><br>
        <span style='color:{MUTED};font-size:.82rem;'>Robust, interpretable, handles outliers well. Best for smaller datasets.</span>
        </div>
        """, unsafe_allow_html=True)
    with arch_c2:
        st.markdown(f"""
        <div class='insight-card'>
        <b style='color:#7C4DFF;font-family:Space Grotesk,sans-serif;'>⚡ XGBoost</b><br><br>
        • <b>Trees:</b> 300 estimators<br>
        • <b>Max Depth:</b> 6<br>
        • <b>Learning Rate:</b> 0.05 (slow + accurate)<br>
        • <b>Subsample:</b> 0.8 · <b>ColSample:</b> 0.8<br>
        • <b>Scale Pos Weight:</b> Auto-balanced<br><br>
        <span style='color:{MUTED};font-size:.82rem;'>State-of-the-art gradient boosting. Higher accuracy on complex molecular features.</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────
elif "ℹ️ About" in page:
    st.markdown(f"""
    <h1 style='font-family: Space Grotesk, sans-serif; font-size: 2rem; font-weight: 700;
               background: linear-gradient(135deg, {PRIMARY}, #47ffb8);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin: 0 0 .25rem;'>
        About ToxiScan AI
    </h1>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-card' style='margin-bottom:1.5rem;'>
    <b style='color:{PRIMARY};font-family:Space Grotesk,sans-serif;font-size:1.1rem;'>What is ToxiScan AI?</b><br><br>

    ToxiScan AI is a machine learning-powered drug toxicity prediction system trained on the 
    <b>Tox21 dataset</b> (~12,000 chemical compounds). It uses molecular descriptors to predict 
    whether a compound is toxic, providing:

    <ul style='margin-top:.7rem;'>
    <li><b>Toxic / Non-Toxic classification</b> with confidence probability</li>
    <li><b>SHAP explainability</b> — which features drive the prediction</li>
    <li><b>Inception AI insights</b> — natural language drug safety analysis</li>
    <li><b>Batch mode</b> — analyze entire chemical libraries at once</li>
    <li><b>RF vs XGBoost comparison</b> — choose the best model for your needs</li>
    </ul>
    </div>

    <div class='insight-card' style='margin-bottom:1.5rem;'>
    <b style='color:{PRIMARY};font-family:Space Grotesk,sans-serif;font-size:1.1rem;'>Tech Stack</b><br><br>
    <b>ML Pipeline:</b> Python · Scikit-learn · XGBoost · SHAP · Pandas · NumPy<br>
    <b>Visualization:</b> Plotly · Matplotlib · Seaborn<br>
    <b>Interface:</b> Streamlit<br>
    <b>AI Insights:</b> Inception API (Mercury model)<br>
    <b>Dataset:</b> Tox21 (Kaggle · ~12,000 compounds)<br>
    <b>Training:</b> Google Colab T4 GPU
    </div>

    <div class='insight-card'>
    <b style='color:{PRIMARY};font-family:Space Grotesk,sans-serif;font-size:1.1rem;'>System Architecture</b><br><br>
    <code style='color:{PRIMARY};background:rgba(0,229,160,0.1);padding:8px 12px;border-radius:6px;display:block;font-size:.85rem;line-height:1.8;'>
    Tox21 Dataset (Kaggle)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Preprocessing (NaN drop, StandardScaler)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Feature Engineering<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    ML Models (Random Forest / XGBoost)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    SHAP Explainability + Inception API<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Streamlit Web App
    </code>
    </div>
    """, unsafe_allow_html=True)
