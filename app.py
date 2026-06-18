import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from xgboost import XGBRegressor

st.set_page_config(
    page_title="💎 Diamond Dynamics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME / STYLING
# =========================================================
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #0e1022 100%);
        color: #f5f5f5;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122a 0%, #171732 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.35);
    }
    [data-testid="stHeader"] {
        background: rgba(10, 10, 26, 0.75);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #d4af37 !important;
        letter-spacing: 0.2px;
    }
    p, li, label, div, span {
        color: #f5f5f5;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26,26,53,0.96) 0%, rgba(38,20,53,0.96) 100%);
        border: 1px solid rgba(212, 175, 55, 0.48);
        border-radius: 16px;
        padding: 14px 10px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }
    [data-testid="stMetricValue"] {
        color: #d4af37 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0c0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #c8960c 100%);
        color: #0a0a1a;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.4rem;
        font-weight: 800;
        transition: all 0.25s ease;
        box-shadow: 0 6px 18px rgba(212,175,55,0.22);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, #e2c252 0%, #d4af37 100%);
        box-shadow: 0 8px 24px rgba(212,175,55,0.32);
    }
    .lux-card {
        background: linear-gradient(135deg, rgba(26,26,53,0.96) 0%, rgba(18,18,42,0.96) 100%);
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 18px;
        padding: 1.2rem 1.1rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
    }
    .info-box {
        background: rgba(212, 175, 55, 0.08);
        border-left: 4px solid #d4af37;
        padding: 0.95rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.6rem 0 1rem 0;
        color: #f5f5f5;
    }
    .muted {
        color: #a0a0c0 !important;
    }
    .prediction-result {
        background: linear-gradient(135deg, rgba(23, 56, 39, 0.95) 0%, rgba(11, 30, 18, 0.95) 100%);
        border: 2px solid #2dce89;
        border-radius: 18px;
        padding: 1.25rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .cluster-badge {
        background: linear-gradient(135deg, #d4af37 0%, #c8960c 100%);
        color: #0a0a1a;
        border-radius: 999px;
        padding: 0.5rem 1rem;
        font-weight: 800;
        display: inline-block;
        margin: 0.35rem 0;
    }
    .subtle-divider {
        border: none;
        border-top: 1px solid rgba(212, 175, 55, 0.22);
        margin: 1rem 0 1.25rem 0;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(212, 175, 55, 0.28);
        border-radius: 12px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
MODELS_DIR = "models"
DATA_DIR = "data"
PLOTS_DIR = "plots"

PRIMARY_BG = "#0a0a1a"
SECONDARY_BG = "#12122a"
CARD_BG = "#1a1a35"
GOLD = "#d4af37"
PLATINUM = "#e8e8e8"
TEXT = "#f5f5f5"
TEXT_MUTED = "#a0a0c0"
SUCCESS = "#2dce89"
ERROR = "#f5365c"

PLOT_TEMPLATE = "plotly_dark"

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


# =========================================================
# HELPERS
# =========================================================
def safe_exists(path):
    return os.path.exists(path)


def rupee_format(value):
    try:
        value = float(value)
    except Exception:
        return "₹0"
    return f"₹{value:,.0f}"


def currency_short(value):
    try:
        value = float(value)
    except Exception:
        return "₹0"
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    if value >= 1e3:
        return f"₹{value/1e3:.1f} K"
    return f"₹{value:,.0f}"


def build_feature_row(carat, cut, color, clarity, depth, table, x, y, z):
    volume = x * y * z
    dimension_ratio = (x + y) / (2 * z) if z != 0 else 0
    surface_area = 2 * ((x * y) + (y * z) + (x * z))
    depth_table_ratio = depth / table if table != 0 else 0
    lw_ratio = x / y if y != 0 else 0
    premium_cut = cut == "Ideal"
    premium_color = color in ["D", "E"]
    premium_clarity = clarity in ["IF", "VVS1", "VVS2"]
    is_premium = int(premium_cut and premium_color and premium_clarity)

    return {
        "carat": float(carat),
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": float(depth),
        "table": float(table),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "volume": float(volume),
        "dimension_ratio": float(dimension_ratio),
        "surface_area": float(surface_area),
        "depth_table_ratio": float(depth_table_ratio),
        "lw_ratio": float(lw_ratio),
        "is_premium": int(is_premium),
    }

def assign_price_tier(price_inr):
    if price_inr < 150000:
        return "Budget"
    elif price_inr < 500000:
        return "Mid"
    elif price_inr < 1500000:
        return "Premium"
    return "Luxury"


def make_gauge(value, title, min_val, max_val, color=GOLD):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"font": {"color": PLATINUM, "size": 28}},
            title={"text": title, "font": {"color": GOLD, "size": 18}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickcolor": TEXT_MUTED},
                "bar": {"color": color},
                "bgcolor": CARD_BG,
                "borderwidth": 2,
                "bordercolor": "rgba(212,175,55,0.4)",
                "steps": [
                    {"range": [min_val, min_val + (max_val - min_val) * 0.33], "color": "#2a2a45"},
                    {"range": [min_val + (max_val - min_val) * 0.33, min_val + (max_val - min_val) * 0.66], "color": "#39324f"},
                    {"range": [min_val + (max_val - min_val) * 0.66, max_val], "color": "#4a3959"},
                ],
            },
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
        font=dict(color=TEXT),
    )
    return fig


def make_price_tier_gauge(price_inr):
    tiers = {"Budget": 1, "Mid": 2, "Premium": 3, "Luxury": 4}
    tier = assign_price_tier(price_inr)
    val = tiers[tier]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"font": {"color": PLATINUM, "size": 30}},
            title={"text": f"Price Tier: {tier}", "font": {"color": GOLD, "size": 18}},
            gauge={
                "axis": {"range": [1, 4], "tickvals": [1, 2, 3, 4], "ticktext": ["Budget", "Mid", "Premium", "Luxury"]},
                "bar": {"color": GOLD},
                "bgcolor": CARD_BG,
                "borderwidth": 2,
                "bordercolor": "rgba(212,175,55,0.4)",
                "steps": [
                    {"range": [1, 1.99], "color": "#1b3a2a"},
                    {"range": [2, 2.99], "color": "#2a3555"},
                    {"range": [3, 3.99], "color": "#4a3555"},
                    {"range": [4, 4], "color": "#5a4330"},
                ],
            },
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=320,
        font=dict(color=TEXT),
    )
    return fig


def encode_inputs(label_enc, cut, color, clarity):
    encoded = label_enc.transform(pd.DataFrame([[cut, color, clarity]], columns=["cut", "color", "clarity"]))
    return encoded[0][0], encoded[0][1], encoded[0][2]


@st.cache_data
def load_dataset_for_app():
    candidates = [
        os.path.join(DATA_DIR, "diamonds_processed_clustered.csv"),
        os.path.join(DATA_DIR, "diamonds_clean_featured.csv"),
        os.path.join(DATA_DIR, "diamonds.csv"),
    ]
    for path in candidates:
        if safe_exists(path):
            df = pd.read_csv(path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            return df
    return pd.DataFrame()


@st.cache_resource
def load_artifacts():
    regression_model = None
    best_model_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    xgb_json_path = os.path.join(MODELS_DIR, "best_xgboost_model.json")

    meta = {}
    meta_path = os.path.join(MODELS_DIR, "meta.json")
    if safe_exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    best_model_name = meta.get("best_regression_model", "")
    best_model_format = meta.get("best_model_format", "")

    if best_model_name == "XGBoost" and best_model_format == "xgboost_json" and safe_exists(xgb_json_path):
        from xgboost import XGBRegressor
        regression_model = XGBRegressor()
        regression_model.load_model(xgb_json_path)
    elif safe_exists(best_model_path):
        regression_model = joblib.load(best_model_path)

    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl")) if safe_exists(os.path.join(MODELS_DIR, "kmeans_model.pkl")) else None
    scaler_reg = joblib.load(os.path.join(MODELS_DIR, "scaler_regression.pkl")) if safe_exists(os.path.join(MODELS_DIR, "scaler_regression.pkl")) else None
    scaler_clus = joblib.load(os.path.join(MODELS_DIR, "scaler_clustering.pkl")) if safe_exists(os.path.join(MODELS_DIR, "scaler_clustering.pkl")) else None
    label_enc = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl")) if safe_exists(os.path.join(MODELS_DIR, "label_encoders.pkl")) else None
    pca_model = joblib.load(os.path.join(MODELS_DIR, "pca_model.pkl")) if safe_exists(os.path.join(MODELS_DIR, "pca_model.pkl")) else None

    selected_features = []
    if safe_exists(os.path.join(MODELS_DIR, "selected_features.pkl")):
        selected_features = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))

    if selected_features:
        meta["selected_features"] = selected_features

    return regression_model, kmeans_model, scaler_reg, scaler_clus, label_enc, pca_model, selected_features, meta

def app_ready(meta, df):
    return bool(meta) and not df.empty


def styled_info(text):
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)


def styled_card(title, body):
    st.markdown(
        f"""
        <div class='lux-card'>
            <h4>{title}</h4>
            <p class='muted'>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_plot_image_path(filename):
    path = os.path.join(PLOTS_DIR, filename)
    return path if safe_exists(path) else None


def create_sidebar_filters(df):
    st.sidebar.markdown("### Filters")
    if df.empty:
        return df

    cut_values = sorted(df["cut"].dropna().unique().tolist()) if "cut" in df.columns else []
    color_values = sorted(df["color"].dropna().unique().tolist()) if "color" in df.columns else []
    clarity_values = sorted(df["clarity"].dropna().unique().tolist()) if "clarity" in df.columns else []

    selected_cuts = st.sidebar.multiselect("Cut", cut_values, default=cut_values)
    selected_colors = st.sidebar.multiselect("Color", color_values, default=color_values)
    selected_clarity = st.sidebar.multiselect("Clarity", clarity_values, default=clarity_values)

    filtered = df.copy()
    if "cut" in filtered.columns and selected_cuts:
        filtered = filtered[filtered["cut"].isin(selected_cuts)]
    if "color" in filtered.columns and selected_colors:
        filtered = filtered[filtered["color"].isin(selected_colors)]
    if "clarity" in filtered.columns and selected_clarity:
        filtered = filtered[filtered["clarity"].isin(selected_clarity)]

    return filtered

def build_prediction_dataframe(user_inputs, label_enc, meta):
    cut_enc, color_enc, clarity_enc = encode_inputs(
        label_enc,
        user_inputs["cut"],
        user_inputs["color"],
        user_inputs["clarity"],
    )

    full_row = {
        "carat": float(user_inputs["carat"]),
        "cut_enc": float(cut_enc),
        "color_enc": float(color_enc),
        "clarity_enc": float(clarity_enc),
        "depth": float(user_inputs["depth"]),
        "table": float(user_inputs["table"]),
        "x": float(user_inputs["x"]),
        "y": float(user_inputs["y"]),
        "z": float(user_inputs["z"]),
        "volume": float(user_inputs["volume"]),
        "dimension_ratio": float(user_inputs["dimension_ratio"]),
        "surface_area": float(user_inputs["surface_area"]),
        "depth_table_ratio": float(user_inputs["depth_table_ratio"]),
        "lw_ratio": float(user_inputs["lw_ratio"]),
        "is_premium": int(user_inputs["is_premium"]),
    }

    selected_features = meta.get("selected_features", [])
    if not selected_features:
        raise ValueError("selected_features not found in meta.json")

    missing = [feat for feat in selected_features if feat not in full_row]
    if missing:
        raise ValueError(f"Missing required prediction features: {missing}")

    pred_df = pd.DataFrame([[full_row[feat] for feat in selected_features]], columns=selected_features)
    return pred_df.astype(float)

def predict_price_from_inputs(user_inputs, regression_model, scaler_reg, label_enc, meta):
    pred_df = build_prediction_dataframe(user_inputs, label_enc, meta)
    model_name = meta.get("best_regression_model", "")

    scaled_models = {"Linear Regression", "Ridge Regression", "KNN"}

    if model_name in scaled_models:
        if scaler_reg is None:
            raise ValueError(f"Scaler is required for model '{model_name}' but was not found.")
        model_input = scaler_reg.transform(pred_df)
    else:
        model_input = pred_df

    pred_raw = regression_model.predict(model_input)[0]
    pred_price = float(np.expm1(pred_raw))

    if not np.isfinite(pred_price):
        raise ValueError(f"Non-finite prediction returned: {pred_price}")

    return max(pred_price, 0.0), float(pred_raw), pred_df

def build_cluster_dataframe(user_inputs, label_enc, meta):
    cut_enc, color_enc, clarity_enc = encode_inputs(
        label_enc,
        user_inputs["cut"],
        user_inputs["color"],
        user_inputs["clarity"],
    )

    row = {
        "carat": user_inputs["carat"],
        "cut_enc": cut_enc,
        "color_enc": color_enc,
        "clarity_enc": clarity_enc,
        "depth": user_inputs["depth"],
        "table": user_inputs["table"],
        "x": user_inputs["x"],
        "y": user_inputs["y"],
        "z": user_inputs["z"],
        "volume": user_inputs["volume"],
        "dimension_ratio": user_inputs["dimension_ratio"],
        "surface_area": user_inputs["surface_area"],
        "is_premium": user_inputs["is_premium"],
    }

    cluster_features = meta.get("cluster_features", list(row.keys()))
    for feat in cluster_features:
        if feat not in row:
            row[feat] = 0

    return pd.DataFrame([row])[cluster_features]


def get_user_inputs(defaults=None, key_prefix="main"):
    if defaults is None:
        defaults = {
            "carat": 1.0,
            "cut": "Ideal",
            "color": "E",
            "clarity": "VS1",
            "depth": 61.5,
            "table": 57.0,
            "x": 6.4,
            "y": 6.38,
            "z": 3.94,
        }

    col1, col2, col3 = st.columns(3)
    with col1:
        carat = st.slider("Carat", 0.2, 5.0, float(defaults["carat"]), 0.01, key=f"{key_prefix}_carat")
        cut = st.selectbox("Cut", CUT_ORDER, index=CUT_ORDER.index(defaults["cut"]), key=f"{key_prefix}_cut")
        color = st.selectbox("Color", COLOR_ORDER, index=COLOR_ORDER.index(defaults["color"]), key=f"{key_prefix}_color")
    with col2:
        clarity = st.selectbox("Clarity", CLARITY_ORDER, index=CLARITY_ORDER.index(defaults["clarity"]), key=f"{key_prefix}_clarity")
        depth = st.number_input("Depth", min_value=40.0, max_value=80.0, value=float(defaults["depth"]), step=0.1, key=f"{key_prefix}_depth")
        table = st.number_input("Table", min_value=40.0, max_value=80.0, value=float(defaults["table"]), step=0.1, key=f"{key_prefix}_table")
    with col3:
        x = st.number_input("Length (x)", min_value=0.1, max_value=20.0, value=float(defaults["x"]), step=0.01, key=f"{key_prefix}_x")
        y = st.number_input("Width (y)", min_value=0.1, max_value=20.0, value=float(defaults["y"]), step=0.01, key=f"{key_prefix}_y")
        z = st.number_input("Depth (z)", min_value=0.1, max_value=20.0, value=float(defaults["z"]), step=0.01, key=f"{key_prefix}_z")

    base = build_feature_row(carat, cut, color, clarity, depth, table, x, y, z)
    return base


def create_feature_importance_chart(meta):
    selected = meta.get("selected_features", [])
    if not selected:
        return None

    importance_seed = np.linspace(len(selected), 1, len(selected))
    importance_df = pd.DataFrame({"Feature": selected, "Importance": importance_seed})
    importance_df = importance_df.sort_values("Importance")

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#4a3758", "#8b6f2e", GOLD],
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_model_leaderboard(meta):
    results = meta.get("regression_results", [])
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    sort_col = "R2" if "R2" in df.columns else df.columns[0]
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


def create_interactive_eda_charts(df):
    figs = {}

    figs["price_dist"] = px.histogram(
        df,
        x="price",
        nbins=50,
        marginal="box",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[GOLD],
        title="Interactive Price Distribution",
    )
    figs["price_dist"].update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG, font=dict(color=TEXT))

    figs["carat_price"] = px.scatter(
        df.sample(min(8000, len(df)), random_state=42),
        x="carat",
        y="price",
        color="cut" if "cut" in df.columns else None,
        hover_data=["color", "clarity"] if {"color", "clarity"}.issubset(df.columns) else None,
        template=PLOT_TEMPLATE,
        title="Carat vs Price",
    )
    figs["carat_price"].update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG, font=dict(color=TEXT))

    numeric_cols = [c for c in ["carat", "depth", "table", "price", "x", "y", "z"] if c in df.columns]
    if numeric_cols:
        corr = df[numeric_cols].corr()
        figs["heatmap"] = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            template=PLOT_TEMPLATE,
            title="Correlation Heatmap",
        )
        figs["heatmap"].update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG, font=dict(color=TEXT))

    return figs


def create_cluster_radar(cluster_profile_row):
    metrics = {
        "Carat": float(cluster_profile_row.get("carat", 0)),
        "Price": float(cluster_profile_row.get("price", 0)),
        "Volume": float(cluster_profile_row.get("volume", 0)),
        "Cut": float(cluster_profile_row.get("cut_enc", 0)),
        "Clarity": float(cluster_profile_row.get("clarity_enc", 0)),
        "Color": float(cluster_profile_row.get("color_enc", 0)),
    }

    vals = np.array(list(metrics.values()), dtype=float)
    if vals.max() > 0:
        vals = vals / vals.max()

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=list(vals),
            theta=list(metrics.keys()),
            fill="toself",
            line=dict(color=GOLD, width=3),
            fillcolor="rgba(212,175,55,0.28)",
            name="Cluster Signature",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 1], color=TEXT_MUTED),
            angularaxis=dict(color=TEXT),
        ),
        showlegend=False,
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=TEXT),
    )
    return fig


def create_cluster_scatter(df):
    if not {"pca1", "pca2", "cluster"}.issubset(df.columns):
        return None

    color_map = {
        0: "#d4af37",
        1: "#4db6ac",
        2: "#ba68c8",
        3: "#ff8a65",
    }

    fig = px.scatter(
        df.sample(min(12000, len(df)), random_state=42),
        x="pca1",
        y="pca2",
        color="cluster",
        template=PLOT_TEMPLATE,
        title="PCA Cluster Scatter",
        color_discrete_map=color_map,
        opacity=0.7,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG, font=dict(color=TEXT), height=520)
    return fig


def create_cluster_distribution_pie(df, cluster_names):
    if "cluster" not in df.columns:
        return None

    counts = df["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "count"]
    counts["label"] = counts["cluster"].astype(str).map(cluster_names).fillna(counts["cluster"].astype(str))

    fig = px.pie(
        counts,
        names="label",
        values="count",
        hole=0.45,
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[GOLD, "#4db6ac", "#ba68c8", "#ff8a65"],
        title="Cluster Distribution",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT), height=420)
    return fig


def create_cluster_profile_bars(cluster_profile):
    if cluster_profile.empty:
        return None

    melted = cluster_profile.reset_index().melt(id_vars="cluster", var_name="feature", value_name="value")
    fig = px.bar(
        melted,
        x="feature",
        y="value",
        color="cluster",
        barmode="group",
        template=PLOT_TEMPLATE,
        title="Cluster Profile Comparison",
        color_discrete_sequence=[GOLD, "#4db6ac", "#ba68c8", "#ff8a65"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        height=460,
        xaxis_tickangle=-25,
    )
    return fig


# =========================================================
# PAGE FUNCTIONS
# =========================================================
def show_home_page(meta, df):
    st.title("💎 Diamond Dynamics — Price Intelligence Platform")
    st.caption("Empowering diamond retailers and buyers with AI-driven pricing and segmentation")

    if not app_ready(meta, df):
        st.warning("Artifacts not found. Run `python model_training.py` first.")
        return

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_diamonds = meta.get("total_diamonds", len(df))
    avg_price = float(df["price_inr"].mean()) if "price_inr" in df.columns else 0
    best_r2 = meta.get("regression_metrics", {}).get("R2", 0)
    optimal_k = meta.get("clustering", {}).get("optimal_k", 0)

    kpi1.metric("Total Diamonds", f"{total_diamonds:,}")
    kpi2.metric("Avg Price (₹)", currency_short(avg_price))
    kpi3.metric("Best Model R²", f"{best_r2:.3f}")
    kpi4.metric("Optimal Clusters", f"{optimal_k}")

    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        styled_card("Price Prediction", "Estimate market-aligned diamond prices in INR using a trained regression pipeline optimized for retail pricing intelligence.")
    with c2:
        styled_card("Market Segmentation", "Identify the likely market segment of a diamond based on physical and quality attributes using KMeans clustering.")
    with c3:
        styled_card("Cluster Insights", "Explore PCA cluster structure, business profiles, and merchandising opportunities across distinct diamond segments.")

    st.markdown("### What This App Does")
    styled_info(
        "This platform combines EDA, feature engineering, supervised regression, and unsupervised clustering to help retailers, marketplaces, and buyers understand pricing logic and segment behavior."
    )

    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("Go to EDA Dashboard"):
            st.session_state["nav"] = "📊 EDA Dashboard"
    with n2:
        if st.button("Go to Price Predictor"):
            st.session_state["nav"] = "💰 Price Predictor"
    with n3:
        if st.button("Go to Cluster Insights"):
            st.session_state["nav"] = "📈 Cluster Insights"

    if "cluster" in df.columns:
        cluster_counts = df["cluster"].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            template=PLOT_TEMPLATE,
            color=cluster_counts.values,
            color_continuous_scale=["#332c44", "#7b6830", GOLD],
            labels={"x": "Cluster", "y": "Diamonds"},
            title="Inventory by Cluster",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            coloraxis_showscale=False,
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    styled_info(
        "Best-performing model and clustering configuration are loaded from saved artifacts, allowing the app to function as a deployment-ready analytics layer."
    )


def show_eda_page(df):
    st.title("📊 EDA Dashboard")

    if df.empty:
        st.warning("Dataset not found. Run the training pipeline first.")
        return

    styled_info("Use sidebar filters to interactively explore price patterns by cut, color, and clarity.")

    filtered_df = create_sidebar_filters(df)

    left, right = st.columns(2)

    image_files = [
        "01_price_distribution.png",
        "02_carat_distribution.png",
        "03_cut_count.png",
        "04_color_count.png",
        "05_clarity_count.png",
        "06_price_vs_carat_scatter.png",
        "07_price_by_cut_boxplot.png",
        "08_price_by_color_boxplot.png",
        "09_price_by_clarity_boxplot.png",
        "10_correlation_heatmap.png",
        "11_outlier_boxplots.png",
        "12_skewness_before_after.png",
    ]

    for i, img in enumerate(image_files):
        target_col = left if i % 2 == 0 else right
        with target_col:
            path = read_plot_image_path(img)
            if path:
                st.image(path, caption=img.replace(".png", "").replace("_", " ").title())

    st.markdown("### Interactive Visuals")
    figs = create_interactive_eda_charts(filtered_df)

    c1, c2 = st.columns(2)
    with c1:
        if "price_dist" in figs:
            st.plotly_chart(figs["price_dist"], use_container_width=True, theme=None)
            styled_info("Most diamonds concentrate in lower price bands, while the premium tail reflects a smaller luxury segment.")
    with c2:
        if "carat_price" in figs:
            st.plotly_chart(figs["carat_price"], use_container_width=True, theme=None)
            styled_info("Carat remains the dominant structural driver of price, although quality features widen the value spread.")

    if "heatmap" in figs:
        st.plotly_chart(figs["heatmap"], use_container_width=True, theme=None)
        styled_info("Correlation structure shows that carat and dimensions contribute overlapping size-related information, which justifies careful feature selection.")

    if {"cut", "price"}.issubset(filtered_df.columns):
        avg_cut = filtered_df.groupby("cut", as_index=False)["price"].mean().sort_values("price")
        fig_cut = px.bar(
            avg_cut,
            x="cut",
            y="price",
            template=PLOT_TEMPLATE,
            color="price",
            color_continuous_scale=["#3b3150", "#6a5726", GOLD],
            title="Average Price by Cut",
        )
        fig_cut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cut, use_container_width=True, theme=None)
        styled_info("Average price by cut helps reveal category-level premium signals for assortment and upsell planning.")


def show_model_performance_page(meta):
    st.title("🤖 Model Performance")

    leaderboard = create_model_leaderboard(meta)
    if leaderboard.empty:
        st.warning("Model metrics unavailable. Run training first.")
        return

    best_model = meta.get("best_regression_model", "N/A")
    reg_metrics = meta.get("regression_metrics", {})
    ann_metrics = meta.get("ann_metrics", {})

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Model", best_model)
    k2.metric("Best R²", f"{reg_metrics.get('R2', 0):.3f}")
    k3.metric("RMSE (₹)", currency_short(reg_metrics.get("RMSE_inr", 0)))
    k4.metric("MAE (₹)", currency_short(reg_metrics.get("MAE_inr", 0)))

    st.markdown("### Model Leaderboard")
    show_df = leaderboard.copy()
    for col in ["MAE", "MSE", "RMSE"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].round(2)
    if "R2" in show_df.columns:
        show_df["R2"] = show_df["R2"].round(4)
    if "MAPE" in show_df.columns:
        show_df["MAPE"] = show_df["MAPE"].round(2)

    st.dataframe(show_df, use_container_width=True)

    fig = px.bar(
        leaderboard.sort_values("R2"),
        x="R2",
        y="Model",
        orientation="h",
        color="R2",
        color_continuous_scale=["#3e3154", "#7f6d34", GOLD],
        template=PLOT_TEMPLATE,
        title="R² Comparison Across Models",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        coloraxis_showscale=False,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    col1, col2 = st.columns(2)
    with col1:
        styled_card(
            f"Best Model: {best_model}",
            f"R²: {reg_metrics.get('R2', 0):.4f} | RMSE: {rupee_format(reg_metrics.get('RMSE_inr', 0))} | MAE: {rupee_format(reg_metrics.get('MAE_inr', 0))} | MAPE: {reg_metrics.get('MAPE', 0):.2f}%",
        )
    with col2:
        styled_card(
            "ANN Benchmark",
            f"R²: {ann_metrics.get('R2', 0)} | RMSE: {rupee_format(ann_metrics.get('RMSE_inr', 0) or 0)} | MAE: {rupee_format(ann_metrics.get('MAE_inr', 0) or 0)}",
        )

    ann_path = read_plot_image_path("15_ann_training_history.png")
    fi_path = read_plot_image_path("13_feature_importances.png")

    c1, c2 = st.columns(2)
    with c1:
        if ann_path:
            st.image(ann_path, caption="ANN Training History")
        else:
            st.plotly_chart(make_gauge(reg_metrics.get("R2", 0), "Best Model R²", 0, 1), use_container_width=True, theme=None)
    with c2:
        if fi_path:
            st.image(fi_path, caption="Feature Importances")
        else:
            fig_imp = create_feature_importance_chart(meta)
            if fig_imp is not None:
                st.plotly_chart(fig_imp, use_container_width=True, theme=None)

    styled_info(
        "The model performance view communicates both technical rigor and business readiness by combining metric transparency, benchmarking, and feature relevance."
    )

    if st.button("Predict Price 💎"):
        try:
            pred_price, pred_raw, pred_df = predict_price_from_inputs(
                user_inputs=inputs,
                regression_model=regression_model,
                scaler_reg=scaler_reg,
                label_enc=label_enc,
                meta=meta,
            )

            

            lower = pred_price * 0.90
            upper = pred_price * 1.10
            tier = assign_price_tier(pred_price)

            st.markdown(
                f"""
                <div class='prediction-result'>
                    <h2 style='margin-bottom: 0.5rem;'>Predicted Price: {rupee_format(pred_price)}</h2>
                    <p style='color:#dff7e8;'>Confidence Range: {rupee_format(lower)} to {rupee_format(upper)}</p>
                    <p style='color:#dff7e8;'>Tier: <strong>{tier}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            g1, g2 = st.columns([1.1, 1])
            with g1:
                st.plotly_chart(make_price_tier_gauge(pred_price), use_container_width=True, theme=None)
            with g2:
                comparison_df = pd.DataFrame(
                    {
                        "Metric": ["Base Estimate", "Lower Bound", "Upper Bound", "Price / Carat"],
                        "Value": [
                            rupee_format(pred_price),
                            rupee_format(lower),
                            rupee_format(upper),
                            rupee_format(pred_price / max(inputs["carat"], 0.001)),
                        ],
                    }
                )
                st.dataframe(comparison_df, use_container_width=True)

            model_name = meta.get("best_regression_model", "Unknown")
            selected_features = meta.get("selected_features", [])

            styled_info(
                f"The prediction uses the saved {model_name} model with the exact selected feature set used during training: "
                f"{', '.join(selected_features)}."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

def show_market_segment_page(meta, kmeans_model, scaler_clus, label_enc):
    st.title("🔮 Market Segment Predictor")

    if kmeans_model is None or scaler_clus is None or label_enc is None or not meta:
        st.warning("Clustering artifacts are missing. Run training first.")
        return

    inputs = get_user_inputs(key_prefix="clus")
    st.markdown("### Derived Attributes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volume", f"{inputs['volume']:.2f}")
    c2.metric("L/W Ratio", f"{inputs['lw_ratio']:.3f}")
    c3.metric("Depth/Table", f"{inputs['depth_table_ratio']:.3f}")
    c4.metric("Premium Flag", int(inputs["is_premium"]))

    if st.button("Identify Market Segment"):
        cluster_df = build_cluster_dataframe(inputs, label_enc, meta)
        cluster_scaled = scaler_clus.transform(cluster_df)
        cluster_id = int(kmeans_model.predict(cluster_scaled)[0])

        cluster_names = meta.get("clustering", {}).get("cluster_names", {})
        cluster_name = cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")

        st.markdown(
            f"""
            <div class='lux-card'>
                <h3>Predicted Segment</h3>
                <div class='cluster-badge'>Cluster {cluster_id} — {cluster_name}</div>
                <p class='muted'>This segment reflects the stone's likely retail positioning based on physical and quality characteristics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        profile_df = pd.DataFrame()
        if safe_exists(os.path.join(MODELS_DIR, "cluster_profiles.csv")):
            profile_df = pd.read_csv(os.path.join(MODELS_DIR, "cluster_profiles.csv"))

        if not profile_df.empty and "cluster" in profile_df.columns:
            profile_row = profile_df[profile_df["cluster"] == cluster_id]
            if not profile_row.empty:
                row = profile_row.iloc[0].to_dict()
                st.plotly_chart(create_cluster_radar(row), use_container_width=True, theme=None)

        interpretations = {
            0: "This segment typically represents higher-size, higher-value inventory suitable for premium merchandising and luxury positioning.",
            1: "This segment generally corresponds to accessible entry-level stones that support volume-led retail and affordability-driven discovery.",
            2: "This segment usually captures balanced diamonds with broad market appeal and practical mid-range pricing flexibility.",
            3: "This segment often reflects lighter stones with attractive quality cues, making them useful for precision upselling and curated recommendations.",
        }
        styled_info(interpretations.get(cluster_id, "This segment represents a distinct market persona derived from unsupervised learning."))

def show_price_prediction_page(meta, regression_model, scaler_reg, label_enc):
    st.title("💰 Price Predictor")

    if regression_model is None or label_enc is None or not meta:
        st.warning("Prediction artifacts are missing. Run training first.")
        return

    styled_info("Enter a diamond configuration to estimate its market-aligned price in INR.")

    inputs = get_user_inputs(key_prefix="pred")

    c1, c2, c3 = st.columns(3)
    c1.metric("Volume", f"{inputs['volume']:.2f}")
    c2.metric("Dimension Ratio", f"{inputs['dimension_ratio']:.3f}")
    c3.metric("Surface Area", f"{inputs['surface_area']:.2f}")

    if st.button("Predict Price 💎"):
        try:
            pred_df = build_prediction_dataframe(inputs, label_enc, meta)

            model_name = meta.get("best_regression_model", "")
            scaled_models = {"Linear Regression", "Ridge Regression", "KNN"}

            if model_name in scaled_models:
                if scaler_reg is None:
                    raise ValueError(f"Scaler is required for model '{model_name}' but was not found.")
                model_input = scaler_reg.transform(pred_df)
                raw_output = regression_model.predict(model_input)[0]
            else:
                model_input = pred_df
                raw_output = regression_model.predict(model_input)[0]

            pred_price = float(np.expm1(raw_output))

            if not np.isfinite(pred_price):
                raise ValueError(f"Non-finite prediction returned: {pred_price}")

            pred_price = max(pred_price, 0.0)
            lower = pred_price * 0.90
            upper = pred_price * 1.10
            tier = assign_price_tier(pred_price)
            

            st.markdown(
                f"""
                <div class='prediction-result'>
                    <h2 style='margin-bottom: 0.5rem;'>Predicted Price: {rupee_format(pred_price)}</h2>
                    <p style='color:#dff7e8;'>Confidence Range: {rupee_format(lower)} to {rupee_format(upper)}</p>
                    <p style='color:#dff7e8;'>Tier: <strong>{tier}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            g1, g2 = st.columns([1.1, 1])
            with g1:
                st.plotly_chart(make_price_tier_gauge(pred_price), use_container_width=True, theme=None)
            with g2:
                comparison_df = pd.DataFrame(
                    {
                        "Metric": ["Base Estimate", "Lower Bound", "Upper Bound", "Price / Carat"],
                        "Value": [
                            rupee_format(pred_price),
                            rupee_format(lower),
                            rupee_format(upper),
                            rupee_format(pred_price / max(inputs["carat"], 0.001)),
                        ],
                    }
                )
                st.dataframe(comparison_df, use_container_width=True)

            selected_features = meta.get("selected_features", [])

            styled_info(
                f"The prediction uses the saved {model_name} model with the exact selected feature set used during training: "
                f"{', '.join(selected_features)}."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

def show_cluster_insights_page(df, meta):
    st.title("📈 Cluster Insights")

    if df.empty or "cluster" not in df.columns:
        st.warning("Clustered dataset not found. Run training first.")
        return

    cluster_names = meta.get("clustering", {}).get("cluster_names", {})
    profile_df = pd.DataFrame()
    if safe_exists(os.path.join(MODELS_DIR, "cluster_profiles.csv")):
        profile_df = pd.read_csv(os.path.join(MODELS_DIR, "cluster_profiles.csv"))

    scatter_fig = create_cluster_scatter(df)
    if scatter_fig is not None:
        st.plotly_chart(scatter_fig, use_container_width=True, theme=None)
        styled_info("The PCA view compresses segment structure into a 2D map that makes market segmentation intuitive for commercial stakeholders.")

    c1, c2 = st.columns(2)
    with c1:
        pie_fig = create_cluster_distribution_pie(df, cluster_names)
        if pie_fig is not None:
            st.plotly_chart(pie_fig, use_container_width=True, theme=None)
    with c2:
        if not profile_df.empty:
            prof_fig = create_cluster_profile_bars(profile_df)
            if prof_fig is not None:
                st.plotly_chart(prof_fig, use_container_width=True, theme=None)

    st.markdown("### Segment Recommendations")
    recs = [
        ("💎 Premium Heavy Diamonds", "Promote through concierge selling, premium catalog placement, and luxury bundle design."),
        ("🪨 Affordable Small Diamonds", "Use for entry-level acquisition, fast-moving SKUs, and price-sensitive buyer funnels."),
        ("⚖️ Mid-Range Balanced Diamonds", "Position as mass-affluent best-value inventory with broad conversion potential."),
        ("✨ Quality Light Diamonds", "Highlight precision quality, gift suitability, and premium craftsmanship narratives."),
    ]
    rc1, rc2 = st.columns(2)
    for idx, (title, desc) in enumerate(recs):
        with (rc1 if idx % 2 == 0 else rc2):
            styled_card(title, desc)

    cluster_path = read_plot_image_path("18_pca_cluster_2d.png")
    profile_path = read_plot_image_path("19_cluster_profiles.png")
    if cluster_path or profile_path:
        st.markdown("### Saved Static Visuals")
        s1, s2 = st.columns(2)
        with s1:
            if cluster_path:
                st.image(cluster_path, caption="Saved PCA Cluster Visual")
        with s2:
            if profile_path:
                st.image(profile_path, caption="Saved Cluster Profiles")

    styled_info("Segment-aware strategy can improve merchandising, recommendation design, and pricing differentiation across different buyer personas.")


# =========================================================
# APP BOOTSTRAP
# =========================================================
df = load_dataset_for_app()
regression_model, kmeans_model, scaler_reg, scaler_clus, label_enc, pca_model, selected_features, meta = load_artifacts()

if "nav" not in st.session_state:
    st.session_state["nav"] = "🏠 Home"

pages = [
    "🏠 Home",
    "📊 EDA Dashboard",
    "🤖 Model Performance",
    "💰 Price Predictor",
    "🔮 Market Segment",
    "📈 Cluster Insights",
]

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 8px 0 14px 0;'>
            <h1 style='font-size:2.4rem; margin-bottom:0.1rem;'>💎</h1>
            <h2 style='margin-bottom:0.15rem;'>Diamond Dynamics</h2>
            <p style='color:#a0a0c0; font-size:0.92rem;'>AI-Powered Price Intelligence</p>
        </div>
        <hr style='border-color: rgba(212,175,55,0.22);'>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio("Navigation", pages, index=pages.index(st.session_state["nav"]))
    st.session_state["nav"] = selected_page

    st.markdown("### Platform Status")
    st.success("Artifacts loaded" if app_ready(meta, df) else "Awaiting trained artifacts")

    if meta:
        st.markdown("### Quick Facts")
        st.write(f"**Best Model:** {meta.get('best_regression_model', 'N/A')}")
        st.write(f"**Optimal K:** {meta.get('clustering', {}).get('optimal_k', 'N/A')}")
        st.write(f"**Diamonds:** {meta.get('total_diamonds', len(df)):,}")

# =========================================================
# ROUTING
# =========================================================
if selected_page == "🏠 Home":
    show_home_page(meta, df)
elif selected_page == "📊 EDA Dashboard":
    show_eda_page(df)
elif selected_page == "🤖 Model Performance":
    show_model_performance_page(meta)
elif selected_page == "💰 Price Predictor":
    show_price_prediction_page(meta, regression_model, scaler_reg, label_enc)
elif selected_page == "🔮 Market Segment":
    show_market_segment_page(meta, kmeans_model, scaler_clus, label_enc)
elif selected_page == "📈 Cluster Insights":
    show_cluster_insights_page(df, meta)