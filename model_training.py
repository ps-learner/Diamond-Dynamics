# model_training.py

import os
import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gdown
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import skew
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from xgboost import XGBRegressor


# =========================
# Global Configuration
# =========================
RANDOM_STATE = 42
USD_TO_INR = 83.5
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1m9HU-CoGXCzLtj9DyAoZt-13BfHKQH0c"
DATA_PATH = "data/diamonds.csv"

PLOTS_DIR = Path("plots")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
SCREENSHOTS_DIR = Path("screenshots")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 120

cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

NUMERIC_COLUMNS_BASE = ["carat", "depth", "table", "price", "x", "y", "z"]
CATEGORICAL_COLUMNS = ["cut", "color", "clarity"]


# =========================
# Utility Functions
# =========================
def ensure_directories():
    for directory in [PLOTS_DIR, MODELS_DIR, DATA_DIR, SCREENSHOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    gitkeep = DATA_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


def print_banner():
    banner = """
==============================================================
💎 DIAMOND DYNAMICS — MODEL TRAINING PIPELINE
==============================================================
A complete ML pipeline for diamond price prediction and market segmentation
==============================================================
"""
    print(banner)


def download_dataset(url: str = GOOGLE_DRIVE_URL, output_path: str = DATA_PATH):
    if not os.path.exists(output_path):
        print(f"[INFO] Downloading dataset to {output_path} ...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"[INFO] Dataset already exists at {output_path}")


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    print("[INFO] Loading dataset...")
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"[INFO] Dataset shape: {df.shape}")
    print("\n[HEAD]")
    print(df.head())
    print("\n[INFO]")
    print(df.info())
    print("\n[DESCRIBE]")
    print(df.describe(include="all"))
    print("\n[NULLS]")
    print(df.isnull().sum())
    print(f"\n[DUPLICATES] {df.duplicated().sum()}")
    return df


def add_subtitle(ax, subtitle: str):
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )


def save_current_plot(filename: str):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches="tight")
    plt.close()


def business_note(title: str, line1: str, line2: str):
    print(f"\n[BUSINESS INSIGHT] {title}")
    print(f" - {line1}")
    print(f" - {line2}")


def validate_dtypes(df: pd.DataFrame):
    print("\n[INFO] Validating dtypes...")
    for col in CATEGORICAL_COLUMNS:
        print(f"{col}: {df[col].dtype}")
    for col in [c for c in df.columns if c not in CATEGORICAL_COLUMNS]:
        print(f"{col}: {df[col].dtype}")


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    cleaned_df = df.copy()
    for col in columns:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
    return cleaned_df.reset_index(drop=True)


def carat_category(c):
    if c < 0.5:
        return "Light"
    elif c <= 1.5:
        return "Medium"
    return "Heavy"


def compute_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    vif_df = df[features].copy().dropna()
    vif_values = []
    for i in range(vif_df.shape[1]):
        vif_values.append(variance_inflation_factor(vif_df.values, i))
    result = pd.DataFrame({"feature": features, "VIF": vif_values}).sort_values("VIF", ascending=False)
    return result


def evaluate_model(y_true_log, y_pred_log, model_name: str) -> dict:
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_orig, y_pred_orig)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.clip(y_true_orig, 1e-8, None))) * 100

    return {
        "Model": model_name,
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


def build_ann(input_dim: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.20),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def save_keras_model(model, output_dir="models"):
    keras_path = os.path.join(output_dir, "best_ann_model.keras")
    model.save(keras_path)
    return keras_path


# =========================
# Preprocessing
# =========================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 2] Data preprocessing started...")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing_before = df.isnull().sum()
    print("\nMissing values before preprocessing:")
    print(missing_before)

    for col in ["x", "y", "z"]:
        df[col] = df[col].replace(0, np.nan)

    for col in ["x", "y", "z"]:
        if df[col].isnull().sum() > 0:
            group_median = df.groupby("cut")[col].transform("median")
            df[col] = df[col].fillna(group_median)
            df[col] = df[col].fillna(df[col].median())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values after preprocessing:")
    print(df.isnull().sum())

    validate_dtypes(df)
    return df


# =========================
# Plotting Functions
# =========================
def plot_distributions(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df["price"], bins=50, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Diamond Price Distribution")
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("Count")
    add_subtitle(ax, "Strong right-skew indicates most diamonds cluster at lower price points.")
    save_current_plot("01_price_distribution.png")
    business_note(
        "Price Distribution",
        "Most diamonds sit in lower price bands, with a long premium tail.",
        "Retail pricing strategies should account for a highly imbalanced premium segment.",
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df["carat"], bins=50, kde=True, color="#c8960c", ax=ax)
    ax.set_title("Carat Distribution")
    ax.set_xlabel("Carat")
    ax.set_ylabel("Count")
    add_subtitle(ax, "Inventory is heavily concentrated in smaller stones.")
    save_current_plot("02_carat_distribution.png")
    business_note(
        "Carat Distribution",
        "Smaller stones dominate the inventory mix.",
        "High-carat stones are scarce and likely drive premium-margin merchandising.",
    )


def plot_countplots(df: pd.DataFrame):
    mappings = [
        ("cut", "03_cut_count.png", "Cut Grade Distribution", "#7b68ee"),
        ("color", "04_color_count.png", "Color Grade Distribution", "#20b2aa"),
        ("clarity", "05_clarity_count.png", "Clarity Grade Distribution", "#ff7f50"),
    ]

    for col, fname, title, color in mappings:
        fig, ax = plt.subplots(figsize=(12, 7))
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, palette="viridis", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Count")
        add_subtitle(ax, f"Category mix for {col} influences assortment and pricing tiers.")
        save_current_plot(fname)
        business_note(
            title,
            f"{col.capitalize()} distribution shows where inventory concentration is strongest.",
            f"This mix can guide procurement, bundling, and segment-focused recommendations.",
        )


def plot_scatter_and_boxplots(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(data=df.sample(min(10000, len(df)), random_state=RANDOM_STATE),
                    x="carat", y="price", alpha=0.35, color="#d4af37", ax=ax)
    ax.set_title("Carat vs Price")
    ax.set_xlabel("Carat")
    ax.set_ylabel("Price (USD)")
    add_subtitle(ax, "Carat is the strongest visible driver of diamond pricing.")
    save_current_plot("06_price_vs_carat_scatter.png")
    business_note(
        "Carat vs Price",
        "Price rises sharply with carat, though premium quality features add spread.",
        "Carat should be central to pricing models and customer-facing quote tools.",
    )

    box_configs = [
        ("cut", "07_price_by_cut_boxplot.png", "Price Distribution by Cut"),
        ("color", "08_price_by_color_boxplot.png", "Price Distribution by Color"),
        ("clarity", "09_price_by_clarity_boxplot.png", "Price Distribution by Clarity"),
    ]

    for col, fname, title in box_configs:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=df, x=col, y="price", palette="magma", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Price (USD)")
        add_subtitle(ax, f"Price spread by {col} reveals quality-premium dynamics.")
        save_current_plot(fname)
        business_note(
            title,
            f"{col.capitalize()} materially affects pricing but not always in a strictly monotonic retail pattern.",
            "Cross-feature interactions mean quality alone does not explain all premium pricing.",
        )


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap — Numerical Features")
    add_subtitle(ax, "Strong interdependence exists among carat, dimensions, and price.")
    save_current_plot("10_correlation_heatmap.png")
    business_note(
        "Correlation Heatmap",
        "Dimensions and carat move closely with price and with each other.",
        "Feature engineering must handle signal overlap and multicollinearity carefully.",
    )


def plot_outlier_boxplots(df: pd.DataFrame):
    cols = ["carat", "price", "x", "y", "z", "depth", "table"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#d4af37")
        axes[i].set_title(col)

    if len(cols) < len(axes):
        for j in range(len(cols), len(axes)):
            fig.delaxes(axes[j])

    fig.suptitle("Outlier Boxplots for Core Numerical Features", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_outlier_boxplots.png", bbox_inches="tight")
    plt.close()

    business_note(
        "Outlier Boxplots",
        "Large outliers appear in price, carat, and dimensions as expected for luxury inventory.",
        "IQR filtering helps stabilize regression performance while preserving business relevance.",
    )


def plot_skewness_before_after(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(df["price"], kde=True, ax=axes[0, 0], color="#4682b4")
    axes[0, 0].set_title("Original Price")

    sns.histplot(df["price_log"], kde=True, ax=axes[0, 1], color="#2e8b57")
    axes[0, 1].set_title("Log-Transformed Price")

    sns.histplot(df["carat"], kde=True, ax=axes[1, 0], color="#b8860b")
    axes[1, 0].set_title("Original Carat")

    sns.histplot(df["carat_log"], kde=True, ax=axes[1, 1], color="#8a2be2")
    axes[1, 1].set_title("Log-Transformed Carat")

    fig.suptitle("Skewness Before and After Log Transformation", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_skewness_before_after.png", bbox_inches="tight")
    plt.close()

    business_note(
        "Skewness Treatment",
        "Log transformation compresses extreme tails and makes regression targets more learnable.",
        "This is especially helpful in luxury pricing where a few very expensive items dominate scale.",
    )


def plot_feature_importances(importances: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 8))
    importances.sort_values().plot(kind="barh", color="#d4af37", ax=ax)
    ax.set_title("Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    add_subtitle(ax, "Model-based ranking highlights the strongest pricing signals.")
    save_current_plot("13_feature_importances.png")

    business_note(
        "Feature Importances",
        "A small set of features usually carries most of the predictive power.",
        "This supports simpler, faster, and more interpretable production-grade pricing systems.",
    )


def plot_model_comparison(results_df: pd.DataFrame):
    sorted_df = results_df.sort_values("R2", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette("magma", n_colors=len(sorted_df))
    sns.barplot(data=sorted_df, x="R2", y="Model", palette=palette, ax=ax)
    ax.set_title("Regression Model Comparison by R² Score")
    ax.set_xlabel("R²")
    ax.set_ylabel("Model")
    add_subtitle(ax, "Higher R² indicates stronger price prediction performance.")
    save_current_plot("14_model_comparison.png")

    business_note(
        "Model Comparison",
        "Tree ensembles and boosting typically outperform simple linear baselines on diamond pricing.",
        "The leaderboard helps justify final model selection in business and recruiter-facing storytelling.",
    )


def plot_ann_history(history):
    history_df = pd.DataFrame(history.history)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(history_df["loss"], label="Train Loss", linewidth=2)
    ax.plot(history_df["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_title("ANN Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    add_subtitle(ax, "Training curves help assess convergence and overfitting.")
    save_current_plot("15_ann_training_history.png")

    business_note(
        "ANN Training History",
        "The validation curve indicates whether the network generalizes or starts to overfit.",
        "Callbacks like early stopping and learning-rate reduction improve training stability.",
    )


def plot_elbow_and_silhouette(k_values, inertia, silhouettes):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(k_values, inertia, marker="o", linewidth=2, color="#d4af37")
    ax.set_title("Elbow Method for KMeans")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    add_subtitle(ax, "The elbow helps identify diminishing returns from additional clusters.")
    save_current_plot("16_elbow_method.png")

    business_note(
        "Elbow Method",
        "Beyond a certain K, cluster compactness improves only marginally.",
        "Choosing a parsimonious K supports clearer business segmentation.",
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(k_values, silhouettes, marker="o", linewidth=2, color="#20b2aa")
    ax.set_title("Silhouette Scores Across K")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    add_subtitle(ax, "Higher silhouette values indicate better separation between segments.")
    save_current_plot("17_silhouette_scores.png")

    business_note(
        "Silhouette Scores",
        "Segment quality improves when clusters are both compact and well-separated.",
        "This metric complements the elbow plot for robust cluster count selection.",
    )


def plot_pca_clusters(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df.sample(min(15000, len(df)), random_state=RANDOM_STATE),
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="viridis",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("PCA Projection of Diamond Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    add_subtitle(ax, "PCA condenses high-dimensional structure into a business-friendly segment view.")
    save_current_plot("18_pca_cluster_2d.png")

    business_note(
        "PCA Cluster View",
        "Cluster visualization makes segment structure intuitive for non-technical stakeholders.",
        "This is useful in merchandising, buyer personas, and inventory positioning.",
    )


def plot_cluster_profiles(cluster_profile: pd.DataFrame):
    profile_plot = cluster_profile.copy()
    profile_plot = profile_plot.reset_index().melt(id_vars="cluster", var_name="Feature", value_name="Value")

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=profile_plot, x="Feature", y="Value", hue="cluster", ax=ax, palette="Set2")
    ax.set_title("Cluster Profile Comparison")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean Value")
    ax.tick_params(axis="x", rotation=30)
    add_subtitle(ax, "Segment-level averages show how clusters differ on quality and size dimensions.")
    save_current_plot("19_cluster_profiles.png")

    business_note(
        "Cluster Profiles",
        "Each cluster represents a distinct retail segment with unique size-quality characteristics.",
        "Segment-aware pricing and recommendation systems can improve targeting and conversion.",
    )


def plot_pairplot(df: pd.DataFrame):
    sample_df = df[["carat", "x", "y", "z", "price"]].sample(min(2000, len(df)), random_state=RANDOM_STATE)
    pairgrid = sns.pairplot(sample_df, diag_kind="kde")
    pairgrid.fig.suptitle("Pairplot — Carat, Dimensions, and Price", y=1.02, fontsize=16, fontweight="bold")
    pairgrid.savefig(PLOTS_DIR / "20_pairplot_sample.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    business_note(
        "Pairplot",
        "Pairwise relationships confirm strong structural links between dimensions and pricing.",
        "This supports engineered proxies such as volume and surface area.",
    )


def plot_grouped_average_bars(df: pd.DataFrame):
    grouped_features = ["cut", "color", "clarity"]

    for feature in grouped_features:
        grp = df.groupby(feature)["price"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(12, 7))
        grp.plot(kind="bar", color="#d4af37", ax=ax)
        ax.set_title(f"Average Price by {feature.capitalize()}")
        ax.set_xlabel(feature.capitalize())
        ax.set_ylabel("Average Price (USD)")
        add_subtitle(ax, f"Mean price variation by {feature} reflects perceived quality premiums.")
        save_current_plot(f"avg_price_by_{feature}.png")

        business_note(
            f"Average Price by {feature.capitalize()}",
            f"Average pricing differs clearly across {feature} tiers.",
            "Merchants can use this to shape assortments and premium upsell paths.",
        )


def plot_volume_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    sample_df = df.sample(min(10000, len(df)), random_state=RANDOM_STATE)
    sns.scatterplot(data=sample_df, x="volume", y="price", alpha=0.35, color="#7fffd4", ax=ax)
    ax.set_title("Volume vs Price")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Price (USD)")
    add_subtitle(ax, "Physical size proxy aligns strongly with retail price, but quality adds variance.")
    save_current_plot("volume_vs_price.png")

    business_note(
        "Volume vs Price",
        "Volume captures a physical-size premium not fully represented by single-axis dimensions.",
        "This engineered feature strengthens price intelligence for real-world catalog data.",
    )


# =========================
# Feature Engineering
# =========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 6] Feature engineering started...")

    df = df.copy()

    df["price_inr"] = df["price"] * USD_TO_INR
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["price_per_carat"] = df["price"] / np.clip(df["carat"], 1e-8, None)
    df["dimension_ratio"] = (df["x"] + df["y"]) / np.clip(2 * df["z"], 1e-8, None)
    df["carat_category"] = df["carat"].apply(carat_category)
    df["surface_area"] = 2 * (
        (df["x"] * df["y"]) + (df["y"] * df["z"]) + (df["x"] * df["z"])
    )
    df["depth_table_ratio"] = df["depth"] / np.clip(df["table"], 1e-8, None)
    df["lw_ratio"] = df["x"] / np.clip(df["y"], 1e-8, None)

    premium_cut = df["cut"] == "Ideal"
    premium_color = df["color"].isin(["D", "E"])
    premium_clarity = df["clarity"].isin(["IF", "VVS1", "VVS2"])
    df["is_premium"] = (premium_cut & premium_color & premium_clarity).astype(int)

    df["log_price"] = np.log1p(df["price"])
    df["log_price_inr"] = np.log1p(df["price_inr"])

    skewed_cols = ["price", "carat", "x", "y", "z"]
    for col in skewed_cols:
        df[f"{col}_log"] = np.log1p(df[col])

    return df


# =========================
# Encoding
# =========================
def encode_ordinal_features(df: pd.DataFrame):
    print("\n[STEP 7] Ordinal encoding started...")

    encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
    df[["cut_enc", "color_enc", "clarity_enc"]] = encoder.fit_transform(df[["cut", "color", "clarity"]])

    joblib.dump(encoder, MODELS_DIR / "label_encoders.pkl")
    return df, encoder


# =========================
# Feature Selection
# =========================
def feature_selection(df: pd.DataFrame):
    print("\n[STEP 8] Feature selection started...")

    feature_cols = [
        "carat",
        "cut_enc",
        "color_enc",
        "clarity_enc",
        "depth",
        "table",
        "x",
        "y",
        "z",
        "volume",
        "price_per_carat",
        "dimension_ratio",
        "surface_area",
        "depth_table_ratio",
        "lw_ratio",
        "is_premium",
    ]

    X_fs = df[feature_cols]
    y_fs = df["log_price_inr"]

    rf_fs = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_fs.fit(X_fs, y_fs)

    importances = pd.Series(rf_fs.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    plot_feature_importances(importances)

    selected_features = importances[importances > 0.01].index.tolist()
    if len(selected_features) < 5:
        selected_features = importances.head(8).index.tolist()

    vif_numeric_candidates = [f for f in selected_features if f not in ["cut_enc", "color_enc", "clarity_enc", "is_premium"]]
    if len(vif_numeric_candidates) >= 2:
        vif_df = compute_vif(df, vif_numeric_candidates)
        print("\nVIF Check:")
        print(vif_df)
    else:
        vif_df = pd.DataFrame(columns=["feature", "VIF"])

    joblib.dump(selected_features, MODELS_DIR / "selected_features.pkl")
    return selected_features, importances, vif_df


# =========================
# Regression Modeling
# =========================
def regression_pipeline(df: pd.DataFrame, selected_features: list):
    print("\n[STEP 9] Regression model building started...")

    X = df[selected_features].copy()
    y = df["log_price_inr"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
    }

    metrics = []
    fitted_models = {}

    scaled_models = {"Linear Regression", "Ridge Regression", "KNN"}

    for model_name, model in models.items():
        print(f"[INFO] Training {model_name}...")
        if model_name in scaled_models:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        result = evaluate_model(y_test, preds, model_name)
        metrics.append(result)
        fitted_models[model_name] = model
        print(result)

    results_df = pd.DataFrame(metrics).sort_values("R2", ascending=False).reset_index(drop=True)
    plot_model_comparison(results_df)

    print("\n[INFO] Training ANN...")
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    ann_model = build_ann(X_train_scaled.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1
    )

    history = ann_model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    ann_preds = ann_model.predict(X_test_scaled, verbose=0).flatten()
    ann_result = evaluate_model(y_test, ann_preds, "ANN")
    results_df = pd.concat([results_df, pd.DataFrame([ann_result])], ignore_index=True)
    results_df = results_df.sort_values("R2", ascending=False).reset_index(drop=True)
    plot_ann_history(history)
    plot_model_comparison(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n[INFO] Best regression model: {best_model_name}")

    if best_model_name == "ANN":
        save_keras_model(ann_model, MODELS_DIR)
        joblib.dump(scaler, MODELS_DIR / "scaler_regression.pkl")
        best_model_artifact_name = "best_ann_model.keras"
    else:
        best_model = fitted_models[best_model_name]
        joblib.dump(best_model, MODELS_DIR / "best_regression_model.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler_regression.pkl")
        best_model_artifact_name = "best_regression_model.pkl"

    results_df.to_csv(MODELS_DIR / "regression_results.csv", index=False)

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model_artifact_name": best_model_artifact_name,
        "ann_result": ann_result,
        "history": history,
        "scaler": scaler,
        "X_train_columns": list(X.columns),
    }


# =========================
# Clustering
# =========================
def assign_cluster_names(cluster_profile: pd.DataFrame) -> dict:
    profile = cluster_profile.copy()

    price_rank = profile["price"].rank(method="dense")
    carat_rank = profile["carat"].rank(method="dense")
    quality_score = profile["cut_enc"] + profile["clarity_enc"] + profile["color_enc"]
    quality_rank = quality_score.rank(method="dense")

    cluster_names = {}

    for cluster_id in profile.index:
        p_rank = price_rank.loc[cluster_id]
        c_rank = carat_rank.loc[cluster_id]
        q_rank = quality_rank.loc[cluster_id]

        if c_rank >= price_rank.max() and q_rank >= quality_rank.max() - 1:
            cluster_names[int(cluster_id)] = "💎 Premium Heavy Diamonds"
        elif c_rank <= carat_rank.min() + 1 and p_rank <= price_rank.min() + 1:
            cluster_names[int(cluster_id)] = "🪨 Affordable Small Diamonds"
        elif q_rank >= quality_rank.max() - 1 and c_rank <= np.median(carat_rank):
            cluster_names[int(cluster_id)] = "✨ Quality Light Diamonds"
        else:
            cluster_names[int(cluster_id)] = "⚖️ Mid-Range Balanced Diamonds"

    used = set()
    deduped = {}
    fallback_names = [
        "💎 Premium Heavy Diamonds",
        "🪨 Affordable Small Diamonds",
        "⚖️ Mid-Range Balanced Diamonds",
        "✨ Quality Light Diamonds",
    ]
    fallback_idx = 0

    for cid in profile.index:
        name = cluster_names[int(cid)]
        if name in used:
            while fallback_names[fallback_idx] in used:
                fallback_idx += 1
            name = fallback_names[fallback_idx]
        used.add(name)
        deduped[int(cid)] = name

    return deduped


def clustering_pipeline(df: pd.DataFrame):
    print("\n[STEP 10] Clustering started...")

    cluster_features = [
        "carat",
        "cut_enc",
        "color_enc",
        "clarity_enc",
        "depth",
        "table",
        "x",
        "y",
        "z",
        "volume",
        "dimension_ratio",
        "surface_area",
        "is_premium",
    ]

    X_cluster = df[cluster_features].copy()

    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

    inertia = []
    sil_scores = []
    k_values = list(range(2, 11))

    print("[INFO] Evaluating K values for clustering...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_cluster_scaled)
        inertia.append(km.inertia_)
        sil = silhouette_score(X_cluster_scaled, labels, sample_size=min(5000, len(df)), random_state=RANDOM_STATE)
        sil_scores.append(sil)
        print(f"K={k} | Inertia={km.inertia_:.2f} | Silhouette={sil:.4f}")

    plot_elbow_and_silhouette(k_values, inertia, sil_scores)

    best_sil_index = int(np.argmax(sil_scores))
    optimal_k = k_values[best_sil_index]

    if optimal_k != 4:
        print(f"[INFO] Silhouette-selected optimal K={optimal_k}. Using K=4 to align with project specification.")
    optimal_k = 4

    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_cluster_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_cluster_scaled)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    plot_pca_clusters(df)

    cluster_profile = (
        df.groupby("cluster")
        .agg(
            {
                "carat": "mean",
                "price": "mean",
                "volume": "mean",
                "cut_enc": "mean",
                "clarity_enc": "mean",
                "color_enc": "mean",
            }
        )
        .round(2)
    )

    print("\nCluster profile:")
    print(cluster_profile)

    plot_cluster_profiles(cluster_profile)

    cluster_names = assign_cluster_names(cluster_profile)

    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(cluster_scaler, MODELS_DIR / "scaler_clustering.pkl")
    joblib.dump(pca, MODELS_DIR / "pca_model.pkl")
    df.to_csv(DATA_DIR / "diamonds_processed_clustered.csv", index=False)

    final_silhouette = silhouette_score(
        X_cluster_scaled,
        df["cluster"],
        sample_size=min(5000, len(df)),
        random_state=RANDOM_STATE,
    )

    return {
        "df": df,
        "cluster_features": cluster_features,
        "kmeans": kmeans,
        "cluster_scaler": cluster_scaler,
        "pca": pca,
        "cluster_profile": cluster_profile,
        "cluster_names": cluster_names,
        "optimal_k": optimal_k,
        "silhouette_score": float(final_silhouette),
    }


# =========================
# Meta Save
# =========================
def save_meta(
    df_clean: pd.DataFrame,
    selected_features: list,
    cluster_features: list,
    regression_outputs: dict,
    clustering_outputs: dict,
):
    results_df = regression_outputs["results_df"]
    best_model_name = regression_outputs["best_model_name"]

    best_metrics = results_df.loc[results_df["Model"] == best_model_name].iloc[0].to_dict()
    ann_row = results_df.loc[results_df["Model"] == "ANN"]
    ann_metrics = ann_row.iloc[0].to_dict() if not ann_row.empty else {}

    meta = {
        "best_regression_model": best_model_name,
        "best_model_artifact_name": regression_outputs["best_model_artifact_name"],
        "regression_metrics": {
            "R2": round(float(best_metrics["R2"]), 4),
            "RMSE_inr": round(float(best_metrics["RMSE"]), 2),
            "MAE_inr": round(float(best_metrics["MAE"]), 2),
            "MAPE": round(float(best_metrics["MAPE"]), 2),
        },
        "ann_metrics": {
            "R2": round(float(ann_metrics.get("R2", np.nan)), 4) if ann_metrics else None,
            "RMSE_inr": round(float(ann_metrics.get("RMSE", np.nan)), 2) if ann_metrics else None,
            "MAE_inr": round(float(ann_metrics.get("MAE", np.nan)), 2) if ann_metrics else None,
            "MAPE": round(float(ann_metrics.get("MAPE", np.nan)), 2) if ann_metrics else None,
        },
        "clustering": {
            "optimal_k": int(clustering_outputs["optimal_k"]),
            "silhouette_score": round(float(clustering_outputs["silhouette_score"]), 4),
            "cluster_names": {str(k): v for k, v in clustering_outputs["cluster_names"].items()},
        },
        "selected_features": selected_features,
        "cluster_features": cluster_features,
        "usd_to_inr": USD_TO_INR,
        "total_diamonds": int(len(df_clean)),
        "cut_order": cut_order,
        "color_order": color_order,
        "clarity_order": clarity_order,
        "regression_results": results_df.to_dict(orient="records"),
    }

    with open(MODELS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[INFO] meta.json saved successfully.")


# =========================
# Main Pipeline
# =========================
def main():
    print_banner()
    ensure_directories()

    # Step 1
    download_dataset()
    df = load_dataset()

    # Step 2
    df = preprocess_data(df)

    # Step 3
    plot_outlier_boxplots(df)
    outlier_cols = ["carat", "price", "x", "y", "z"]
    rows_before = len(df)
    df_clean = remove_outliers_iqr(df.copy(), outlier_cols)
    rows_after = len(df_clean)

    print(f"\n[STEP 3] Rows before outlier removal: {rows_before}")
    print(f"[STEP 3] Rows after IQR outlier removal: {rows_after}")

    z_scores = np.abs(stats.zscore(df_clean[["carat", "price", "x", "y", "z"]]))
    z_outliers = (z_scores > 3).any(axis=1).sum()
    print(f"[STEP 3] Rows with |z| > 3 after IQR filtering: {z_outliers}")

    # Step 4 + Step 6 prep
    numeric_skew = df_clean[["carat", "price", "x", "y", "z", "depth", "table"]].skew()
    print("\n[STEP 4] Skewness values:")
    print(numeric_skew)

    df_clean = engineer_features(df_clean)
    plot_skewness_before_after(df_clean)

    # Step 5
    print("\n[STEP 5] EDA plots generation started...")
    plot_distributions(df_clean)
    plot_countplots(df_clean)
    plot_scatter_and_boxplots(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_pairplot(df_clean)
    plot_grouped_average_bars(df_clean)
    plot_volume_vs_price(df_clean)

    # Step 7
    df_clean, encoder = encode_ordinal_features(df_clean)

    # Save cleaned enriched dataset
    df_clean.to_csv(DATA_DIR / "diamonds_clean_featured.csv", index=False)

    # Step 8
    selected_features, importances, vif_df = feature_selection(df_clean)
    vif_df.to_csv(MODELS_DIR / "vif_report.csv", index=False)

    # Step 9
    regression_outputs = regression_pipeline(df_clean, selected_features)

    # Save XGBoost as default artifact name if not ANN
    if regression_outputs["best_model_name"] != "ANN":
        best_model = None
        if regression_outputs["best_model_name"] == "XGBoost":
            best_model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                objective="reg:squarederror",
            )
        if best_model is not None:
            pass

    # Step 10
    clustering_outputs = clustering_pipeline(df_clean)

    # Step 11
    save_meta(
        df_clean=clustering_outputs["df"],
        selected_features=selected_features,
        cluster_features=clustering_outputs["cluster_features"],
        regression_outputs=regression_outputs,
        clustering_outputs=clustering_outputs,
    )

    # Final save
    clustering_outputs["cluster_profile"].to_csv(MODELS_DIR / "cluster_profiles.csv")

    print("\n==============================================================")
    print("✅ TRAINING COMPLETE")
    print("Artifacts saved:")
    print(f" - Dataset: {DATA_DIR / 'diamonds.csv'}")
    print(f" - Clean dataset: {DATA_DIR / 'diamonds_clean_featured.csv'}")
    print(f" - Clustered dataset: {DATA_DIR / 'diamonds_processed_clustered.csv'}")
    print(f" - Models directory: {MODELS_DIR.resolve()}")
    print(f" - Plots directory: {PLOTS_DIR.resolve()}")
    print("==============================================================")


if __name__ == "__main__":
    main()