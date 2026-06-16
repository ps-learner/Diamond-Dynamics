# 💎 Diamond Dynamics — AI-Powered Diamond Price Intelligence

## 📌 Table of Contents

- [🎯 Problem Statement](#-problem-statement)
- [🌍 Real-World Use Cases](#-real-world-use-cases)
- [📸 App Screenshots](#-app-screenshots)
- [🏗️ Project Architecture](#️-project-architecture)
- [🗂️ Repository Structure](#️-repository-structure)
- [📊 Dataset Description](#-dataset-description)
- [⚙️ Feature Engineering](#️-feature-engineering)
- [📊 Model Results](#-model-results)
- [🔮 Clustering Results](#-clustering-results)
- [🚀 How to Run](#-how-to-run)
- [☁️ Deployment](#️-deployment)
- [🛠️ Tech Stack](#️-tech-stack)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## 🎯 Problem Statement

Diamond retail pricing is influenced by a complex interaction of size, proportions, cut, color, clarity, and physical dimensions. Traditional manual pricing logic is often inconsistent, difficult to scale, and hard to explain across large e-commerce catalogs.

**Diamond Dynamics** solves this by building:
- A complete end-to-end machine learning pipeline for **diamond price prediction in INR**
- A **market segmentation engine** using clustering
- A recruiter-grade **Streamlit dashboard** for business storytelling, live inference, and segment exploration

This project is designed to look strong from every angle:
- Business relevance
- Technical rigor
- Clean engineering structure
- Deployability
- Visual polish

---

## 🌍 Real-World Use Cases

| Use Case | Problem | Solution from Diamond Dynamics |
|---|---|---|
| Retail pricing engine | Manual price setting is inconsistent | Predict fair INR pricing using trained ML models |
| E-commerce catalog strategy | Large inventories are hard to classify | Segment diamonds into market-driven clusters |
| Buyer recommendation support | Customers need guided product positioning | Use cluster labels and quality signals to explain value |
| Procurement intelligence | Merchants need stock mix clarity | Analyze which segments dominate inventory |
| Luxury merchandising | Premium stones need differentiated treatment | Detect premium-heavy clusters and pricing bands |
| Sales enablement | Teams struggle to justify quotes | Provide model-backed price ranges and derived insights |

---

## 📸 App Screenshots

> Save screenshots from the Streamlit app into the `screenshots/` folder after running the project.

### 1. Home Dashboard
`screenshots/home_dashboard.png`

### 2. EDA Dashboard
`screenshots/eda_dashboard.png`

### 3. Model Performance
`screenshots/model_performance.png`

### 4. Price Predictor
`screenshots/price_predictor.png`

### 5. Market Segment Predictor
`screenshots/market_segment.png`

### 6. Cluster Insights
`screenshots/cluster_insights.png`
---

## 🏗️ Project Architecture

```text
                     ┌────────────────────────────┐
                     │   Diamonds Dataset CSV     │
                     │   (Auto-downloaded)        │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Data Cleaning              │
                     │ - Null checks              │
                     │ - Zero dimension repair    │
                     │ - Type validation          │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Outlier + Skew Handling    │
                     │ - IQR filtering            │
                     │ - Z-score check            │
                     │ - log1p transforms         │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ EDA + Plot Generation      │
                     │ - Distributions            │
                     │ - Boxplots                 │
                     │ - Heatmap                  │
                     │ - Pairplot                 │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Feature Engineering        │
                     │ - Volume                   │
                     │ - Surface Area             │
                     │ - Premium Flag             │
                     │ - INR Conversion           │
                     └─────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
      ┌──────────────────────────┐   ┌──────────────────────────┐
      │ Regression Modelling     │   │ Clustering               │
      │ - Linear / Ridge         │   │ - KMeans                 │
      │ - Decision Tree          │   │ - Elbow Method           │
      │ - Random Forest          │   │ - Silhouette Score       │
      │ - KNN                    │   │ - PCA Visualization      │
      │ - XGBoost                │   │ - Cluster Naming         │
      │ - ANN (TensorFlow)       │   └─────────────┬────────────┘
      └─────────────┬────────────┘                 │
                    ▼                              ▼
      ┌──────────────────────────┐   ┌──────────────────────────┐
      │ Saved Artifacts          │   │ Saved Artifacts          │
      │ - best_regression_model  │   │ - kmeans_model           │
      │ - scaler_regression      │   │ - scaler_clustering      │
      │ - selected_features      │   │ - pca_model              │
      │ - meta.json              │   │ - cluster names          │
      └─────────────┬────────────┘   └─────────────┬────────────┘
                    └──────────────┬───────────────┘
                                   ▼
                     ┌────────────────────────────┐
                     │ Streamlit App              │
                     │ - KPI dashboard            │
                     │ - EDA explorer             │
                     │ - Model leaderboard        │
                     │ - Price prediction         │
                     │ - Segment prediction       │
                     │ - Cluster insights         │
                     └────────────────────────────┘
```

---

## 🗂️ Repository Structure

```text
diamond-dynamics/
│
├── diamond_eda_modelling.ipynb
├── model_training.py
├── app.py
│
├── models/
│   ├── best_regression_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler_regression.pkl
│   ├── scaler_clustering.pkl
│   ├── label_encoders.pkl
│   ├── pca_model.pkl
│   ├── selected_features.pkl
│   ├── cluster_profiles.csv
│   ├── regression_results.csv
│   ├── vif_report.csv
│   └── meta.json
│
├── plots/
│   ├── 01_price_distribution.png
│   ├── 02_carat_distribution.png
│   ├── 03_cut_count.png
│   ├── 04_color_count.png
│   ├── 05_clarity_count.png
│   ├── 06_price_vs_carat_scatter.png
│   ├── 07_price_by_cut_boxplot.png
│   ├── 08_price_by_color_boxplot.png
│   ├── 09_price_by_clarity_boxplot.png
│   ├── 10_correlation_heatmap.png
│   ├── 11_outlier_boxplots.png
│   ├── 12_skewness_before_after.png
│   ├── 13_feature_importances.png
│   ├── 14_model_comparison.png
│   ├── 15_ann_training_history.png
│   ├── 16_elbow_method.png
│   ├── 17_silhouette_scores.png
│   ├── 18_pca_cluster_2d.png
│   ├── 19_cluster_profiles.png
│   ├── 20_pairplot_sample.png
│   ├── avg_price_by_cut.png
│   ├── avg_price_by_color.png
│   ├── avg_price_by_clarity.png
│   └── volume_vs_price.png
│
├── data/
│   ├── diamonds.csv
│   ├── diamonds_clean_featured.csv
│   ├── diamonds_processed_clustered.csv
│   └── .gitkeep
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset Description

The dataset contains diamond characteristics commonly used in pricing analysis.

| Column | Type | Description |
|---|---|---|
| carat | Numeric | Weight of the diamond |
| cut | Categorical (ordinal) | Quality of cut |
| color | Categorical (ordinal) | Diamond color grade |
| clarity | Categorical (ordinal) | Clarity grade |
| depth | Numeric | Total depth percentage |
| table | Numeric | Width of the top relative to widest point |
| price | Numeric | Price in USD |
| x | Numeric | Length in mm |
| y | Numeric | Width in mm |
| z | Numeric | Depth in mm |

---

## ⚙️ Feature Engineering

The project creates business-aware and geometry-aware features to improve prediction quality.

| # | Feature | Description | Business Value |
|---|---|---|---|
| 1 | `price_inr` | USD to INR conversion using 83.5 | Makes predictions market-ready for Indian pricing |
| 2 | `volume` | `x * y * z` | Physical size proxy |
| 3 | `price_per_carat` | `price / carat` | Value efficiency metric |
| 4 | `dimension_ratio` | `(x + y) / (2*z)` | Shape/proportion indicator |
| 5 | `carat_category` | Light / Medium / Heavy | Commercial segmentation label |
| 6 | `surface_area` | Surface area proxy from dimensions | Enhanced geometry representation |
| 7 | `depth_table_ratio` | `depth / table` | Approximate cut-proportion signal |
| 8 | `lw_ratio` | `x / y` | Symmetry indicator |
| 9 | `is_premium` | Ideal + D/E + IF/VVS | Premium luxury flag |
| 10 | `log_price` / `log_price_inr` | Log-transformed target | Stabilizes skew for regression |

---

## 📊 Model Results

> These values are automatically saved into `models/meta.json` and `models/regression_results.csv` after running training.

| Rank | Model | R² | RMSE (₹) | MAE (₹) | MAPE |
|---|---|---:|---:|---:|---:|
| 1 | XGBoost / Best Model | _Auto-filled after training_ | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ |
| 2 | Random Forest | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ |
| 3 | ANN | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ |
| 4 | Decision Tree | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ |
| 5 | Ridge Regression | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ |
| 6 | Linear Regression | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ |
| 7 | KNN | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ | _Auto-filled_ |

### Evaluation Metrics Used

- **MAE** — average absolute error in INR
- **MSE** — squared error penalty
- **RMSE** — interpretable average deviation magnitude
- **R²** — goodness of fit
- **MAPE** — average percentage error

---

## 🔮 Clustering Results

KMeans clustering is used to group diamonds into commercial segments using physical and quality attributes only.

| Cluster | Proposed Name | Typical Characteristics | Business Interpretation |
|---|---|---|---|
| 0 | 💎 Premium Heavy Diamonds | High carat, higher value, premium quality mix | Luxury-focused merchandising |
| 1 | 🪨 Affordable Small Diamonds | Lower carat, smaller dimensions, lower price zone | Entry-level and volume inventory |
| 2 | ⚖️ Mid-Range Balanced Diamonds | Moderate size and quality | Broad market appeal |
| 3 | ✨ Quality Light Diamonds | Lower weight but attractive quality profile | Precision upsell / gifting segment |

### Clustering Workflow
- Standardize clustering features
- Test K from 2 to 10
- Compare **Elbow Method** and **Silhouette Score**
- Fit final **KMeans**
- Use **PCA** for 2D visualization
- Build cluster profiles and assign business-friendly names

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diamond-dynamics.git
cd diamond-dynamics
```

### 2. Create a virtual environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the full pipeline

```bash
python model_training.py
```

This step will:
- Download the dataset automatically
- Generate all plots
- Train all regression models
- Train the ANN
- Run clustering
- Save all artifacts to `models/`, `plots/`, and `data/`

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## ☁️ Deployment

This project is ready for **Streamlit Community Cloud** deployment.

### Deploy steps
1. Push the repository to GitHub
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Click **Create app**
4. Select your repository, branch, and `app.py` as the entry file
5. Ensure `requirements.txt` is present in the repo root
6. Deploy

### Important deployment notes
- Keep `requirements.txt` in the same repository as `app.py`
- Make sure trained model artifacts exist in `models/`
- If deploying a fresh repo, either:
  - commit the generated artifacts, or
  - run training once before deployment and upload outputs

---

## 🛠️ Tech Stack

| Layer | Tools / Libraries |
|---|---|
| Language | Python |
| Data handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Classical ML | scikit-learn |
| Boosting | XGBoost |
| Deep Learning | TensorFlow / Keras |
| Clustering | KMeans, PCA |
| Statistics | SciPy, statsmodels |
| Serialization | joblib, JSON |
| App framework | Streamlit |
| Data download | gdown |

---

## ✨ Highlights

- End-to-end ML pipeline from raw CSV to deployment-ready app
- Recruiter-grade repository structure
- 19+ saved visualizations for storytelling
- Multiple regression baselines plus ANN
- Unsupervised clustering with business naming
- INR-ready prediction output

---

## 👤 Author

Pratyusha Sharma

![GitHub]: (https://github.com/ps-learner)
![LinkedIn]: (https://linkedin.com/in/pratyusha-sharma-46b038324)

---

## 📄 License

This project is licensed under the **MIT License**.

You may use, modify, and distribute it with attribution under the terms of the license.

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```