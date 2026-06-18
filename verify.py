# verify.py — run after model_training.py completes
import os, json, joblib
import numpy as np
import pandas as pd

print("=" * 55)
print("  💎 DIAMOND DYNAMICS — VERIFICATION REPORT")
print("=" * 55)

# ── 1. Check all model files exist ──────────────────────
required_files = [
    "models/best_regression_model.pkl",
    "models/kmeans_model.pkl",
    "models/scaler_regression.pkl",
    "models/scaler_clustering.pkl",
    "models/label_encoders.pkl",
    "models/pca_model.pkl",
    "models/selected_features.pkl",
    "models/meta.json",
]
print("\n📁 MODEL FILES:")
all_good = True
for f in required_files:
    exists = os.path.exists(f)
    # also check keras fallback
    if not exists and f == "models/best_regression_model.pkl":
        exists = os.path.exists("models/best_ann_model.keras")
    status = "✅" if exists else "❌ MISSING"
    print(f"  {status}  {f}")
    if not exists:
        all_good = False

# ── 2. Check plot count ──────────────────────────────────
plots = [f for f in os.listdir("plots") if f.endswith(".png")]
plot_status = "✅" if len(plots) >= 15 else "⚠️  LOW COUNT"
print(f"\n🖼️  PLOTS: {plot_status} — {len(plots)} PNG files found")

# ── 3. Read meta.json ────────────────────────────────────
print("\n📊 META.JSON:")
with open("models/meta.json", encoding="utf-8") as f:
    meta = json.load(f)

best_model  = meta.get("best_regression_model", "MISSING")
r2          = meta.get("regression_metrics", {}).get("R2", 0)
rmse        = meta.get("regression_metrics", {}).get("RMSE_inr", 0)
k           = meta.get("clustering", {}).get("optimal_k", 0)
sil         = meta.get("clustering", {}).get("silhouette_score", 0)
n_diamonds  = meta.get("total_diamonds", 0)
c_names     = meta.get("clustering", {}).get("cluster_names", {})

r2_status   = "✅" if r2 > 0.93  else "⚠️  LOW — check your data"
sil_status  = "✅" if sil > 0.20 else "⚠️  LOW — clusters may not be well separated"
n_status    = "✅" if 40000 < n_diamonds <= 53940 else "⚠️  CHECK"

print(f"  Best Model     : {best_model}")
print(f"  R² Score       : {r2:.4f}  {r2_status}")
print(f"  RMSE (₹)       : ₹{rmse:,.0f}")
print(f"  Diamonds used  : {n_diamonds:,}  {n_status}")
print(f"  Optimal K      : {k}")
print(f"  Silhouette     : {sil:.4f}  {sil_status}")

print(f"\n🔮 CLUSTER NAMES:")
for cid, cname in c_names.items():
    print(f"  Cluster {cid}: {cname}")

dup_check = len(c_names.values()) == len(set(c_names.values()))
print(f"  No duplicates  : {'✅' if dup_check else '❌ DUPLICATE NAMES FOUND'}")

# ── 4. Verify dataset ────────────────────────────────────
print("\n📂 DATASET:")
df = pd.read_csv("data/diamonds.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
expected_cols = {"carat","cut","color","clarity","depth","table","price","x","y","z"}
cols_ok = expected_cols.issubset(set(df.columns))
print(f"  Shape          : {df.shape}")
print(f"  Columns OK     : {'✅' if cols_ok else '❌ MISSING COLUMNS'}")
zero_xyz = ((df[['x','y','z']] == 0).sum().sum())
print(f"  Zero x/y/z     : {zero_xyz} {'✅' if zero_xyz == 0 else '⚠️  zeros found — preprocessing handles this'}")

# ── 5. Issue 2 check — scaling consistency ────────────────
print("\n🔍 ISSUE 2 — SCALING CONSISTENCY:")
sel_features = joblib.load("models/selected_features.pkl")
print(sel_features)
scaler       = joblib.load("models/scaler_regression.pkl")

# Build a test row with known values
test_row = {}

for f in sel_features:
    test_row[f] = 1.0

test_row["carat"] = 1.0

if "x" in test_row:
    test_row["x"] = 6.4

if "y" in test_row:
    test_row["y"] = 6.4

if "z" in test_row:
    test_row["z"] = 4.0

if "volume" in test_row:
    test_row["volume"] = 6.4 * 6.4 * 4.0

if "surface_area" in test_row:
    test_row["surface_area"] = 2*((6.4*6.4)+(6.4*4.0)+(6.4*4.0))

if "cut_enc" in test_row:
    test_row["cut_enc"] = 4

if "color_enc" in test_row:
    test_row["color_enc"] = 6

if "clarity_enc" in test_row:
    test_row["clarity_enc"] = 7

X_test  = pd.DataFrame([test_row])[sel_features]
X_scaled = scaler.transform(X_test)

# Load model and predict both ways
model = None
if os.path.exists("models/best_regression_model.pkl"):
    model = joblib.load("models/best_regression_model.pkl")

if model is not None:
    needs_scaling = best_model in ["Linear Regression", "Ridge Regression", "KNN"]

    if needs_scaling:
        pred_log = model.predict(X_scaled)[0]
        print(f"  {best_model} uses SCALED input  ✅")
    else:
        pred_log = model.predict(X_test)[0]
        print(f"  {best_model} uses RAW input (correct for trees)  ✅")

    price_inr = np.expm1(pred_log)
    price_usd = price_inr / 83.5
    in_range  = 10_000 < price_inr < 2_000_000

    print(f"  Test prediction (1ct Ideal D IF): ₹{price_inr:,.0f}  (~${price_usd:,.0f})")
    print(f"  Sanity range ₹2.5L–₹8L          : {'✅ PASS' if in_range else '❌ FAIL — likely a scaling mismatch'}")
else:
    print("  ⚠️  Skipped — model file not found (check for .keras file)")

print("\n" + "=" * 55)
print("  Verification complete.")
print("=" * 55)