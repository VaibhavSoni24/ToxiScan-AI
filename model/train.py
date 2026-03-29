# ToxiScan AI — Model Training Script (Standalone CLI)
import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_tox21(path: str = "data/tox21.csv"):
    """Load and return cleaned Tox21 dataframe."""
    try:
        import kagglehub
        dl_path = kagglehub.dataset_download("epicskills/tox21-dataset")
        import glob
        csv_files = glob.glob(os.path.join(dl_path, "**/*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError("No CSV found in dataset download.")
        df = pd.read_csv(csv_files[0])
        print(f"[✓] Loaded dataset from KaggleHub: {csv_files[0]}")
    except Exception as e:
        print(f"[!] KaggleHub failed ({e}), trying local file: {path}")
        df = pd.read_csv(path)

    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns[:10])} ...")
    return df


def preprocess(df: pd.DataFrame):
    """
    Preprocess the Tox21 dataset.

    The Kaggle Tox21 dataset has 12 binary assay columns (NR-AR, NR-AhR, etc.)
    plus 'mol_id' and 'smiles'. There are NO pre-computed molecular descriptors.

    Strategy:
      - Features  = the 12 assay columns (fill NaN with 0 = 'not tested')
      - Target    = is_toxic: 1 if **any** assay is positive, else 0
    """
    ASSAY_COLS = [c for c in df.columns if c not in ("mol_id", "smiles")]

    # Create binary target: toxic in at least one assay
    df = df.copy()
    df["is_toxic"] = (df[ASSAY_COLS].fillna(0).max(axis=1) > 0).astype(int)
    target_col = "is_toxic"

    X = df[ASSAY_COLS].fillna(0)
    y = df[target_col].astype(int)

    # Drop rows with NaN target (safety-net)
    mask = y.notna()
    X, y = X[mask], y[mask]

    print(f"[✓] Features : {X.shape[1]} assay columns")
    print(f"    Samples  : {X.shape[0]}")
    print(f"    Class balance: {y.value_counts().to_dict()}")

    # Scale (keeps RF/XGBoost happy with standardised inputs)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, X_scaled, scaler, list(X.columns)


# ─────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    print("\n[→] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("[✓] Random Forest trained.")
    return rf


def train_xgboost(X_train, y_train):
    print("\n[→] Training XGBoost ...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    print("[✓] XGBoost trained.")
    return xgb


# ─────────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)
    print(f"\n{'='*50}")
    print(f"{name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(classification_report(y_test, preds, target_names=["Non-Toxic", "Toxic"]))
    return {"name": name, "accuracy": acc, "roc_auc": roc}


# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    os.makedirs("assets", exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#0A0F1E")
    fig.patch.set_facecolor("#0A0F1E")
    colors = ["#00E5A0" if i < top_n // 3 else "#7FFFD4" if i < 2 * top_n // 3 else "#4FC3F7"
              for i in range(top_n)]
    bars = ax.barh(range(top_n), top_importances[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], color="white", fontsize=9)
    ax.set_xlabel("Importance Score", color="white")
    ax.set_title(title, color="#00E5A0", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2A3A")
    plt.tight_layout()
    path = f"assets/feature_importance_{title.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {path}")
    return path


# ─────────────────────────────────────────────
# 5. SAVE MODELS
# ─────────────────────────────────────────────
def save_models(rf_model, xgb_model, scaler, feature_names):
    os.makedirs("model", exist_ok=True)
    joblib.dump(rf_model, "model/rf_model.pkl")
    joblib.dump(xgb_model, "model/xgb_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(feature_names, "model/feature_names.pkl")
    joblib.dump(xgb_model, "model/model.pkl")  # default model = XGBoost (better AUC)
    print("[✓] Models saved to model/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_tox21()
    X, y, X_scaled, scaler, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb, X_test, y_test, "XGBoost")

    plot_feature_importance(rf, feature_names, title="Random Forest Feature Importance")
    plot_feature_importance(xgb, feature_names, title="XGBoost Feature Importance")

    save_models(rf, xgb, scaler, feature_names)
    print("\n[✓] Training complete! Run: streamlit run app.py")
