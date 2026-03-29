import os
import numpy as np
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()
INCEPTION_API_KEY = os.getenv("INCEPTION_API_KEY")

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
_MODELS = {}

def load_model(model_type: str = "xgb"):
    """Cache and return model + scaler + feature names."""
    global _MODELS
    if model_type not in _MODELS:
        model_path = f"model/{'rf_model' if model_type == 'rf' else 'xgb_model'}.pkl"
        if not os.path.exists(model_path):
            model_path = "model/model.pkl"

        model = joblib.load(model_path)
        scaler = joblib.load("model/scaler.pkl")
        feature_names = joblib.load("model/feature_names.pkl")
        _MODELS[model_type] = (model, scaler, feature_names)

    return _MODELS[model_type]


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_toxicity(input_data: list, model_type: str = "xgb"):
    """
    Predict toxicity for a single compound.
    
    Args:
        input_data: list of feature values (in the same order as feature_names)
        model_type: 'xgb' or 'rf'
    
    Returns:
        dict: {
            'prediction': 0 or 1,
            'label': 'Toxic' or 'Non-Toxic',
            'probability': float (0-1),
            'risk_level': 'High' | 'Medium' | 'Low',
            'top_features': [(feature_name, importance), ...]
        }
    """
    model, scaler, feature_names = load_model(model_type)

    arr = np.array(input_data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    pred = int(model.predict(arr_scaled)[0])
    prob = float(model.predict_proba(arr_scaled)[0][1])

    # Risk level
    if prob >= 0.7:
        risk = "High"
    elif prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # Top contributing features via feature importance
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [(feature_names[i], float(importances[i])) for i in top_idx]

    return {
        "prediction": pred,
        "label": "Toxic" if pred == 1 else "Non-Toxic",
        "probability": prob,
        "risk_level": risk,
        "top_features": top_features,
    }


def batch_predict(df, model_type: str = "xgb"):
    """
    Run toxicity prediction on a DataFrame.
    Returns the df with new columns: prediction, probability, risk_level.
    """
    model, scaler, feature_names = load_model(model_type)

    # Align columns
    available = [col for col in feature_names if col in df.columns]
    X = df[available].fillna(0)

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    result_df = df.copy()
    result_df["prediction"] = preds
    result_df["label"] = ["Toxic" if p == 1 else "Non-Toxic" for p in preds]
    result_df["probability"] = np.round(probs, 4)
    result_df["risk_level"] = [
        "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
        for p in probs
    ]

    return result_df


# ─────────────────────────────────────────────
# INCEPTION API INTEGRATION
# ─────────────────────────────────────────────
INCEPTION_BASE_URL = "https://api.inceptionlabs.ai/v1"

def get_inception_insights(
    compound_features: dict,
    probability: float,
    label: str,
    top_features: list
) -> str:
    """
    Call Inception API (Mercury model) to get AI-driven drug safety insights.
    
    Args:
        compound_features: dict of {feature_name: value}
        probability: toxicity probability (0-1)
        label: 'Toxic' or 'Non-Toxic'
        top_features: list of (feature_name, importance) tuples
    
    Returns:
        str: AI-generated insight text
    """
    if not INCEPTION_API_KEY:
        return "⚠️ Inception API key not configured. Add INCEPTION_API_KEY to .env"

    top_feat_str = "\n".join(
        [f"  - {name}: importance={imp:.4f}" for name, imp in top_features[:5]]
    )
    feat_str = "\n".join(
        [f"  - {k}: {v}" for k, v in list(compound_features.items())[:10]]
    )

    prompt = f"""You are a drug safety AI assistant specializing in computational toxicology.

A chemical compound has been analyzed by our ML model (trained on Tox21 dataset):
- Prediction: {label}
- Toxicity Probability: {probability:.1%}

Top contributing molecular features:
{top_feat_str}

Key compound descriptors:
{feat_str}

Based on this analysis, provide:
1. A brief interpretation of the toxicity prediction (2-3 sentences)
2. Which molecular properties are most concerning and why
3. Potential toxicity mechanism (e.g., hepatotoxicity, genotoxicity, cardiotoxicity)
4. Drug development recommendations (2-3 bullet points)

Keep your response concise, scientific, and actionable. Format with clear sections."""

    try:
        headers = {
            "Authorization": f"Bearer {INCEPTION_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mercury-2",
            "messages": [
                {"role": "system", "content": "You are a computational toxicology expert."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 600,
            "temperature": 0.3
        }
        resp = requests.post(
            f"{INCEPTION_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not reach Inception API: {e}"
    except (KeyError, IndexError) as e:
        return f"⚠️ Unexpected API response format: {e}"
