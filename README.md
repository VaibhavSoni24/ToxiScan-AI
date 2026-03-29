# ToxiScan AI

Drug toxicity prediction app built with Streamlit, scikit-learn, XGBoost, and SHAP.

ToxiScan AI predicts whether a compound is likely Toxic or Non-Toxic using a model trained on the Tox21 dataset. It supports single-compound input, batch CSV scoring, model comparison, and optional AI-generated interpretation via the Inception API.

## Highlights

- Single-compound prediction with probability and risk level (Low, Medium, High)
- Batch CSV prediction for many compounds at once
- Two model choices in the UI: XGBoost and Random Forest
- Feature-importance and SHAP visualization support
- Optional Inception API summary for model output interpretation
- Built-in analytics and comparison dashboards

## Demo Pages (Streamlit App)

- Predict: manual entry of descriptor values
- Batch Upload: upload CSV, score compounds, download results
- Analytics: feature importance and SHAP/dataset image panels
- Model Comparison: RF vs XGB metrics and architecture summary
- About: system overview and stack summary

## Repository Structure

```text
ToxiScan AI/
├─ app.py
├─ README.md
├─ requirements.txt
├─ data/
│  └─ .gitkeep
├─ model/
│  ├─ __init__.py
│  ├─ feature_names.pkl
│  ├─ metrics.json
│  ├─ model.pkl
│  ├─ predict.py
│  ├─ rf_model.pkl
│  ├─ scaler.pkl
│  ├─ train.py
│  └─ xgb_model.pkl
└─ notebooks/
   └─ ToxiScan_Training.ipynb
```

Note: files like model artifacts and charts may be regenerated depending on your training flow.

## How It Works

1. Load Tox21 data (KaggleHub download, with local CSV fallback).
2. Build target label `is_toxic` from assay columns.
3. Train both Random Forest and XGBoost models.
4. Save model artifacts to `model/`.
5. Serve inference through Streamlit UI (`app.py`).
6. Optionally call Inception API for narrative toxicity insights.

## Requirements

- Python 3.10+
- pip
- Internet access for KaggleHub dataset download (or local CSV in `data/tox21.csv`)

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root if you want AI-generated text insights:

```env
INCEPTION_API_KEY=your_api_key_here
```

If the key is missing, predictions still work; only the Inception insight panel is disabled.

## Train Models

### Option A: Notebook (recommended for experimentation)

Open and run:

- `notebooks/ToxiScan_Training.ipynb`

### Option B: CLI script

```bash
python model/train.py
```

The script trains both models and saves artifacts to `model/`:

- `rf_model.pkl`
- `xgb_model.pkl`
- `model.pkl` (default fallback)
- `scaler.pkl`
- `feature_names.pkl`

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Batch CSV Format

For batch mode, your CSV should contain feature columns matching the names in `model/feature_names.pkl`.

App output columns include:

- `prediction` (0/1)
- `label` (`Toxic` or `Non-Toxic`)
- `probability` (class-1 probability)
- `risk_level` (`Low`, `Medium`, `High`)

Tip: the Batch Upload page provides a sample template download.

## Metrics

Current metrics are read from `model/metrics.json` and displayed in the UI.

At the time of writing, `model/metrics.json` contains:

- Random Forest: accuracy 1.000, ROC-AUC 1.000, F1 1.000
- XGBoost: accuracy 1.000, ROC-AUC 1.000, F1 1.000
- Samples: 7,831
- Features: 12

These values can change after retraining with different data splits or preprocessing.

## Generated Visual Assets

The Analytics page can display the following files if present in project root:

- `shap_summary.png`
- `shap_waterfall.png`
- `eda_overview.png`
- `model_evaluation.png`

If they are missing, run the notebook/training flow that generates them.

## Troubleshooting

- Models not found in app:
  - Train with `python model/train.py` or run the notebook and place artifacts in `model/`.
- Dataset download fails:
  - Put `tox21.csv` in `data/` and rerun training.
- Inception insight errors:
  - Check `INCEPTION_API_KEY` and network access.
- Batch upload fails:
  - Ensure CSV columns align with `feature_names.pkl`.

## Disclaimer

This project is for research/educational use only and is not a medical or regulatory decision system.
