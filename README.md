# Hyperuricemia Health Agent 🩺

Conversational web agent for hyperuricemia (HUA) risk assessment, built on an interpretable machine-learning model.

**English / 中文 bilingual** · Medical Q&A · Free-text lab-value input · Lab-report photo OCR · Guided 15-item assessment · SHAP explanation

## About the model

- Soft-voting ensemble of 6 algorithms (GBM, CatBoost, XGBoost, Random Forest, Logistic Regression, AdaBoost)
- 15 features selected by LASSO ∩ Boruta ∩ RFECV on a Hainan community cohort (n = 6,779 adults)
- Internal test AUC **0.810** (0.762–0.858); geographic external validation AUC **0.810** (0.746–0.874)
- Youden cutoff 0.233; Brier 0.123

⚠️ **Disclaimer**: This agent is for health reference and research demonstration only. It is **not medical advice** and does not replace a physician's diagnosis.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app (bilingual Q&A, metric parsing, OCR, guided assessment) |
| `kb_data.py` | Bilingual knowledge base (53 entries) |
| `webapp_models.pkl` | Fitted 6-model ensemble + preprocessing pipeline |
| `medians_bg.pkl` | Population medians + 10 k-means cluster centroids of the standardized training distribution (aggregate statistics only, no individual-level data) |
