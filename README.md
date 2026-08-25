# Fraud Analyst Copilot

An LLM-based copilot that explains flagged fraud transactions in plain English, built as Phase 1 of a two-phase Engineering Project (8 credits).

## What it does

Takes a transaction flagged as potential fraud by a trained classifier and generates a short, plain-English explanation of *why* it was flagged — grounded in SHAP feature attributions rather than the LLM guessing. Built for a bank fraud analyst who needs to quickly understand a model's reasoning without reading raw SHAP values or feature codes.

## Pipeline

```
Transaction → XGBoost classifier → SHAP TreeExplainer → 
structured feature dict (top drivers + values) → 
LLM prompt (Gemini) → plain-English explanation
```

1. **Model**: XGBoost classifier trained on preprocessed transaction data (SMOTE-balanced, StandardScaler-normalized). Accuracy 0.999, ROC-AUC 0.983, recall 0.88 on the minority (fraud) class.
2. **Explainability**: SHAP `TreeExplainer` computes exact per-transaction feature attributions.
3. **Structuring**: Top contributing features (name, SHAP impact, actual value) are extracted into a clean dict per transaction.
4. **Generation**: The dict is formatted into a prompt and sent to Gemini, which returns a 3-4 sentence explanation suitable for a non-technical analyst.
5. **Output**: Single-transaction function (`analyze_transaction`) and a batch mode that processes multiple flagged transactions into a CSV of probability + explanation pairs.

## Example output

> This transaction has been flagged with an extremely high fraud probability of 99.2%. The primary drivers behind this risk score are severe abnormal patterns in key transaction features, which closely match known fraudulent behavior profiles. While one minor indicator slightly pushed toward legitimacy, it is heavily outweighed by these strong risk signals. We recommend prioritizing this account for immediate review or temporary block.

## Scope and limitations (stated plainly)

- **Demo-scale, not production**: runs on a fixed, static test set — no live/streaming transaction ingestion.
- **PCA-anonymized features**: the dataset's features (V1–V28) are PCA components with no real-world meaning. Explanations describe *statistical deviation* from normal patterns, not real-world reasons like "unusual location" or "high amount." This is an honest constraint of the dataset, not the pipeline.
- **No human-in-the-loop feedback**: analyst corrections or overrides aren't fed back into the model.
- **No monitoring or retraining loop**: that's out of scope for this phase.
- **Batch demo covers a representative sample** (8-10 flagged transactions), not the full flagged set — sufficient to prove the pipeline works end-to-end.

## Roadmap

This is **Phase 1** of a two-phase Engineering Project:

- **Phase 1 (current)**: This demo — SHAP-grounded LLM explanations for flagged transactions.
- **Phase 2 (next semester)**: Real-time MLOps pipeline on AWS — streaming ingestion (Kinesis), inference (Lambda/Fargate/SageMaker), drift monitoring (Evidently AI), automated retraining (Step Functions), and observability (CloudWatch).

## Stack

XGBoost, SHAP, Google Gemini API (`gemini-3.1-flash-lite`), pandas, scikit-learn, Google Colab
