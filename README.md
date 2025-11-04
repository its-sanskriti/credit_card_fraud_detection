🧠 Project Overview

This project demonstrates the end-to-end data preprocessing pipeline — from handling missing values and scaling to feature transformation and model evaluation — using the Credit Card Fraud Detection Dataset.
It’s designed as a learning guide for anyone diving deep into data preprocessing, emphasizing how data quality impacts model performance.

📂 Dataset

Source: Kaggle - Credit Card Fraud Detection

Shape: 284,807 rows × 31 columns

Features V1–V28 are PCA-transformed components

Amount represents transaction value

Class is the target (0 → Genuine, 1 → Fraud)

1️⃣ Data Loading & Initial Exploration

Loaded dataset using Pandas

Checked shape, data types, and null values

No missing values found ✅
2️⃣ Data Imbalance Analysis

Verified class distribution (fraud vs. genuine)

Severe imbalance observed: Fraud ≈ 0.17% of total transactions

Used SMOTE later to synthetically balance classes

3️⃣ Outlier & Distribution Analysis

Visualized Amount column using boxplots and histograms

Identified heavy right-skew → opted for log transformation

4️⃣ Transformation

Applied log transformation to Amount

Resulted in near-Gaussian distribution

Confirmed via describe() summary and visualization

📏 5️⃣ Feature Scaling

Used StandardScaler for normalized Gaussian-like data

Resulting Amount_scaled feature:

Mean ≈ 0

Std ≈ 1

Gaussian-shaped distribution ✅

🧩 6️⃣ Splitting Dataset

Split into X_train, X_test, y_train, y_test (80:20 ratio)

Used stratify=y to maintain class ratio

🧠 7️⃣ Model Training (Baseline)

Trained Logistic Regression on SMOTE-balanced data

Tuned max_iter=1000 to avoid convergence warning

Evaluated with accuracy, recall, F1, and ROC-AUC metrics

📊 8️⃣ Model Evaluation

Confusion Matrix:

True Negatives: ✅ 56,267

False Positives: ⚠️ 597

False Negatives: ❌ 9

True Positives: ✅89
Metrics Summary:

Metric	Score
Accuracy	0.99
Recall (Fraud)	0.91
ROC-AUC	0.97
9️⃣ Precision–Recall & ROC Curves

Visualized Confusion Matrix, ROC Curve, and Precision-Recall Curve

PR curve offers clearer insights for imbalanced data

Observed strong recall with moderate precision — ideal tradeoff for fraud detection
