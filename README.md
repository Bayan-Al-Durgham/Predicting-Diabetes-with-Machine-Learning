🩺 Diabetes Prediction with Multi-Model Explainable AI (SHAP)
📌 Project Overview
This project is an end-to-end machine learning pipeline for predicting diabetes using clinical features, with a strong emphasis on model explainability through SHAP (Shapley Additive Explanations).

The pipeline covers data preprocessing, model training, evaluation, performance comparison, and interpretability across seven machine learning algorithms.

Dataset: Pima Indians Diabetes Database — Kaggle

🎯 Objectives
Predict whether a patient has diabetes (1) or not (0) using clinical and demographic data.

Compare multiple classifiers on key performance metrics.

Integrate SHAP explainability for each model to provide transparent and interpretable insights.

📊 Dataset Description
Feature	Description
Pregnancies	Number of pregnancies
Glucose	Plasma glucose concentration after 2 hours (OGTT)
BloodPressure	Diastolic blood pressure (mm Hg)
SkinThickness	Triceps skin fold thickness (mm)
Insulin	2-hour serum insulin (mu U/ml)
BMI	Body Mass Index ((weight in kg / height in m²))
DiabetesPedigreeFunction	Diabetes likelihood score based on family history
Age	Age in years
Outcome	Target variable: 0 = no diabetes, 1 = diabetes

🛠️ Workflow
Data Loading & Inspection

Check for null values, duplicates, and invalid entries (e.g., zero as missing).

Data Cleaning

Replace invalid zeros with mean (normal distribution) or median (skewed distribution).

Exploratory Data Analysis (EDA)

Histograms, boxplots, and correlation heatmap to understand feature distribution and relationships.

Class Imbalance Handling

Applied RandomOverSampler to balance positive and negative cases.

Model Training & Evaluation

Models used:

Logistic Regression

Support Vector Classifier (SVC)

Random Forest Classifier

Gradient Boosting Classifier

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Decision Tree Classifier

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Model Comparison

Plotted Accuracy, Precision, Recall, and F1-score for all models.

Explainable AI (SHAP)

TreeExplainer: RandomForest, GradientBoosting, DecisionTree

LinearExplainer: LogisticRegression

KernelExplainer: SVC, KNN, GaussianNB

Generated beeswarm and bar plots for each model showing top contributing features.

📈 Results
Top features across models: Glucose, BMI, Age

Class balancing improved Recall — crucial for medical screening use-cases.

SHAP plots provide clinician-friendly insights into how each feature influences predictions.
📦 Requirements
Python 3.8+

pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

shap
