# 2025ac05918 ML-Assignment2
# Bank Marketing Classification

a. Problem Statement
The goal is to predict if a client will subscribe to a term deposit (variable 'y') based on marketing campaign data. This is a binary classification problem aimed at increasing campaign efficiency.

b. Dataset Description

Source: UCI Machine Learning Repository (Bank Marketing Dataset).
Instances: 45,211.
Features: 17 (Including Age, Job, Marital Status, Education, Balance, Housing, Loan, etc.).
d. Models Comparison Table

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.89	0.82	0.58	0.20	0.30	0.29
Decision Tree	0.87	0.70	0.46	0.47	0.46	0.38
kNN	0.88	0.74	0.51	0.28	0.36	0.31
Naive Bayes	0.82	0.79	0.34	0.51	0.41	0.32
Random Forest	0.90	0.92	0.65	0.38	0.48	0.45
(Note: These values are estimates based on standard dataset performance; replace with your actual script output.)

Observations

Logistic Regression: Good baseline but suffers from low recall on the minority class.
Decision Tree: Highly prone to overfitting, showing high variance.
kNN: Computationally expensive during inference; performance is average.
Naive Bayes: Best Recall for the "Yes" class, but lowest overall accuracy.
Random Forest: Best performer across most metrics, specifically AUC and Accuracy.
Overall Winner: Random Forest (Highest AUC and F1 Score).
