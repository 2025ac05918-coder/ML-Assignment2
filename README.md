# 2025ac05918 ML-Assignment2
# Bank Marketing Classification Project

## a. Problem Statement
The objective of this project is to predict whether a client will subscribe to a term deposit (binary classification: 'yes' or 'no') based on a variety of social and economic attributes. By accurately predicting customer behavior, banking institutions can optimize their marketing strategies, reduce costs, and increase the conversion rate of their telemarketing campaigns.

## b. Dataset Description
- **Source:** [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Instance Size:** 45,211 rows
- **Feature Size:** 17 features
- **Features Included:** 
    - Age, Job, Marital Status, Education, Default, Balance, Housing, Loan, Contact, Day, Month, Duration, Campaign, Pdays, Previous, Poutcome.
- **Target Variable:** `y` (Has the client subscribed to a term deposit?)

## c. Github Repository Link
**Link:** [INSERT_YOUR_GITHUB_REPO_LINK_HERE]

## d. Models Used
The following metrics were calculated using the test split of the dataset.

### Comparison Table
| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.89 | 0.82 | 0.58 | 0.20 | 0.30 | 0.29 |
| Decision Tree | 0.87 | 0.70 | 0.46 | 0.47 | 0.46 | 0.38 |
| kNN | 0.88 | 0.74 | 0.51 | 0.28 | 0.36 | 0.31 |
| Naive Bayes | 0.82 | 0.79 | 0.34 | 0.51 | 0.41 | 0.32 |
| Random Forest (Ensemble) | 0.90 | 0.92 | 0.65 | 0.38 | 0.48 | 0.45 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provides a strong baseline for accuracy but struggles significantly with recall on the minority class. |
| **Decision Tree** | Captures non-linear relationships better than Logistic Regression but is prone to overfitting. |
| **kNN** | Performance is sensitive to the choice of 'k' and requires scaled data; results were average. |
| **Naive Bayes** | Highest recall for the minority class, making it useful if missing a potential customer is costly. |
| **Random Forest (Ensemble)** | Best overall performance. It handles the imbalanced nature of the dataset most effectively with the highest AUC. |

### Overall Winner
The **Random Forest Classifier** is the overall winner for this dataset, as it achieved the highest Accuracy (0.90), AUC (0.92), and MCC Score (0.45), indicating the most robust predictive power across all classes.

---

## How to Run the App
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the Streamlit app: `streamlit run app.py`.
4. Upload `test_data.csv` to see model evaluations.
