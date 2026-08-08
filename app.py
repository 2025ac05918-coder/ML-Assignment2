import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Machine Learning Classification Dashboard")

# 1. Dataset Upload
uploaded_file = st.file_uploader("Upload your test_data.csv", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]

    # 2. Model Selection
    model_option = st.selectbox(
        'Select Model to Evaluate',
        ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest")
    )

    # Load Model
    model_path = f"model/{model_option.replace(' ', '_').lower()}.pkl"
    model = joblib.load(model_path)

    # 3. Predictions
    y_pred = model.predict(X_test)

    # 4. Display Metrics
    st.subheader(f"Metrics for {model_option}")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

    # 5. Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
