import os
import pathlib
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bank Marketing - ML Models", layout="wide")
st.title("Bank Marketing - ML Model Comparison Dashboard")
st.markdown("This application downloads data, trains five classifiers, and presents evaluation metrics interactively.")

# --- SETUP DIRECTORIES ---
PROJECT_ROOT = pathlib.Path(os.getcwd())
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
TEST_DATA_FILE = PROJECT_ROOT / "test_data.csv"

# --- MODEL FILE PATHS ---
SCALER_FILE = MODEL_DIR / "scaler.pkl"
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "Logistic_Regression.pkl",
    "Decision Tree": MODEL_DIR / "Decision_Tree.pkl",
    "kNN": MODEL_DIR / "kNN.pkl",
    "Naive Bayes": MODEL_DIR / "Naive_Bayes.pkl",
    "Random Forest": MODEL_DIR / "Random_Forest.pkl"
}

def train_and_save_models():
    st.info("Training models since no pre-trained artifacts were found...")
    try:
        k_path = kagglehub.dataset_download('janiobachmann/bank-marketing-dataset')
    except Exception as e:
        st.error(f"Error downloading dataset from Kaggle: {e}")
        st.stop()

    bank_csv = os.path.join(k_path, "bank.csv")
    try:
        df_raw = pd.read_csv(bank_csv, sep=',')
    except Exception:
        df_raw = pd.read_csv(bank_csv, sep=';')

    # The Kaggle dataset column is 'deposit'
    target_col = "deposit" if "deposit" in df_raw.columns else "y"

    df = df_raw.copy()
    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Encode target
    if df[target_col].dtype == 'object':
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Save test data using 'y' as the key for the dashboard dashboard code consistency
    test_df = X_test.copy()
    test_df['y'] = y_test.values
    test_df.to_csv(TEST_DATA_FILE, index=False)
    joblib.dump(scaler, SCALER_FILE)

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
    }

    for name, mdl in models_dict.items():
        mdl.fit(X_train_scaled, y_train)
        joblib.dump(mdl, MODEL_FILES[name])
    st.success("Training complete. Models saved!")

if not SCALER_FILE.exists() or not all(p.exists() for p in MODEL_FILES.values()):
    train_and_save_models()

scaler = joblib.load(SCALER_FILE)
model_names = list(MODEL_FILES.keys())
models = {name: joblib.load(MODEL_FILES[name]) for name in model_names}

uploaded = st.file_uploader("Upload test_data.csv (optional)", type=['csv'])
if uploaded:
    df_eval = pd.read_csv(uploaded)
else:
    if os.path.exists(TEST_DATA_FILE):
        df_eval = pd.read_csv(TEST_DATA_FILE)
        st.info("Using bundled test_data.csv.")
    else:
        st.error("Test data not found. Please train models first.")
        st.stop()

eval_target = "y" if "y" in df_eval.columns else "deposit"
X_eval = df_eval.drop(columns=[eval_target])
y_eval = df_eval[eval_target]
X_eval_scaled = scaler.transform(X_eval)

chosen = st.selectbox("Choose classifier", model_names)
model = models[chosen]
y_pred = model.predict(X_eval_scaled)
try:
    y_prob = model.predict_proba(X_eval_scaled)[:, 1]
except AttributeError:
    y_prob = y_pred

metrics = {
    "Accuracy": accuracy_score(y_eval, y_pred),
    "AUC": roc_auc_score(y_eval, y_prob),
    "Precision": precision_score(y_eval, y_pred),
    "Recall": recall_score(y_eval, y_pred),
    "F1 score": f1_score(y_eval, y_pred),
    "MCC": matthews_corrcoef(y_eval, y_pred),
}

c1, c2, c3 = st.columns(3)
for idx, (k, v) in enumerate(metrics.items()):
    (c1 if idx < 2 else c2 if idx < 4 else c3).metric(k, f"{v:.4f}")

st.subheader("Confusion matrix")
cm = confusion_matrix(y_eval, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
