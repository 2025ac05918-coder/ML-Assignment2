import os
import pathlib
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bank Marketing - ML Models", layout="wide")
st.title("🏦 Bank Marketing - ML Model Comparison Dashboard")
st.markdown("This application compares different classifiers on the Bank Marketing dataset.")

# --- SETUP DIRECTORIES ---
PROJECT_ROOT = pathlib.Path(os.getcwd())
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
TEST_DATA_FILE = PROJECT_ROOT / "test_data.csv"

SCALER_FILE = MODEL_DIR / "scaler.pkl"
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "Logistic_Regression.pkl",
    "Decision Tree": MODEL_DIR / "Decision_Tree.pkl",
    "kNN": MODEL_DIR / "kNN.pkl",
    "Naive Bayes": MODEL_DIR / "Naive_Bayes.pkl",
    "Random Forest": MODEL_DIR / "Random_Forest.pkl"
}

def train_and_save_models():
    if not KAGGLE_AVAILABLE:
        st.error("The 'kagglehub' library is missing. Please add it to requirements.txt.")
        st.stop()

    st.info("Training models... this may take a moment.")
    try:
        k_path = kagglehub.dataset_download('janiobachmann/bank-marketing-dataset')
        bank_csv = os.path.join(k_path, "bank.csv")
        df_raw = pd.read_csv(bank_csv, sep=',')
    except Exception as e:
        st.error(f"Dataset access error: {e}")
        st.stop()

    target_col = "deposit" if "deposit" in df_raw.columns else "y"
    df = df_raw.copy()

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

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
    st.success("Models trained successfully.")

# --- LOGIC ---
if not SCALER_FILE.exists() or not all(p.exists() for p in MODEL_FILES.values()):
    train_and_save_models()

scaler = joblib.load(SCALER_FILE)
model_names = list(MODEL_FILES.keys())
models = {name: joblib.load(MODEL_FILES[name]) for name in model_names}

uploaded = st.file_uploader("Upload test_data.csv", type=['csv'])
if uploaded:
    df_eval = pd.read_csv(uploaded)
elif TEST_DATA_FILE.exists():
    df_eval = pd.read_csv(TEST_DATA_FILE)
else:
    st.warning("No evaluation data found.")
    st.stop()

eval_target = "y" if "y" in df_eval.columns else "deposit"
df_eval = df_eval.copy()

# Standardize labels to integers 0 and 1
if df_eval[eval_target].dtype == 'object':
    mapping = {'no': 0, 'yes': 1, '0': 0, '1': 1, 'n': 0, 'y': 1}
    df_eval[eval_target] = df_eval[eval_target].str.lower().map(mapping).fillna(0).astype(int)

X_eval = df_eval.drop(columns=[eval_target])
y_eval = df_eval[eval_target]
X_eval_scaled = scaler.transform(X_eval)

chosen = st.selectbox("Select Model", model_names)
model = models[chosen]
y_pred = model.predict(X_eval_scaled)
try:
    y_prob = model.predict_proba(X_eval_scaled)[:, 1]
except:
    y_prob = y_pred

metrics = {
    "Accuracy": accuracy_score(y_eval, y_pred),
    "AUC": roc_auc_score(y_eval, y_prob),
    "Precision": precision_score(y_eval, y_pred, zero_division=0),
    "Recall": recall_score(y_eval, y_pred, zero_division=0),
    "F1": f1_score(y_eval, y_pred, zero_division=0)
}

cols = st.columns(len(metrics))
for i, (k, v) in enumerate(metrics.items()):
    cols[i].metric(k, f"{v:.3f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_eval, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
