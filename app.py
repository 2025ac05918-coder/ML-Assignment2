import os
import pathlib
import shutil
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bank Marketing – ML Models", layout="wide")
st.title("   Bank Marketing – ML Model Comparison Dashboard")
st.markdown("This app downloads data, trains classifiers (if needed), and evaluates them on test data. This was prepared by Punitha S. 2025ac05918-ML-Assignment-2")

PROJECT_ROOT = pathlib.Path(os.getcwd())
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
TEST_DATA_FILE = PROJECT_ROOT / "test_data.csv"
SCALER_FILE = MODEL_DIR / "scaler.py"
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "Logistic_Regression.py",
    "Decision Tree": MODEL_DIR / "Decision_Tree.py",
    "kNN": MODEL_DIR / "kNN.py",
    "Naive Bayes": MODEL_DIR / "Naive_Bayes.py",
    "Random Forest": MODEL_DIR / "Random_Forest.py"
}

def train_and_save_models():
    if not KAGGLE_AVAILABLE:
        st.error("The 'kagglehub' library is missing. Please add it to requirements.txt.")
        st.stop()

    st.info("No pre-trained artifacts found – training models now...")
    try:
        k_path = kagglehub.dataset_download('janiobachmann/bank-marketing-dataset')
        bank_csv = os.path.join(k_path, "bank.csv")
        df_raw = pd.read_csv(bank_csv, sep=',' if 'janiobachmann' in k_path else ';')
    except Exception as e:
        st.error(f"Dataset access error: {e}")
        st.stop()

    target_col = "deposit" if "deposit" in df_raw.columns else "y"
    df = df_raw.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, SCALER_FILE)
    
    test_df = X_test.copy()
    test_df["y"] = y_test.values
    test_df.to_csv(TEST_DATA_FILE, index=False)

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
    }

    for name, mdl in models_dict.items():
        mdl.fit(scaler.transform(X_train), y_train)
        joblib.dump(mdl, MODEL_FILES[name])
    st.success("Training complete!")

if not SCALER_FILE.exists() or not all(f.exists() for f in MODEL_FILES.values()):
    train_and_save_models()

@st.cache_resource
def load_artifacts():
    sc = joblib.load(SCALER_FILE)
    mdls = {name: joblib.load(MODEL_FILES[name]) for name in MODEL_FILES.keys()}
    return mdls, sc

models, scaler = load_artifacts()

st.subheader("Data Upload")
uploaded = st.file_uploader("Upload test_data.csv (optional)", type=['csv'])
df_eval = pd.read_csv(uploaded) if uploaded else pd.read_csv(TEST_DATA_FILE)

eval_target = "y" if "y" in df_eval.columns else "deposit"
def clean_labels(val):
    v = str(val).lower().strip()
    return 1 if v in ['yes', 'y', '1', '1.0'] else 0

df_eval[eval_target] = df_eval[eval_target].apply(clean_labels)
X_eval = scaler.transform(df_eval.drop(eval_target, axis=1))
y_true = df_eval[eval_target]

chosen_model = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
model = models[chosen_model]
y_pred = model.predict(X_eval)
try:
    y_prob = model.predict_proba(X_eval)[:, 1]
except:
    y_prob = y_pred

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_prob),
    "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred)
}

cols = st.columns(6)
for i, (k, v) in enumerate(metrics.items()):
    cols[i].metric(k, f"{v:.3f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
