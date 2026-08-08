# ⬇️ Colab-only: quiet installs so the log is small
import os, textwrap, json, subprocess, pathlib, shutil, getpass

ROOT = pathlib.Path('/content/bank_marketing_project')
MODEL_DIR = ROOT / 'model'
ROOT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
print(f"Project root: {ROOT}")

import pandas as pd, numpy as np, kagglehub, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# download once – cached by kagglehub
k_path = kagglehub.dataset_download('janiobachmann/bank-marketing-dataset')
bank_csv  = os.path.join(k_path, 'bank.csv')
# Fixed: Changed sep=';' to sep=',' as per the raw data state
df_raw = pd.read_csv(bank_csv, sep=',')

# encode categoricals
df = df_raw.copy()
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Fixed: The target column in this dataset is 'deposit', not 'y'
target_col = 'deposit'
X, y = df.drop(target_col, axis=1), df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y)

# scale numeric features
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# save test split for Streamlit upload fallback
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['y'] = y_test.values
test_df.to_csv(ROOT / 'test_data.csv', index=False)

joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
print("Dataset ready ✓")

from sklearn.linear_model import LogisticRegression
from sklearn.tree   import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'kNN'                : KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes'        : GaussianNB(),
    'Random Forest'      : RandomForestClassifier(n_estimators=150, random_state=42)
}

metrics_table = []

for name, mdl in models.items():
    mdl.fit(X_train_s, y_train)
    joblib.dump(mdl, MODEL_DIR / f"{name.replace(' ','_')}.pkl")
    
    y_pred = mdl.predict(X_test_s)
    try:         y_prob = mdl.predict_proba(X_test_s)[:,1]
    except:      y_prob = y_pred
    
    metrics_table.append([
        name,
        accuracy_score(y_test,y_pred),
        roc_auc_score(y_test,y_prob),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred),
        matthews_corrcoef(y_test,y_pred)
    ])

print(pd.DataFrame(metrics_table,
      columns=['Model','Acc','AUC','Prec','Recall','F1','MCC']))
print("Models trained & saved ✓")

APP_CODE = r"""
import streamlit as st, pandas as pd, joblib, os, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)

st.set_page_config(page_title="Bank Marketing Models", layout="wide")
st.title("🏦 Bank Marketing - ML Model Comparison App")

# -------- Load trained models ------------------------------------------------
MODEL_DIR = 'model'
model_names = ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
models = {n: joblib.load(os.path.join(MODEL_DIR,f"{n.replace(' ','_')}.pkl")) for n in model_names}
scaler = joblib.load(os.path.join(MODEL_DIR,'scaler.pkl'))

# -------- Data upload --------------------------------------------------------
uploaded = st.file_uploader("Upload test_data.csv (optional)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv('test_data.csv')
    st.info("Using bundled test_data.csv. Upload to override.")

X = df.drop('y', axis=1)
y = df['y']
X_scaled = scaler.transform(X)  # ensure scaling identical to training

# -------- Model selection ----------------------------------------------------
chosen = st.selectbox("Select a model", model_names)
model  = models[chosen]

# -------- Predictions & Metrics ---------------------------------------------
y_pred = model.predict(X_scaled)
try:    y_prob = model.predict_proba(X_scaled)[:,1]
except: y_prob = y_pred

metrics = {
    "Accuracy" : accuracy_score(y,y_pred),
    "AUC"      : roc_auc_score(y,y_prob),
    "Precision": precision_score(y,y_pred),
    "Recall"   : recall_score(y,y_pred),
    "F1 Score" : f1_score(y,y_pred),
    "MCC"      : matthews_corrcoef(y,y_pred)
}

c1,c2,c3 = st.columns(3)
for idx,(k,v) in enumerate(metrics.items()):
    (c1 if idx<2 else c2 if idx<4 else c3).metric(k, f"{v:.4f}")

# -------- Confusion Matrix ---------------------------------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y,y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
st.pyplot(fig)
"""
(Path := pathlib.Path('/content/bank_marketing_project/app.py')).write_text(APP_CODE)
print(f"app.py written to {Path}")
