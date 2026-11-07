import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")

bundle = load_model()
model, thr, feats = bundle["model"], bundle["threshold"], bundle["features"]
explainer = shap.TreeExplainer(model)

def prepare(df):
    df = df.copy()
    df["type"] = (df["type"] == "TRANSFER").astype(int)
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    if "hour" not in df.columns:
        df["hour"] = df["step"] % 24
    return df[feats]

def shap_plot(X):
    sv = explainer(X)
    vals = sv.values[0]
    base = float(sv.base_values[0])
    final = base + vals.sum()

    names = np.array([f"{f} = {X.iloc[0][f]:,.0f}" for f in X.columns])
    order = np.argsort(np.abs(vals))
    v_sorted = vals[order]
    colors = ["#ff0051" if v > 0 else "#008bfb" for v in v_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names[order], v_sorted, color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    for i, v in enumerate(v_sorted):
        ax.text(v, i, f" {v:+.2f} ", va="center",
                ha="left" if v > 0 else "right", fontsize=10)
    ax.margins(x=0.2)
    ax.set_xlabel("Impact on fraud score (SHAP value)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(f"Base score: {base:.2f}  →  Final score: {final:.2f}  "
               f"(positive = fraud, negative = legitimate)")

st.title("🛡️ Fraud Detection System")
tab1, tab2 = st.tabs(["Single Transaction", "Batch Analysis"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        t = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        amount = st.number_input("Amount", min_value=0.0, value=10000.0)
        hour = st.slider("Hour of Day", 0, 23, 12)
    with c2:
        old_o = st.number_input("Sender Balance Before", min_value=0.0, value=10000.0)
        new_o = st.number_input("Sender Balance After", min_value=0.0, value=0.0)
        old_d = st.number_input("Receiver Balance Before", min_value=0.0, value=0.0)
        new_d = st.number_input("Receiver Balance After", min_value=0.0, value=0.0)

    if st.button("Check Transaction"):
        row = pd.DataFrame([{
            "type": t, "amount": amount, "hour": hour,
            "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
            "oldbalanceDest": old_d, "newbalanceDest": new_d
        }])
        X = prepare(row)
        p = model.predict_proba(X)[0, 1]
        if p >= thr:
            st.error(f"⚠️ Suspicious Transaction — Fraud probability: {p:.1%}")
        else:
            st.success(f"✅ Legitimate Transaction — Fraud probability: {p:.1%}")

        st.subheader("Why did the model decide this?")
        st.caption("🔴 Red bars push the decision towards FRAUD  |  🔵 Blue bars push it towards LEGITIMATE. "
                   "Longer bars = stronger effect.")
        shap_plot(X)

with tab2:
    file = st.file_uploader("Upload a CSV file (PaySim format)", type="csv")
    if file:
        df = pd.read_csv(file)
        df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
        X = prepare(df)
        df["fraud_prob"] = model.predict_proba(X)[:, 1]
        df["flag"] = df["fraud_prob"] >= thr

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", len(df))
        c2.metric("Suspicious", int(df["flag"].sum()))
        c3.metric("Fraud Rate", f"{df['flag'].mean():.2%}")

        st.subheader("Suspicious Transactions")
        st.dataframe(df[df["flag"]].sort_values("fraud_prob", ascending=False))
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")