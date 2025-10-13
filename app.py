# ===========================
#  CoinryzeAnalyzer Full Smart Dashboard 🔮
#  Features:
#   - Manual & bulk input
#   - Smart predictive engine
#   - Auto-refresh
#   - Accuracy tracker with trend chart
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from io import StringIO
from datetime import datetime

# ===========================
# R2 Cloudflare Config
# ===========================
R2_KEY_ID = "YOUR_R2_KEY_ID"
R2_SECRET = "YOUR_R2_SECRET"
R2_BUCKET = "YOUR_BUCKET_NAME"
R2_ENDPOINT = "https://<account_id>.r2.cloudflarestorage.com"

s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET
)

# ===========================
# Utility: Load / Save CSV
# ===========================
def load_csv(filename):
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=filename)
        return pd.read_csv(obj["Body"])
    except Exception:
        return pd.DataFrame(columns=["issue_id", "timestamp", "number", "color", "size", "odd_even"])

def save_csv(df, filename):
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=R2_BUCKET, Key=filename, Body=buf.getvalue())

# ===========================
# SMART PREDICTION ENGINE 🔮
# ===========================
def smart_predict(df_last, n_predictions=5):
    """Predict next outcomes using weighted pattern analysis"""
    if df_last.empty:
        return pd.DataFrame(columns=df_last.columns)

    df = df_last.copy()
    preds = []

    color_freq = df["color"].value_counts(normalize=True).to_dict()
    size_freq = df["size"].value_counts(normalize=True).to_dict()
    oe_freq = df["odd_even"].value_counts(normalize=True).to_dict()

    recent = df.tail(10)
    last_color = recent["color"].iloc[-1]
    last_size = recent["size"].iloc[-1]
    last_oe = recent["odd_even"].iloc[-1]
    last_number = recent["number"].iloc[-1]

    color_streak = len(recent[recent["color"] == last_color])
    size_streak = len(recent[recent["size"] == last_size])
    oe_streak = len(recent[recent["odd_even"] == last_oe])

    for i in range(n_predictions):
        mean_num = df["number"].mean()
        drift = np.random.randint(-2, 3)
        next_number = int(abs(mean_num + drift) % 37)

        if color_streak >= 3:
            next_color = "Red" if last_color == "Black" else "Black"
        else:
            next_color = np.random.choice(["Red", "Black", "Green"], p=[
                color_freq.get("Red", 0.4),
                color_freq.get("Black", 0.4),
                color_freq.get("Green", 0.2)
            ])

        if size_streak >= 3:
            next_size = "Small" if last_size == "Big" else "Big"
        else:
            next_size = np.random.choice(["Small", "Big"], p=[
                size_freq.get("Small", 0.5),
                size_freq.get("Big", 0.5)
            ])

        if oe_streak >= 3:
            next_oe = "Odd" if last_oe == "Even" else "Even"
        else:
            next_oe = np.random.choice(["Odd", "Even"], p=[
                oe_freq.get("Odd", 0.5),
                oe_freq.get("Even", 0.5)
            ])

        try:
            next_issue = str(int(df["issue_id"].iloc[-1]) + i + 1)
        except:
            next_issue = f"{len(df)+i+1}"

        preds.append({
            "issue_id": next_issue,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "number": next_number,
            "color": next_color,
            "size": next_size,
            "odd_even": next_oe
        })

    return pd.DataFrame(preds)

# ===========================
# ACCURACY TRACKER 🎯
# ===========================
def calculate_accuracy(df_actual, df_pred):
    """Compare latest actual draws vs. predictions"""
    if df_actual.empty or df_pred.empty:
        return None

    n = min(len(df_actual), len(df_pred))
    df_a = df_actual.tail(n).reset_index(drop=True)
    df_p = df_pred.tail(n).reset_index(drop=True)

    total = n
    color_ok = (df_a["color"] == df_p["color"]).sum()
    size_ok = (df_a["size"] == df_p["size"]).sum()
    oe_ok = (df_a["odd_even"] == df_p["odd_even"]).sum()
    num_ok = ((df_a["number"] - df_p["number"]).abs() <= 1).sum()

    return {
        "color_acc": round(color_ok / total * 100, 1),
        "size_acc": round(size_ok / total * 100, 1),
        "oe_acc": round(oe_ok / total * 100, 1),
        "num_acc": round(num_ok / total * 100, 1),
        "sample": total
    }

# Log and update accuracy history for trend chart
def update_accuracy_log(acc_dict):
    df_log = load_csv("accuracy_log.csv")
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "color_acc": acc_dict["color_acc"],
        "size_acc": acc_dict["size_acc"],
        "oe_acc": acc_dict["oe_acc"],
        "num_acc": acc_dict["num_acc"]
    }
    df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    save_csv(df_log, "accuracy_log.csv")
    return df_log

# ===========================
# STREAMLIT UI
# ===========================
st.set_page_config(page_title="CoinryzeAnalyzer", layout="wide")
st.title("🔮 CoinryzeAnalyzer Smart Dashboard")

refresh_sec = st.sidebar.number_input("Auto-refresh (seconds)", 5, 300, 10)
st.sidebar.info("Page auto-refreshes automatically.")

# --- Bulk Upload ---
st.subheader("📥 Add Last Draw Results (Bulk)")
uploaded = st.file_uploader("Upload CSV (issue_id,timestamp,number,color,size,odd_even)", type=["csv"])
if uploaded:
    df_upload = pd.read_csv(uploaded)
    df_last = load_csv("last_draws.csv")
    df_last = pd.concat([df_last, df_upload], ignore_index=True)
    save_csv(df_last, "last_draws.csv")
    st.success(f"✅ {len(df_upload)} new rows added!")

# --- Manual Input ---
st.subheader("✏️ Add Single Last Draw")
with st.form("add_draw"):
    issue_id = st.text_input("Issue ID")
    number = st.number_input("Number", 0, 36)
    color = st.selectbox("Color", ["Red", "Black", "Green"])
    size = st.selectbox("Size", ["Small", "Big"])
    odd_even = st.selectbox("Odd/Even", ["Odd", "Even"])
    submit = st.form_submit_button("➕ Add Draw")
if submit:
    df_last = load_csv("last_draws.csv")
    row = {
        "issue_id": issue_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "number": number,
        "color": color,
        "size": size,
        "odd_even": odd_even
    }
    df_last = pd.concat([df_last, pd.DataFrame([row])], ignore_index=True)
    save_csv(df_last, "last_draws.csv")
    st.success("✅ Draw added successfully!")

# --- Display Draws ---
st.subheader("📊 Last Draws Table")
df_last = load_csv("last_draws.csv")
st.dataframe(df_last)

# --- Smart Predictions ---
st.subheader("🤖 Smart Predicted Outcomes")
num_pred = st.slider("Number of future predictions", 1, 20, 5)
df_pred = smart_predict(df_last, n_predictions=num_pred)
save_csv(df_pred, "predictions.csv")
st.dataframe(df_pred)

# --- Accuracy Tracker ---
st.subheader("🎯 Prediction Accuracy Tracker")
df_pred = load_csv("predictions.csv")
acc = calculate_accuracy(df_last, df_pred)

if acc:
    st.markdown(f"""
    **Based on last {acc['sample']} draws**
    - 🎨 Color Accuracy: **{acc['color_acc']}%**
    - ⚖️ Size Accuracy: **{acc['size_acc']}%**
    - 🔢 Odd/Even Accuracy: **{acc['oe_acc']}%**
    - 🎯 Number (±1) Accuracy: **{acc['num_acc']}%**
    """)
    df_log = update_accuracy_log(acc)
else:
    st.info("Not enough data yet to calculate accuracy.")
    df_log = load_csv("accuracy_log.csv")

# --- Accuracy Trend Chart ---
if not df_log.empty:
    st.subheader("📈 Accuracy Trend Over Time")
    st.line_chart(df_log.set_index("timestamp")[["color_acc", "size_acc", "oe_acc", "num_acc"]])

st.caption(f"🔁 Auto-refreshes every {refresh_sec} seconds to update predictions & accuracy.")
st.experimental_rerun()
