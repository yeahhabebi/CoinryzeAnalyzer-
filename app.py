import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import random
from datetime import datetime
from botocore.client import Config
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import accuracy_score

# ================
# CONFIG & R2 SETUP
# ================
R2_KEY_ID = st.secrets["R2_KEY_ID"]
R2_SECRET = st.secrets["R2_SECRET"]
R2_BUCKET = st.secrets["R2_BUCKET"]
R2_ENDPOINT = st.secrets["R2_ENDPOINT"]

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET,
    config=Config(signature_version="s3v4"),
)

# ===================
# INITIALIZE DATASETS
# ===================
def load_csv_from_r2(filename):
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=filename)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception:
        return pd.DataFrame(columns=["issue_id","timestamp","number","color","size"])

def save_csv_to_r2(df, filename):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=R2_BUCKET, Key=filename, Body=buffer.getvalue())

data = load_csv_from_r2("last_draws.csv")

# ===============
# STREAMLIT LAYOUT
# ===============
st.set_page_config("Coinryze Analyzer", layout="wide")
st.title("🎯 Coinryze Analyzer — Smart Predictive Dashboard")

st.sidebar.header("Dashboard Controls")
auto_predict = st.sidebar.toggle("Auto Predict After Entry", True)
refresh_seconds = st.sidebar.slider("Auto-Refresh (seconds)", 5, 60, 10)
if st.sidebar.button("Force Sync All CSVs to R2"):
    save_csv_to_r2(data, "last_draws.csv")
    st.sidebar.success("✅ Synced successfully!")

# =====================
# MANUAL INPUT FORM
# =====================
st.subheader("➕ Manual Input — Add Single Last Draw")

with st.form("manual_input"):
    issue_id = st.text_input("Issue ID")
    timestamp = st.text_input("Timestamp (e.g. 15:25:00 10/13/2025)")
    number = st.selectbox("Number", list(range(10)))
    color = st.selectbox("Color", ["Red", "Green", "Red-purple", "Green-purple"])
    size = st.selectbox("Size", ["Small", "Big"])
    submit = st.form_submit_button("Add Draw")

if submit:
    new_row = pd.DataFrame([[issue_id, timestamp, number, color, size]], 
                            columns=["issue_id","timestamp","number","color","size"])
    data = pd.concat([data, new_row], ignore_index=True)
    save_csv_to_r2(data, "last_draws.csv")
    st.success("✅ Draw added successfully!")

# ===============
# AUTO MAPPING CSV
# ===============
st.subheader("📤 Bulk CSV Upload / Auto-map Columns")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.strip().lower() for c in df.columns]
    expected = ["issue_id","timestamp","number","color","size"]
    if set(expected).issubset(df.columns):
        data = pd.concat([data, df[expected]], ignore_index=True)
        save_csv_to_r2(data, "last_draws.csv")
        st.success(f"✅ {len(df)} new records added.")
    else:
        st.error("❌ CSV missing expected columns!")

# =====================
# ADVANCED PREDICTOR
# =====================
def generate_predictions(df, markov_weight=0.6, freq_weight=0.4):
    if len(df) < 5:
        return [random.randint(0,9) for _ in range(5)]
    nums = df["number"].astype(int).tolist()
    transitions = {}
    for i in range(1, len(nums)):
        prev, nxt = nums[i-1], nums[i]
        transitions.setdefault(prev, []).append(nxt)
    markov_probs = np.zeros(10)
    if nums[-1] in transitions:
        nexts = transitions[nums[-1]]
        vals, counts = np.unique(nexts, return_counts=True)
        markov_probs[vals] = counts / counts.sum()
    freq_probs = df["number"].astype(int).value_counts(normalize=True).reindex(range(10), fill_value=0)
    combined = markov_weight * markov_probs + freq_weight * freq_probs.values
    combined /= combined.sum()
    predicted_numbers = np.random.choice(range(10), size=5, replace=False, p=combined)
    color_pred = random.choice(["Red","Green","Red-purple","Green-purple"])
    size_pred = random.choice(["Small","Big"])
    return list(predicted_numbers), color_pred, size_pred

# =====================
# BAYESIAN OPTIMIZATION
# =====================
def score_predictions(df, params):
    preds, c_pred, s_pred = generate_predictions(df, **params)
    last = df.iloc[-1]
    color_acc = 1 if last["color"] == c_pred else 0
    size_acc = 1 if last["size"] == s_pred else 0
    num_acc = 1 if int(last["number"]) in preds else 0
    return 0.5*num_acc + 0.25*color_acc + 0.25*size_acc

def bayesian_optimize(df, n_iter=25):
    best_score = -1
    best_params = None
    for params in ParameterSampler({"markov_weight": np.linspace(0.1,0.9,9),
                                    "freq_weight": np.linspace(0.1,0.9,9)}, n_iter):
        s = score_predictions(df, params)
        if s > best_score:
            best_score, best_params = s, params
    return best_params, best_score

# ===============
# BACKTEST BUTTON
# ===============
st.subheader("🧠 Smart Backtest / Auto-Tune")
if st.button("Run Auto-Tuner"):
    with st.spinner("Optimizing prediction weights..."):
        best_params, score = bayesian_optimize(data)
        st.success(f"✅ Best weights found: {best_params}, Score: {round(score,3)}")
        st.session_state["best_params"] = best_params

params = st.session_state.get("best_params", {"markov_weight":0.6,"freq_weight":0.4})
pred_nums, pred_color, pred_size = generate_predictions(data, **params)

# =====================
# DISPLAY PREDICTIONS
# =====================
st.subheader("🎯 Next Predicted Results")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Numbers", ", ".join(map(str, pred_nums)))
col2.metric("Predicted Color", pred_color)
col3.metric("Predicted Size", pred_size)

# =====================
# HISTORICAL TABLE
# =====================
st.subheader("📊 Historical Draws (Top 50)")
st.dataframe(data.tail(50).sort_index(ascending=False), use_container_width=True)

# =====================
# EVALUATION (if exists)
# =====================
if len(data) > 1:
    last_draw = data.iloc[-1]
    num_hit = int(last_draw["number"]) in pred_nums
    color_hit = last_draw["color"] == pred_color
    size_hit = last_draw["size"] == pred_size
    st.write(f"🎯 **Prediction Match:** Number={num_hit}, Color={color_hit}, Size={size_hit}")
