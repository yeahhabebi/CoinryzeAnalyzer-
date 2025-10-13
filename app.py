# app.py — Coinryze Analyzer (Markov Upgrade)

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import datetime
import json

# --------------------------
# Load secrets from .streamlit/secrets.toml
# --------------------------
R2_KEY_ID = st.secrets["R2_KEY_ID"]
R2_SECRET = st.secrets["R2_SECRET"]
R2_BUCKET = st.secrets["R2_BUCKET"]
R2_ENDPOINT = st.secrets["R2_ENDPOINT"]

F_LAST_DRAWS = "last_draws.csv"
F_PREDICTIONS = "predictions.csv"
F_ACCURACY = "accuracy_log.csv"

# --------------------------
# S3 / R2 client
# --------------------------
s3 = boto3.client(
    "s3",
    region_name="auto",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET,
)

# --------------------------
# Helpers for R2
# --------------------------
def r2_get_bytes(key):
    try:
        resp = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return resp["Body"].read()
    except ClientError:
        return None

def r2_put_bytes(key, bts, content_type="text/csv"):
    try:
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=bts, ContentType=content_type)
        return True
    except ClientError as e:
        st.error(f"R2 upload error: {e}")
        return False

def r2_list_keys(prefix=""):
    try:
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]
    except ClientError:
        return []

# --------------------------
# Load data or create empty
# --------------------------
DRAW_COLS = ["issue_id", "timestamp", "number", "color", "size"]

def load_df(key, cols=None):
    b = r2_get_bytes(key)
    if b:
        try:
            return pd.read_csv(BytesIO(b), dtype=str)
        except Exception:
            return pd.read_csv(StringIO(b.decode("utf-8")), dtype=str)
    return pd.DataFrame(columns=cols or [])

last_draws = load_df(F_LAST_DRAWS, DRAW_COLS)
predictions_df = load_df(F_PREDICTIONS, ["created_at", "prediction", "pick_count"])
accuracy_log = load_df(F_ACCURACY, ["timestamp", "predicted", "actual", "accuracy_pct"])

# --------------------------
# Utility: color & size inference
# --------------------------
def infer_size(n):
    try:
        n = int(n)
        return "Small" if 0 <= n <= 4 else "Big"
    except:
        return ""

def infer_color(n):
    try:
        n = int(n)
        if n == 0:
            return "Red-purple"
        if n == 5:
            return "Green-purple"
        return "Red" if 0 <= n <= 4 else "Green"
    except:
        return ""

# --------------------------
# Advanced Prediction (Markov + Frequency)
# --------------------------
def predict_next_numbers(df, pick_count=1):
    if df.empty:
        return []
    nums = df["number"].dropna().astype(int).tolist()
    if len(nums) < 2:
        return []

    # Build Markov transition counts
    transitions = {i: {j: 0 for j in range(10)} for i in range(10)}
    for a, b in zip(nums[:-1], nums[1:]):
        transitions[a][b] += 1

    # Normalize into probabilities
    probs = {i: {j: c / sum(row.values()) if sum(row.values()) > 0 else 0 for j, c in row.items()} for i, row in transitions.items()}

    # Get last number and compute likely nexts
    last_num = nums[-1]
    markov_scores = probs.get(last_num, {})

    # Combine Markov + frequency weighting
    freq = pd.Series(nums).value_counts(normalize=True).to_dict()
    combined = {n: (markov_scores.get(n, 0) * 0.7 + freq.get(n, 0) * 0.3) for n in range(10)}

    sorted_nums = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nums[:pick_count]]

# --------------------------
# Parse pasted line
# --------------------------
def parse_line_to_row(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    while len(parts) < 5:
        parts.append("")
    issue_id, timestamp, number, color, size = parts[:5]
    return {
        "issue_id": issue_id or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "timestamp": timestamp or datetime.datetime.utcnow().isoformat(),
        "number": str(int(float(number))) if number else "",
        "color": color or infer_color(number),
        "size": size or infer_size(number),
    }

# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="Coinryze Analyzer (Markov Upgrade)", layout="wide")
st.title("🎯 Coinryze Analyzer — Advanced Markov Prediction Dashboard")

st.sidebar.header("⚙️ Settings")
pick_count = st.sidebar.number_input("How many numbers to predict", 1, 5, 1)
auto_predict = st.sidebar.checkbox("Auto-predict after adding draw", True)
show_debug = st.sidebar.checkbox("Show debug info", False)

# --------------------------
# Manual Input Section (Dropdowns)
# --------------------------
st.header("🧩 Manual Input — Add Single Draw")
with st.form("single_draw_form"):
    c1, c2, c3, c4, c5 = st.columns([1.5, 2, 1, 1, 1])
    with c1:
        issue_id = st.text_input("Issue ID", placeholder="e.g. 202510141230")
    with c2:
        timestamp = st.text_input("Timestamp", value=datetime.datetime.utcnow().strftime("%H:%M:%S %m/%d/%Y"))
    with c3:
        number = st.selectbox("Number (0–9)", list(range(10)))
    with c4:
        color = st.selectbox("Color", ["Red", "Green", "Red-purple", "Green-purple"])
    with c5:
        size = st.selectbox("Size", ["Small", "Big"])
    submitted = st.form_submit_button("Add Draw")

if submitted:
    new_row = {
        "issue_id": issue_id or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "timestamp": timestamp,
        "number": str(number),
        "color": color,
        "size": size,
    }
    last_draws = pd.concat([pd.DataFrame([new_row]), last_draws], ignore_index=True)
    r2_put_bytes(F_LAST_DRAWS, last_draws.to_csv(index=False).encode("utf-8"))
    st.success("✅ Draw added and synced to R2.")
    if auto_predict:
        picks = predict_next_numbers(last_draws, pick_count)
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count}
        predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
        r2_put_bytes(F_PREDICTIONS, predictions_df.to_csv(index=False).encode("utf-8"))
        st.info(f"🔮 Auto Prediction: {picks}")
    st.rerun()

# --------------------------
# Paste Section (with auto-map button)
# --------------------------
st.write("---")
st.subheader("📋 Paste multiple rows (any column order)")
pasted = st.text_area("Paste CSV lines (any header order, auto-mappable)", height=120)
if st.button("Auto-map & Add"):
    lines = [l.strip() for l in pasted.splitlines() if l.strip()]
    added = 0
    for line in lines:
        parsed = parse_line_to_row(line)
        if parsed:
            last_draws = pd.concat([pd.DataFrame([parsed]), last_draws], ignore_index=True)
            added += 1
    if added:
        r2_put_bytes(F_LAST_DRAWS, last_draws.to_csv(index=False).encode("utf-8"))
        st.success(f"Added {added} rows and synced to R2.")
        if auto_predict:
            picks = predict_next_numbers(last_draws, pick_count)
            st.info(f"Auto Prediction: {picks}")
        st.rerun()
    else:
        st.warning("No valid rows found — check format.")

# --------------------------
# Display History & Predictions
# --------------------------
st.write("---")
st.subheader("📜 Historical Draws (latest 50)")
st.dataframe(last_draws.head(50))

st.write("---")
st.subheader("🔮 Smart Predictions (Markov Enhanced)")
if last_draws.empty:
    st.info("No data yet to predict.")
else:
    picks = predict_next_numbers(last_draws, pick_count)
    st.success(f"Predicted Next Numbers: {picks}")
    st.table(pd.DataFrame([{"number": p, "color": infer_color(p), "size": infer_size(p)} for p in picks]))

if st.button("Save This Prediction"):
    rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count}
    predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
    r2_put_bytes(F_PREDICTIONS, predictions_df.to_csv(index=False).encode("utf-8"))
    st.success("Prediction saved and synced to R2.")

# --------------------------
# Prediction History
# --------------------------
st.write("---")
st.subheader("📈 Predictions History (latest 20)")
if predictions_df.empty:
    st.info("No predictions saved yet.")
else:
    st.dataframe(predictions_df.head(20))

# --------------------------
# Accuracy Logging
# --------------------------
st.write("---")
st.subheader("🎯 Evaluate Last Prediction vs Latest Draw")
if not predictions_df.empty and not last_draws.empty:
    last_pred = json.loads(predictions_df.iloc[0]["prediction"])
    latest = int(last_draws.iloc[0]["number"])
    overlap = len(set(last_pred) & {latest}) / len(last_pred) * 100
    st.write(f"Overlap Accuracy: {overlap:.1f}%")
    if st.button("Log Accuracy"):
        rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(last_pred), "actual": json.dumps([latest]), "accuracy_pct": overlap}
        accuracy_log = pd.concat([pd.DataFrame([rec]), accuracy_log], ignore_index=True)
        r2_put_bytes(F_ACCURACY, accuracy_log.to_csv(index=False).encode("utf-8"))
        st.success("Accuracy logged and synced to R2.")
else:
    st.info("Need at least one prediction and one draw to evaluate.")

# --------------------------
# R2 Utilities
# --------------------------
st.write("---")
st.subheader("🗂️ R2 Bucket Files")
keys = r2_list_keys()
if keys:
    for k in keys:
        st.write("•", k)
else:
    st.info("No files found in bucket.")

if show_debug:
    st.sidebar.write("DEBUG →", {"last_draws": last_draws.shape, "predictions": predictions_df.shape, "accuracy": accuracy_log.shape})
